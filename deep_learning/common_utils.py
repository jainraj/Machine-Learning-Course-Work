import torch
from torch.nn import Embedding, init
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss
from time import time
from math import fabs
from sklearn.model_selection import StratifiedKFold
import numpy
import random
import gensim.downloader

__all__ = ['get_device', 'train_and_test']


def reset_seeds(s):
    """Set all seeds to some value to ensure reproducibility"""
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    random.seed(s)
    numpy.random.seed(s)


def get_device():
    if torch.cuda.is_available():  # Check if we are GPU or CPU and use appropriate code
        device = torch.device("cuda")
        print("We are on GPU :)")
    else:
        device = torch.device("cpu")
        print("We are on CPU :(")
    return device


def to_device(data, dest_device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, dest_device) for x in data]
    return data.to(dest_device, non_blocking=True)


class DeviceDataLoader(object):
    """Wrapper over DataLoader for ease of usage"""

    def __init__(self, dl, dest_device):
        self.dl = dl
        self.device = dest_device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


def train_epoch_batch(model, data, optimiser):
    """For one batch in one training epoch"""
    premise, premise_length, hypothesis, hypothesis_length, label_target = data
    label_pred = model(premise, premise_length, hypothesis, hypothesis_length)
    loss = CrossEntropyLoss()(label_pred, label_target)
    loss.backward()  # compute updates for each parameter
    optimiser.step()  # make the updates for each parameter
    optimiser.zero_grad()  # cleanup step
    _, label_argmax = torch.max(label_pred.data, 1)
    correct = (label_argmax == label_target).sum()
    n = label_target.shape[0]
    return loss.detach().item() * n, n, correct.detach().item()


def test_epoch_batch(model, data):
    """For one batch in one testing epoch"""
    premise, premise_length, hypothesis, hypothesis_length, label_target = data
    label_pred = model(premise, premise_length, hypothesis, hypothesis_length)
    loss = CrossEntropyLoss()(label_pred, label_target)
    _, label_argmax = torch.max(label_pred.data, 1)
    correct = (label_argmax == label_target).sum()
    n = label_target.shape[0]
    return loss.detach().item() * n, n, correct.detach().item()


def get_tensors(section_df):
    """Get tensors that will be fed into a TensorDataset"""
    return [
        torch.tensor(section_df.premise.tolist(), dtype=torch.long),
        torch.tensor(section_df.premise_length.tolist(), dtype=torch.long),
        torch.tensor(section_df.hypothesis.tolist(), dtype=torch.long),
        torch.tensor(section_df.hypothesis_length.tolist(), dtype=torch.long),
        torch.tensor(section_df.label.tolist(), dtype=torch.long),
    ]


def train_epoch(model, train_dataloader, optimiser):
    """For one training epoch"""
    model.train()  # set mode to training
    batch_train_loss, num_samples, corrects = [], [], []
    for data in train_dataloader:
        loss, n, correct = train_epoch_batch(model, data, optimiser)
        batch_train_loss.append(loss), num_samples.append(n), corrects.append(correct)

    epoch_train_loss = sum(batch_train_loss) / sum(num_samples)
    epoch_train_accuracy = sum(corrects) / sum(num_samples)
    return epoch_train_loss, epoch_train_accuracy


def test_epoch(model, test_dataloader):
    """For one testing epoch"""
    model.eval()  # set mode to evaluation
    batch_test_loss, num_samples, corrects = [], [], []
    with torch.no_grad():  # speed up computation for evaluation
        for data in test_dataloader:
            loss, n, correct = test_epoch_batch(model, data)
            batch_test_loss.append(loss), num_samples.append(n), corrects.append(correct)

    epoch_test_loss = sum(batch_test_loss) / sum(num_samples)
    epoch_test_accuracy = sum(corrects) / sum(num_samples)
    return epoch_test_loss, epoch_test_accuracy


def train_and_test_model(model, train_dataloader, test_dataloader, num_epochs, optimiser_class, optimiser_kwargs):
    """Train and test model for all epochs"""
    train_loss, test_loss = [], []
    train_acc, test_acc = [], []
    optimiser = optimiser_class(model.parameters(), **optimiser_kwargs)

    prev_epoch_test_loss = numpy.inf

    for epoch in range(num_epochs):
        start = time()
        epoch_train_loss, epoch_train_accuracy = train_epoch(model, train_dataloader, optimiser)
        train_loss.append(epoch_train_loss), train_acc.append(epoch_train_accuracy)

        epoch_test_loss, epoch_test_accuracy = test_epoch(model, test_dataloader)
        test_loss.append(epoch_test_loss), test_acc.append(epoch_test_accuracy)

        print(f'[{epoch + 1:03d}] Training Loss:{epoch_train_loss:9.5f} Acc:{epoch_train_accuracy:9.5f}'
              f' - Validation Loss:{epoch_test_loss:9.5f} Acc:{epoch_test_accuracy:9.5f}'
              f'   |   Epoch Time Lapsed:{time() - start:7.3f} sec')

        # if fabs(prev_epoch_test_loss - epoch_test_loss) < threshold:
        #     print("Validation loss hasn't improved, stopping!")
        #     break
        # else:
        #     prev_epoch_test_loss = epoch_test_loss

    return train_loss, train_acc, test_loss, test_acc


def get_dataloader(section_df, bs):
    """Get DataLoader object for a dataframe"""
    section_dataset = TensorDataset(*get_tensors(section_df))
    return DataLoader(section_dataset, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)


def get_dataloaders(train_df, test_df, bs):
    """Get train and test DataLoader objects"""
    train_dataloader = get_dataloader(train_df, bs)
    test_dataloader = get_dataloader(test_df, bs)
    return train_dataloader, test_dataloader


def process_sent_type(sent_type, train_df, test_df, embedding_type, embedding_dim):
    """
    For a given sentence type i.e., premise or hypothesis, we do the following
    1. Get all the tokens from that type in train data and form a label mapping (2 to size+1)
    2. Convert tokens in test data using the label mapping - if it doesn't exist, we use a special number 1
    3. Not all sentences will be of same length - we will pad with a special number 0
    4. Create embedding layer based on embedding_type param but keep zeros for 0-index and some random vector for 1-index (our special indices)
    5. The padding_idx will be kept to 0 as that indicates padding
    """
    train_df, test_df = train_df.copy(), test_df.copy()
    # Convert tokens to numbers
    all_tokens = set(w for l in train_df[f'{sent_type}_tokens'] for w in l)
    vocab = {w: i + 2 for i, w in enumerate(all_tokens)}
    train_df[sent_type] = train_df[f'{sent_type}_tokens'].apply(lambda l: [vocab[w] for w in l])
    test_df[sent_type] = test_df[f'{sent_type}_tokens'].apply(lambda l: [vocab.get(w, 1) for w in l])  # 1 for unknown token

    # Get max length and pad with 0's
    train_df[f'{sent_type}_length'] = train_df[sent_type].apply(len)
    test_df[f'{sent_type}_length'] = test_df[sent_type].apply(len)

    max_train_length = train_df[f'{sent_type}_length'].max()
    train_df[sent_type] = train_df[[sent_type, f'{sent_type}_length']].apply(lambda row: row[sent_type] + [0] * (max_train_length - row[f'{sent_type}_length']), axis=1)
    max_test_length = test_df[f'{sent_type}_length'].max()
    test_df[sent_type] = test_df[[sent_type, f'{sent_type}_length']].apply(lambda row: row[sent_type] + [0] * (max_test_length - row[f'{sent_type}_length']), axis=1)

    # Create embedding layer
    num_embeddings = len(all_tokens) + 2

    if embedding_type == 0 or embedding_type == 1:  # 'pre-trained-untrainable' or 'pre-trained-trainable'
        embeddings_matrix = numpy.zeros((num_embeddings, 300))  # 0-index gets a zero vector
        word2vec_vectors = gensim.downloader.load('word2vec-google-news-300')

        embeddings_matrix[1] = numpy.random.uniform(-0.05, 0.05, size=(300,))  # 1-index gets a random vector
        for i, w in enumerate(all_tokens):
            word_idx = i + 2
            if w in word2vec_vectors.wv:
                embeddings_matrix[word_idx] = word2vec_vectors.wv[w]  # tokens present in pre-trained get the respective vector
            else:
                embeddings_matrix[word_idx] = numpy.random.uniform(-0.05, 0.05, size=(300,))  # tokens not present get a random vector

        embedding = Embedding(num_embeddings, 300, padding_idx=0)
        embedding.load_state_dict({'weight': torch.tensor(embeddings_matrix)})
        embedding.weight.requires_grad = embedding_type == 1
    else:  # 'untrained'
        embedding = Embedding(num_embeddings, embedding_dim, padding_idx=0)
        init.uniform_(embedding.weight, -0.05, 0.05)
        embeddings_matrix = embedding.weight.detach().numpy()
        embeddings_matrix[0] = numpy.zeros(shape=(embedding_dim,))
        embedding.load_state_dict({'weight': torch.tensor(embeddings_matrix)})
        embedding.weight.requires_grad = True

    return train_df, test_df, embedding, vocab


def train_and_test(data_df, device, seed, num_epochs, bs, model_class,
                   embedding_type, embedding_dim,
                   hidden_size, layers,
                   feed_forward_model,
                   optimiser_class, optimiser_kwargs):
    """Get loss and accuracy curves for a given set of hyper-parameters using a cross-validation technique"""
    train_loss_curves, test_loss_curves = {}, {}
    train_acc_curves, test_acc_curves = {}, {}

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    for idx, (train_index, test_index) in enumerate(skf.split(data_df, data_df.label)):
        print(f"Starting CV fold {idx + 1}")
        train_df, test_df = data_df.iloc[train_index], data_df.iloc[test_index]

        reset_seeds(seed)  # for fair comparison - all models will start from the same initial point
        train_df, test_df, premise_embedding, premise_vocab = process_sent_type('premise', train_df, test_df, embedding_type, embedding_dim)
        train_df, test_df, hypothesis_embedding, hypothesis_vocab = process_sent_type('hypothesis', train_df, test_df, embedding_type, embedding_dim)
        model = model_class(premise_embedding, hypothesis_embedding,
                            hidden_size, layers,
                            feed_forward_model)
        to_device(model, device)

        reset_seeds(seed + idx)  # for reproducibility
        train_dataloader, test_dataloader = get_dataloaders(train_df, test_df, bs)
        train_dataloader, test_dataloader = DeviceDataLoader(train_dataloader, device), DeviceDataLoader(test_dataloader, device)

        train_loss_curves[idx], train_acc_curves[idx], test_loss_curves[idx], test_acc_curves[idx] = \
            train_and_test_model(model, train_dataloader, test_dataloader, num_epochs, optimiser_class, optimiser_kwargs)

    return train_loss_curves, train_acc_curves, test_loss_curves, test_acc_curves
