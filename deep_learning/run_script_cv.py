# Script for running a full train and validation run
from common_utils import train_and_validate, get_device
from data_prep import get_preprocessed_data
from models import SNLI_GRU_Interaction

from torch.optim import Adam
from torch.nn import Sequential, Linear, ReLU, Dropout
import pickle

# Hyperparameter setting

hidden_size = 200
layers = 3
inp_size = hidden_size * layers * 2 * 2  # bidirectional * (premise+hypothesis)
first_hid_size = int(inp_size / 2)
sec_hid_size = int(inp_size / 4)
learning_rate = 1e-3

seq = Sequential(
    Linear(inp_size, first_hid_size),
    ReLU(),
    Dropout(0.2),
    Linear(first_hid_size, sec_hid_size),
    ReLU(),
    Dropout(0.2),
    Linear(sec_hid_size, 3),
)

result = {
    'model': 'gru-interaction',

    'optimiser': 'adam',
    'learning_rates': learning_rate,

    'regularisation': 'dropout-0.2',

    'recurrent_hidden': hidden_size,
    'recurrent_layers': layers,

    'layers_nodes': 'ip/2-act-drop-ip/4-act-drop-pred',
    'activations': 'relu',

    'embedding': 'w2v',
}

train_data, _ = get_preprocessed_data()

train_loss_curves, train_acc_curves, test_loss_curves, test_acc_curves = train_and_validate(
    data_df=train_data, device=get_device(), seed=0, num_epochs=100, bs=1024, model_class=SNLI_GRU_Interaction,
    embedding_type=0, embedding_dim=300,
    hidden_size=hidden_size, layers=layers,
    feed_forward_model=seq,
    optimiser_class=Adam, optimiser_kwargs={'lr': learning_rate})

result['train_loss_curves'] = train_loss_curves
result['train_acc_curves'] = train_acc_curves
result['test_loss_curves'] = test_loss_curves
result['test_acc_curves'] = test_acc_curves

with open('/kaggle/working/gru-interaction_adam_0.001_dropout-0.2_200_3_ip2-act-drop-ip4-act-drop-pred_relu_w2v.pkl', 'wb') as f:
    pickle.dump(result, f, protocol=pickle.DEFAULT_PROTOCOL)
