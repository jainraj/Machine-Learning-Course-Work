import torch
from torch.nn import init, Module, LSTM, Sequential
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SNLI_LSTM(Module):
    """Define the deep learning model"""

    def __init__(self,
                 premise_embedding, hypothesis_embedding,
                 premise_hidden_size, premise_layers,
                 hypothesis_hidden_size, hypothesis_layers,
                 feed_forward_model: Sequential):
        super(SNLI_LSTM, self).__init__()

        self.premise_embedding = premise_embedding
        self.hypothesis_embedding = hypothesis_embedding

        self.premise_LSTM = LSTM(input_size=premise_embedding.embedding_dim, hidden_size=premise_hidden_size, num_layers=premise_layers, batch_first=True, bidirectional=True)
        self.hypothesis_LSTM = LSTM(input_size=hypothesis_embedding.embedding_dim, hidden_size=hypothesis_hidden_size, num_layers=hypothesis_layers, batch_first=True, bidirectional=True)

        self.feed_forward_model = feed_forward_model  # a Sequential model

        # Initialisations
        for name, param in self.premise_LSTM.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                init.orthogonal_(param)
            elif 'bias_ih_l0' == name:
                init.zeros_(param)
        for name, param in self.hypothesis_LSTM.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                init.orthogonal_(param)
            elif 'bias_ih_l0' == name:
                init.zeros_(param)
        for name, param in self.feed_forward_model.named_parameters():
            if 'weight' in param:
                init.xavier_uniform_(param)
            if 'bias' in param:
                init.zeros_(param)
        with torch.no_grad():
            self.premise_LSTM.bias_hh_l0.data = torch.tensor([0] * premise_hidden_size + [1] * premise_hidden_size + [0] * premise_hidden_size * 2).float()
            self.hypothesis_LSTM.bias_hh_l0.data = torch.tensor([0] * hypothesis_hidden_size + [1] * hypothesis_hidden_size + [0] * hypothesis_hidden_size * 2).float()

        self.double()

    def __get_lstm_output(self, lstm, lstm_inp, lengths):  # lstm_inp is of shape (batch_size, seq_length, embedding_size)
        packed_input = pack_padded_sequence(lstm_inp, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = lstm(packed_input)  # output is of shape (batch_size, seq_length, all_hidden_states)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        return output[:, -1, :]  # (batch_size, all_hidden_states) of the last item in the sequence

    def forward(self, premise, premise_length, hypothesis, hypothesis_length):
        concatenate_list = [
            self.__get_lstm_output(self.premise_LSTM, self.premise_embedding(premise), premise_length),  # get sentence vector for premise
            self.__get_lstm_output(self.hypothesis_LSTM, self.hypothesis_embedding(hypothesis), hypothesis_length),  # get sentence vector for hypothesis
        ]
        pred = torch.cat(concatenate_list, dim=1)  # concatenate both the sentence vectors
        pred = self.feed_forward_model(pred)
        return pred
