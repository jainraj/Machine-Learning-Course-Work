import torch
from torch.nn import init, Module, LSTM, Sequential, GRU, RNN
from torch.nn.utils.rnn import pack_padded_sequence


class SNLI_LSTM(Module):
    """Define the model with LSTM as base"""

    def __init__(self,
                 premise_embedding, hypothesis_embedding,
                 hidden_size, layers,
                 feed_forward_model: Sequential):
        super(SNLI_LSTM, self).__init__()

        self.premise_embedding = premise_embedding
        self.hypothesis_embedding = hypothesis_embedding

        self.premise_LSTM = LSTM(input_size=premise_embedding.embedding_dim, hidden_size=hidden_size, num_layers=layers, batch_first=True, bidirectional=True)
        self.hypothesis_LSTM = LSTM(input_size=hypothesis_embedding.embedding_dim, hidden_size=hidden_size, num_layers=layers, batch_first=True, bidirectional=True)

        self.feed_forward_model = feed_forward_model  # a Sequential model

        # Initialisations
        for name, param in self.premise_LSTM.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                init.orthogonal_(param)
            elif 'bias_ih' in name:
                init.zeros_(param)
            elif 'bias_hh' in name:
                param.data = torch.tensor([0] * hidden_size + [1] * hidden_size + [0] * hidden_size * 2, dtype=torch.float)
        for name, param in self.hypothesis_LSTM.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                init.orthogonal_(param)
            elif 'bias_ih' in name:
                init.zeros_(param)
            elif 'bias_hh' in name:
                param.data = torch.tensor([0] * hidden_size + [1] * hidden_size + [0] * hidden_size * 2, dtype=torch.float)
        for name, param in self.feed_forward_model.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param)
            if 'bias' in name:
                init.zeros_(param)

        self.float()

    def __get_lstm_output(self, lstm, lstm_inp, lengths):  # lstm_inp is of shape (batch_size, seq_length, embedding_size)
        packed_input = pack_padded_sequence(lstm_inp, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = lstm(packed_input)  # h is of shape (D*num_layers, batch_size, hidden_size)
        batch_size = h.shape[1]
        return h.transpose(0, 1).reshape(batch_size, -1)  # (batch_size, all_hidden_states) of the last item in the sequence

    def forward(self, premise, premise_length, hypothesis, hypothesis_length):
        concatenate_list = [
            self.__get_lstm_output(self.premise_LSTM, self.premise_embedding(premise), premise_length),  # get sentence vector for premise
            self.__get_lstm_output(self.hypothesis_LSTM, self.hypothesis_embedding(hypothesis), hypothesis_length),  # get sentence vector for hypothesis
        ]
        pred = torch.cat(concatenate_list, dim=1)  # concatenate both the sentence vectors
        pred = self.feed_forward_model(pred)
        return pred


class SNLI_GRU(Module):
    """Define the model with GRU as base"""

    def __init__(self,
                 premise_embedding, hypothesis_embedding,
                 hidden_size, layers,
                 feed_forward_model: Sequential):
        super(SNLI_GRU, self).__init__()

        self.premise_embedding = premise_embedding
        self.hypothesis_embedding = hypothesis_embedding

        self.premise_GRU = GRU(input_size=premise_embedding.embedding_dim, hidden_size=hidden_size, num_layers=layers, batch_first=True, bidirectional=True)
        self.hypothesis_GRU = GRU(input_size=hypothesis_embedding.embedding_dim, hidden_size=hidden_size, num_layers=layers, batch_first=True, bidirectional=True)

        self.feed_forward_model = feed_forward_model  # a Sequential model

        # Initialisations
        for name, param in self.premise_GRU.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                init.orthogonal_(param)
            elif 'bias_' in name:
                init.zeros_(param)
        for name, param in self.hypothesis_GRU.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                init.orthogonal_(param)
            elif 'bias_' in name:
                init.zeros_(param)
        for name, param in self.feed_forward_model.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param)
            if 'bias' in name:
                init.zeros_(param)

        self.float()

    def __get_gru_output(self, gru, gru_inp, lengths):  # gru_inp is of shape (batch_size, seq_length, embedding_size)
        packed_input = pack_padded_sequence(gru_inp, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = gru(packed_input)  # h is of shape (D*num_layers, batch_size, hidden_size)
        batch_size = h.shape[1]
        return h.transpose(0, 1).reshape(batch_size, -1)  # (batch_size, all_hidden_states) of the last item in the sequence

    def forward(self, premise, premise_length, hypothesis, hypothesis_length):
        concatenate_list = [
            self.__get_gru_output(self.premise_GRU, self.premise_embedding(premise), premise_length),  # get sentence vector for premise
            self.__get_gru_output(self.hypothesis_GRU, self.hypothesis_embedding(hypothesis), hypothesis_length),  # get sentence vector for hypothesis
        ]
        pred = torch.cat(concatenate_list, dim=1)  # concatenate both the sentence vectors
        pred = self.feed_forward_model(pred)
        return pred


class SNLI_RNN(Module):
    """Define the model with vanilla RNN as base"""

    def __init__(self,
                 premise_embedding, hypothesis_embedding,
                 hidden_size, layers,
                 feed_forward_model: Sequential):
        super(SNLI_RNN, self).__init__()

        self.premise_embedding = premise_embedding
        self.hypothesis_embedding = hypothesis_embedding

        self.premise_RNN = RNN(input_size=premise_embedding.embedding_dim, hidden_size=hidden_size, num_layers=layers, batch_first=True, bidirectional=True)
        self.hypothesis_RNN = RNN(input_size=hypothesis_embedding.embedding_dim, hidden_size=hidden_size, num_layers=layers, batch_first=True, bidirectional=True)

        self.feed_forward_model = feed_forward_model  # a Sequential model

        # Initialisations
        for name, param in self.premise_RNN.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                init.orthogonal_(param)
            elif 'bias_' in name:
                init.zeros_(param)
        for name, param in self.hypothesis_RNN.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                init.orthogonal_(param)
            elif 'bias_' in name:
                init.zeros_(param)
        for name, param in self.feed_forward_model.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param)
            if 'bias' in name:
                init.zeros_(param)

        self.float()

    def __get_rnn_output(self, rnn, rnn_inp, lengths):  # rnn_inp is of shape (batch_size, seq_length, embedding_size)
        packed_input = pack_padded_sequence(rnn_inp, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = rnn(packed_input)  # h is of shape (D*num_layers, batch_size, hidden_size)
        batch_size = h.shape[1]
        return h.transpose(0, 1).reshape(batch_size, -1)  # (batch_size, all_hidden_states) of the last item in the sequence

    def forward(self, premise, premise_length, hypothesis, hypothesis_length):
        concatenate_list = [
            self.__get_rnn_output(self.premise_RNN, self.premise_embedding(premise), premise_length),  # get sentence vector for premise
            self.__get_rnn_output(self.hypothesis_RNN, self.hypothesis_embedding(hypothesis), hypothesis_length),  # get sentence vector for hypothesis
        ]
        pred = torch.cat(concatenate_list, dim=1)  # concatenate both the sentence vectors
        pred = self.feed_forward_model(pred)
        return pred


class SNLI_GRU_Interaction(Module):
    """Define the interaction model with LSTM as base"""

    def __init__(self,
                 premise_embedding, hypothesis_embedding,
                 hidden_size, layers,
                 feed_forward_model: Sequential):
        super(SNLI_GRU_Interaction, self).__init__()

        self.premise_embedding = premise_embedding
        self.hypothesis_embedding = hypothesis_embedding

        self.premise_GRU = GRU(input_size=premise_embedding.embedding_dim, hidden_size=hidden_size, num_layers=layers, batch_first=True, bidirectional=True)
        output_size_from_premise_LSTM = hidden_size * layers * 2  # bidirectional
        self.hypothesis_GRU = GRU(input_size=hypothesis_embedding.embedding_dim + output_size_from_premise_LSTM, hidden_size=hidden_size, num_layers=layers, batch_first=True, bidirectional=True)

        self.feed_forward_model = feed_forward_model  # a Sequential model

        # Initialisations
        for name, param in self.premise_GRU.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                init.orthogonal_(param)
            elif 'bias_' in name:
                init.zeros_(param)
        for name, param in self.hypothesis_GRU.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                init.orthogonal_(param)
            elif 'bias_' in name:
                init.zeros_(param)
        for name, param in self.feed_forward_model.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param)
            if 'bias' in name:
                init.zeros_(param)

        self.float()

    def __get_gru_output(self, gru, gru_inp, lengths):  # gru_inp is of shape (batch_size, seq_length, embedding_size)
        packed_input = pack_padded_sequence(gru_inp, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = gru(packed_input)  # h is of shape (D*num_layers, batch_size, hidden_size)
        batch_size = h.shape[1]
        return h.transpose(0, 1).reshape(batch_size, -1)  # (batch_size, all_hidden_states) of the last item in the sequence

    def forward(self, premise, premise_length, hypothesis, hypothesis_length):
        # Get premise representation
        premise_sentence_repr = self.__get_gru_output(self.premise_GRU, self.premise_embedding(premise), premise_length)  # get sentence vector for premise

        # Make hypothesis GRU input
        hypothesis_embedding = self.hypothesis_embedding(hypothesis)
        premise_size = premise_sentence_repr.shape[1]
        batch_size, timestamps, embedding_dim = hypothesis_embedding.shape
        hypothesis_GRU_inp = torch.zeros((batch_size, timestamps, embedding_dim + premise_size))
        hypothesis_GRU_inp[:, :, :embedding_dim] = hypothesis_embedding
        for t in range(timestamps):
            hypothesis_GRU_inp[:, t, embedding_dim:] = premise_sentence_repr

        # Get hypothesis representation
        hypothesis_sentence_repr = self.__get_gru_output(self.hypothesis_GRU, hypothesis_GRU_inp, hypothesis_length)  # get sentence vector for hypothesis

        pred = torch.cat([premise_sentence_repr, hypothesis_sentence_repr], dim=1)  # concatenate both the sentence vectors
        pred = self.feed_forward_model(pred)
        return pred
