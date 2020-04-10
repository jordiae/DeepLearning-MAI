import torch.nn as nn
import torch
from rnn.utils import pack_right_padded_seq
import torch.nn.functional as F
import math


class BinaryClassifier(nn.Module):
    def __init__(self, in_features: int):
        """
        MLP classifier with one hidden layer
        :param in_features:
        """
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x


class BaseRNNLayer(nn.Module):
    def __init__(self, input_features: int, hidden_features: int, activation: str = 'tanh'):
        """
        Base class for RNN layers
        :param input_features:
        :param hidden_features:
        :param activation:
        """
        super().__init__()
        assert input_features > 0
        self.input_features = input_features
        assert hidden_features > 0
        self.hidden_features = hidden_features
        assert activation in ['tanh', 'relu']
        self.activation = torch.tanh if activation == 'tanh' else torch.relu

    def _reset_parameters(self):
        """
        Reset parameters. From: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py
        :return:
        """
        stdv = 1.0 / math.sqrt(self.hidden_features)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch, input_features]
        :param h_prev: [batch, hidden_features]
        :return: [batch, hidden_features]
        """
        raise NotImplementedError()


class BaseRNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_features: int, n_layers: int, dropout: float = 0.0,
                 bidirectional: bool = False, activation: str = 'tanh'):
        """
        Base class for RNN networks such that:
        - The input is a discrete sequence of tokens.
        - The target is a binary label.
        The final hidden state is input to a single-layer MLP classifier.
        :param vocab_size: Vocabulary size for the embedding layer
        :param embedding_dim: Embedding dimension (input to the first recurrent layer)
        :param hidden_features: Number of hidden features in the RNN layers.
        :param n_layers: Number of recurrent layers.
        :param dropout: Dropout probability, for for the classifier and the recurrent layers.
        :param bidirectional: Whether to use bidirectional RNNs (and concat the output of both directions).
        :param activation: Activation function to be used in the recurrent layers (not in the classifier, which will
        always be a sigmoid).
        """
        super().__init__()
        assert vocab_size > 0
        self.vocab_size = vocab_size
        assert embedding_dim > 0
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.input_features = embedding_dim
        assert hidden_features > 0
        self.hidden_features = hidden_features
        assert n_layers > 0
        self.n_layers = n_layers
        assert dropout >= 0.0 and dropout < 1
        self.dropout = dropout
        if self.dropout > 0:
            self.fc_dropout_layer = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.activation = activation
        self.layers = self._init_layers()
        if self.bidirectional:
            self.layers_reverse = self._init_layers()
        self.classifier = BinaryClassifier(hidden_features if not self.bidirectional else hidden_features*2)

    def _init_layers(self) -> torch.nn.ModuleList:
        """Instantiate recurrent layers. To be implemented by each subclass. Dropout should be placed between RNN
        layers. """
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch, seq_len, (right-padded) tokens]
        :return: [batch, 1]
        """
        bs = x.shape[0]
        x, effective_batch_sizes = pack_right_padded_seq(x)

        x = self.embedding(x)

        if self.bidirectional:
            reverse_x = x.clone()
            reverse_x = torch.flip(reverse_x, dims=(-1, ))
            reverse_effective_batch_sizes = effective_batch_sizes.clone()
            reverse_effective_batch_sizes = torch.flip(reverse_effective_batch_sizes, dims=(-1, ))

        done_batches = 0
        hidden = torch.zeros(bs, self.n_layers, self.hidden_features)
        for effective_batch_size in effective_batch_sizes:
            effective_batch = x[done_batches:effective_batch_size+done_batches]
            for idx, layer in enumerate(self.layers):
                effective_batch = layer(effective_batch, hidden[:, idx])
                hidden[:, idx] = effective_batch
            done_batches += effective_batch_size
        x = hidden[:, -1]

        if self.bidirectional:
            done_batches = 0
            hidden = torch.zeros(bs, self.n_layers, self.hidden_features)
            for effective_batch_size in reverse_effective_batch_sizes:
                effective_batch = reverse_x[done_batches:effective_batch_size + done_batches]
                for idx, layer in enumerate(self.layers):
                    effective_batch = layer(effective_batch, hidden[:, idx])
                    hidden[:, idx] = effective_batch
                done_batches += effective_batch_size
            reverse_x = hidden[:, -1]

        if self.bidirectional:
            x = torch.cat((x, reverse_x), dim=-1)

        if self.dropout > 0.0:
            x = self.fc_dropout_layer(x)

        x = self.classifier(x)

        return x
