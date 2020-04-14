import torch.nn as nn
import torch
from rnn.utils import pack_right_padded_seq
import math
from typing import Union
from typing import Tuple


class BaseRNNLayer(nn.Module):
    def __init__(self, input_features: int, hidden_features: int):
        """
        Base class for RNN layers
        :param input_features: Input dims
        :param hidden_features: Hidden dims
        """
        super().__init__()
        assert input_features > 0
        self.input_features = input_features
        assert hidden_features > 0
        self.hidden_features = hidden_features

    def _reset_parameters(self):
        """
        Reset parameters. From: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py
        :return:
        """
        stdv = 1.0 / math.sqrt(self.hidden_features)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, s_prev: Union[torch.Tensor, None] = None) ->\
            Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        :param x: [batch, input_features]
        :param h_prev: [batch, hidden_features]
        :param s_prev: [batch, hidden_Features]: optional, only for gated RNNs having a cell state (self.cell).
        :return: Tuple of tensors with shape [batch, hidden_features]. If s_prev is not used (vanilla RNN), tuple of
        only one tensor, h, and None. If s_prev is used, the second element is s.
        """
        raise NotImplementedError()


class BaseRNN(nn.Module):
    def __init__(self, device: torch.device, vocab_size: int, embedding_dim: int, hidden_features: int, n_layers: int,
                 dropout: float = 0.0, bidirectional: bool = False, cell: bool = False,
                 embeddings: Union[nn.Embedding, None] = None):
        """
        Base class for RNN networks such that the input is a discrete sequence of tokens.
        :param device: torch device
        :param vocab_size: Vocabulary size for the embedding layer
        :param embedding_dim: Embedding dimension (input to the first recurrent layer)
        :param hidden_features: Number of hidden features in the RNN layers.
        :param n_layers: Number of recurrent layers.
        :param dropout: Dropout probability, for the recurrent layers.
        :param bidirectional: Whether to use bidirectional RNNs (and concat the output of both directions).
        :param cell: Whether the network has an internal cell state.
        :param embeddings: If not None, initialized embedding layer (for embedding sharing).
        """
        super().__init__()
        self.device = device
        assert vocab_size > 0
        self.vocab_size = vocab_size
        assert embedding_dim > 0
        self.embedding = nn.Embedding(vocab_size, embedding_dim) if embeddings is None else embeddings
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
        self.cell = cell
        self.layers = self._init_layers()
        if self.bidirectional:
            self.layers_reverse = self._init_layers()

    def _init_layers(self) -> torch.nn.ModuleList:
        """Instantiate recurrent layers. To be implemented by each subclass. Dropout should be placed between RNN
        layers. """
        raise NotImplementedError()

    def _forward(self, x: torch.Tensor, layers: torch.nn.ModuleList, bs: int, effective_batch_sizes: torch.Tensor,
                 initial_hidden: Union[torch.Tensor, None], initial_cell: Union[torch.Tensor, None]) ->\
            Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        Auxiliar method for computing the forward pass of a RNN network (it will be called twice if it is bidirectional)
        :param x: Input tokens, already packed passed through the embedding layer.
        :param layers: Layers of the network.
        :param bs: Batch size.
        :param effective_batch_sizes: Effective batch sizes due to packing.
        :param initial_hidden: Initial hidden state (set to 0 if None)
        :param initial_cell: Initial cell state, if the model has an internal cell state, eg. LSTMs (set to 0 if None)
        :return: Final hidden states. :return: Final hidden states of each layer: [batch, n_layers, hidden_features].
        """
        done_batches = 0
        hidden = torch.zeros(bs, self.n_layers, self.hidden_features).to(self.device)\
            if initial_hidden is None else initial_hidden
        if self.cell:
            cell = torch.zeros(bs, self.n_layers, self.hidden_features).to(self.device)\
                if initial_cell is None else initial_cell
        for effective_batch_size in effective_batch_sizes:
            effective_batch = x[done_batches:effective_batch_size + done_batches]
            for idx, layer in enumerate(layers):
                if isinstance(layer, nn.Dropout):
                    effective_batch = layer(effective_batch)
                    continue
                if self.dropout:
                    idx //= 2

                hidden_batch = hidden[:effective_batch_size, idx].clone()
                hidden_batch.retain_grad()
                if not self.cell:
                    effective_batch, _ = layer(effective_batch, hidden_batch)
                    hidden[:effective_batch_size, idx] = effective_batch
                else:
                    cell_batch = cell[:effective_batch_size, idx].clone()
                    effective_batch, s = layer(effective_batch, hidden_batch, cell_batch)
                    hidden[:effective_batch_size, idx] = effective_batch
                    cell[:effective_batch_size, idx] = s
            done_batches += effective_batch_size
        return hidden, None if not self.cell else cell

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, initial_hidden: Union[torch.Tensor, None] = None,
                initial_cell: Union[torch.Tensor, None] = None) -> \
            Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, None]]:
        """
        :param x: [batch, seq_len, (right-padded) tokens]
        :param lengths: [batch]
        :param initial_hidden: Either None or [batch, n_layers, hidden_features]
        :param initial_cell: Either None or [batch, n_layers, hidden_features]
        :return: Tuple [batch, hidden_Features], [batch, n_layers, hidden_features], [batch, n_layers, hidden_features],
        where the first element contains the final hidden states of the last layer, and the second and third one contain
        the final hidden and cell states from all layers, respectively.
        """
        bs = x.shape[0]
        x, effective_batch_sizes = pack_right_padded_seq(x, lengths, self.device)

        x = self.embedding(x)

        if self.bidirectional:
            reverse_x = x.clone()
            reverse_x = torch.flip(reverse_x, dims=(-1, ))
            reverse_effective_batch_sizes = effective_batch_sizes.clone()
            reverse_effective_batch_sizes = torch.flip(reverse_effective_batch_sizes, dims=(-1, ))
            if initial_hidden is not None:
                initial_hidden, reverse_initial_hidden = initial_hidden[:, :, :self.hidden_features],\
                                                         initial_hidden[:, :, self.hidden_features:]
            else:
                reverse_initial_hidden = None
            if initial_cell is not None:
                initial_cell, reverse_initial_cell = initial_cell[:, :, :self.hidden_features], \
                                                       initial_cell[:, :, self.hidden_features:]
            else:
                reverse_initial_cell = None

        hidden, cell = self._forward(x, self.layers, bs, effective_batch_sizes, initial_hidden, initial_cell)
        x = hidden[:, -1]

        if self.bidirectional:
            reverse_hidden, reverse_cell = self._forward(reverse_x, self.layers, bs, reverse_effective_batch_sizes,
                                                         reverse_initial_hidden, reverse_initial_cell)
            reverse_x = reverse_hidden[:, -1]
            hidden = torch.cat((hidden, reverse_hidden), dim=-1)
            if self.cell:
                cell = torch.cat((cell, reverse_cell), dim=-1)
            x = torch.cat((x, reverse_x), dim=-1)

        return x, hidden, None if not self.cell else cell


class Decoder(nn.Module):
    def __init__(self, net: nn.Module, vocab_size: int):
        """
        Wrapper around BaseRNN for adding a projection layer.
        :param net: BaseRNN instance
        :param vocab_size:  Vocabulary size
        """
        super(Decoder, self).__init__()
        assert not net.bidirectional
        self.net = net
        self.linear = nn.Linear(net.hidden_features, vocab_size)

    def forward(self, tgt_tokens, tgt_lengths, initial_hidden, initial_cell):
        x, hidden, cell = self.net(tgt_tokens, tgt_lengths, initial_hidden, initial_cell)
        x = self.linear(x)
        return x, hidden, cell


class PyTorchBaseRNN(nn.Module):
    def __init__(self, device: torch.device, vocab_size: int, embedding_dim: int, hidden_features: int, n_layers: int,
                 dropout: float = 0.0, bidirectional: bool = False, cell: bool = False,
                 embeddings: Union[nn.Embedding, None] = None, arch: str = 'lstm'):
        """
        Like BaseRNN, but using PyTorch implementations for the sake of efficiency.
        The final hidden state is input to a single-layer MLP classifier.
        :param device: torch device
        :param vocab_size: Vocabulary size for the embedding layer
        :param embedding_dim: Embedding dimension (input to the first recurrent layer)
        :param hidden_features: Number of hidden features in the RNN layers.
        :param n_layers: Number of recurrent layers.
        :param dropout: Dropout probability, for the recurrent layers.
        :param bidirectional: Whether to use bidirectional RNNs (and concat the output of both directions).
        :param cell: Whether the network has an internal cell state.
        :param embeddings: If not None, initialized embedding layer (for embedding sharing).
        :param arch: Architecture ('elman', 'gru' or 'lstm').
        """
        super().__init__()
        self.device = device
        assert vocab_size > 0
        self.vocab_size = vocab_size
        assert embedding_dim > 0
        self.embedding = nn.Embedding(vocab_size, embedding_dim) if embeddings is None else embeddings
        self.input_features = embedding_dim
        assert hidden_features > 0
        self.hidden_features = hidden_features
        assert n_layers > 0
        self.n_layers = n_layers
        assert dropout >= 0.0 and dropout < 1
        self.dropout = dropout
        self.bidirectional = bidirectional
        assert arch in ['elman', 'gru', 'lstm']
        self.cell = True if arch == 'lstm' else False
        if arch == 'elman':
            self.rnn = nn.RNN(self.input_features, self.hidden_features, num_layers=self.n_layers,
                              bidirectional=self.bidirectional, dropout=self.dropout, batch_first=True)
        elif arch == 'gru':
            self.rnn = nn.GRU(self.input_features, self.hidden_features, num_layers=self.n_layers,
                              bidirectional=self.bidirectional, dropout=self.dropout, batch_first=True)
        else:
            self.rnn = nn.LSTM(self.input_features, self.hidden_features, num_layers=self.n_layers,
                               bidirectional=self.bidirectional, dropout=self.dropout, batch_first=True)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor, initial_hidden: Union[torch.Tensor, None] = None,
                initial_cell: Union[torch.Tensor, None] = None) -> \
            Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, None]]:
        """
        Similar to the forward pass of BaseRNN, we apply permutations for API compatibility with BaseRNN.
        :param tokens: [batch, seq_len, (right-padded) tokens]
        :param lengths: [batch]
        :param initial_hidden: Either None or [batch, n_layers, hidden_features]
        :param initial_cell: Either None or [batch, n_layers, hidden_features]
        :return: Tuple [batch, hidden_Features], [batch, n_layers, hidden_features], [batch, n_layers, hidden_features],
        where the first element contains the final hidden states of the last layer, and the second and third one contain
        the final hidden and cell states from all layers, respectively.
        """
        bs = tokens.shape[0]
        hidden_layers = initial_hidden.shape[1] if initial_hidden is not None else 0
        tokens = self.embedding(tokens)
        tokens = torch.nn.utils.rnn.pack_padded_sequence(tokens, lengths, batch_first=True, enforce_sorted=False)
        if not self.cell:
            cell = None
            if initial_hidden is None:
                x, hidden = self.rnn(tokens)
            else:
                if self.bidirectional:
                    initial_hidden = initial_hidden.view(bs, self.n_layers, self.hidden_features*2)
                elif hidden_layers == 2*self.n_layers:
                    initial_hidden = initial_hidden.view(bs, self.n_layers, self.hidden_features)
                x, hidden = self.rnn(tokens, initial_hidden.permute(1, 0, 2))
        else:
            if initial_hidden is None:
                x, (hidden, cell) = self.rnn(tokens)
            else:
                if self.bidirectional:
                    initial_hidden = initial_hidden.view(bs, self.n_layers, self.hidden_features*2)
                    initial_cell = initial_cell.view(bs, self.n_layers, self.hidden_features * 2)
                elif hidden_layers == 2*self.n_layers:
                    initial_hidden = initial_hidden.view(bs, self.n_layers, self.hidden_features)
                    initial_cell = initial_cell.view(bs, self.n_layers, self.hidden_features)
                x, (hidden, cell) = self.rnn(tokens, (initial_hidden.permute(1, 0, 2), initial_cell.permute(1, 0, 2)))
            cell = cell.permute(1, 0, 2)
            if hidden_layers == 2 * self.n_layers:
                cell = torch.cat((cell[:, :, :self.hidden_features // 2], cell[:, :, self.hidden_features // 2:]),
                                 dim=1)
        hidden = hidden.permute(1, 0, 2)
        if hidden_layers == 2*self.n_layers:
            hidden = torch.cat((hidden[:, :, :self.hidden_features//2], hidden[:, :, self.hidden_features//2:]), dim=1)
        x = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x[0][:, -1]
        return x, hidden, cell
