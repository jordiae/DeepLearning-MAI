from torch import nn
import torch
from typing import Union
import math


class RNNLayer(nn.Module):
    def __init__(self, input_features: int, hidden_features: int, dropout: float = 0.0,
                 mode: str = 'jordan', elman_activation: Union[str, None] = 'tanh'):
        super(RNNLayer).__init__()
        assert input_features > 0
        self.input_features = input_features
        assert hidden_features > 0
        self.hidden_features = hidden_features
        assert dropout >= 0.0 and dropout < 1
        self.dropout = dropout
        assert mode in ['elman', 'jordan']
        self.mode = mode
        assert elman_activation in ['tanh', 'relu', None]
        if elman_activation == 'tanh':
            self.elman_activation = torch.tanh
        elif elman_activation == 'relu':
            self.elman_activation = torch.relu
        else:
            self.elman_activation = None
        self.b = nn.Parameter(torch.Tensor(self.hidden_features), requires_grad=True)
        self.W = nn.Parameter(torch.Tensor(self.hidden_features, self.hidden_features), requires_grad=True)
        self.U = nn.Parameter(torch.Tensor(self.hidden_features, self.input_features), requires_grad=True)
        if self.mode == 'jordan':
            self.V = nn.Parameter(torch.Tensor(self.hidden_features, self.hidden_features), requires_grad=True)
            self.c = nn.Parameter(torch.Tensor(self.hidden_features), requires_grad=True)

        self._reset_parameters()

    def _reset_parameters(self):
        # From: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py
        stdv = 1.0 / math.sqrt(self.hidden_features)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch, input_features]
        :param h_prev: [batch, hidden_features]
        :return: [batch, hidden_features]
        """
        a = self.b + h_prev.matmul(self.W.t()) + x.matmul(self.U.t())
        h = a if self.elman_activation is None else self.elman_activation(a)
        if self.mode == 'elman':
            return h
        o = self.c + h.matmul(self.V.t())
        return o


class RNNNet(nn.Module):
    def __init__(self, input_features: int, hidden_features: int, n_layers: int, output_features: int,
                 dropout: float = 0.0, bidirectional: bool = False, mode: str = 'jordan',
                 activation: str = 'tanh', final_activation: Union[str, None] = None):
        super(RNNNet).__init__()
        assert input_features > 0
        self.input_features = input_features
        assert hidden_features > 0
        self.hidden_features = hidden_features
        assert n_layers > 0
        self.n_layers = n_layers
        assert output_features > 0
        self.output_features = output_features
        assert dropout >= 0.0 and dropout < 1
        self.dropout = dropout
        self.bidirectional = bidirectional
        assert mode in ['elman', 'jordan']
        self.mode = mode
        assert activation in ['tanh', 'relu']
        self.activation = activation
        assert final_activation in ['likewise', None]
        self.final_activation = final_activation
        self.layers = self._init_layers()
        if self.bidirectional:
            self.layers_reverse = self._init_layers()

    def _init_layers(self) -> torch.nn.ModuleList:
        if self.n_layers == 1:
            layers = nn.ModuleList([RNNLayer(self.input_features, self.output_features, self.dropout, self.mode,
                                             self.elman_activation)])
        elif self.n_layers == 2:
            layers = nn.ModuleList([RNNLayer(self.input_features, self.hidden_features, self.dropout, self.mode,
                                             self.activation),
                                    RNNLayer(self.hidden_features, self.output_features, self.dropout, self.mode,
                                             self.activation)
                                    ])
        else:
            initial_layer = RNNLayer(self.input_features, self.hidden_features, self.dropout, self.mode,
                                             self.activation)
            intermediate_layers = [RNNLayer(self.hidden_features, self.hidden_features, self.dropout, self.mode,
                                             self.activation) for _ in range(0, self.n_layers - 2)]
            final_layer = RNNLayer(self.hidden_features, self.output_features, self.dropout, self.mode,
                                             self.activation)
            layers = nn.ModuleList([initial_layer] + intermediate_layers + [final_layer])
        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch, seq_len, features]
        :return: [batch, features]
        """
        activation = torch.tanh if self.activation == 'tanh' else torch.relu
        if self.bidirectional:
            reverse_x = x.clone()
            reverse_x = torch.flip(reverse_x, dims=(1, ))
        needs_last_layer = False
        # TODO: for each element of the sequence
        for idx, layer in enumerate(self.layers):
            if idx == len(self.layers) - 1 and self.mode == 'jordan' and self.final_activation is None:
                needs_last_layer = True
                break
            x = layer(x)
            if self.mode == 'jordan':
                x = activation(x)
            if self.bidirectional:
                reverse_x = self.layers_reverse[idx](reverse_x)
                if self.mode == 'jordan':
                    reverse_x = activation(reverse_x)
        if needs_last_layer:
            x = self.layers[-1](x)
            if self.bidirectional:
                reverse_x = self.layers_reverse[-1](reverse_x)
        if self.bidirectional:
            x = torch.cat((x, reverse_x), dim=-1)
        return x
