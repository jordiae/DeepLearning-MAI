from torch import nn
import torch
from rnn.models.base_rnn import BaseRNN, BaseRNNLayer
from typing import Union
from typing import Tuple


class GRULayer(BaseRNNLayer):
    def __init__(self, *args, **kwargs):
        super(GRULayer, self).__init__(*args, **kwargs)

        # Update gate
        self.bu = nn.Parameter(torch.Tensor(self.hidden_features), requires_grad=True)
        self.Uu = nn.Parameter(torch.Tensor(self.hidden_features, self.input_features), requires_grad=True)
        self.Wu = nn.Parameter(torch.Tensor(self.hidden_features, self.hidden_features), requires_grad=True)

        # Reset gate
        self.br = nn.Parameter(torch.Tensor(self.hidden_features), requires_grad=True)
        self.Ur = nn.Parameter(torch.Tensor(self.hidden_features, self.input_features), requires_grad=True)
        self.Wr = nn.Parameter(torch.Tensor(self.hidden_features, self.hidden_features), requires_grad=True)

        # Output
        self.b = nn.Parameter(torch.Tensor(self.hidden_features), requires_grad=True)
        self.U = nn.Parameter(torch.Tensor(self.hidden_features, self.input_features), requires_grad=True)
        self.W = nn.Parameter(torch.Tensor(self.hidden_features, self.hidden_features), requires_grad=True)

        self._reset_parameters()

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, s_prev: Union[torch.Tensor, None] = None) ->\
            Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        Notation and equations from the Deep Learning book-
        :param x: [batch, input_features]
        :param h_prev: [batch, hidden_features]
        :param s_prev: Not used
        :return: [batch, hidden_features], None
        """

        # Update gate
        u = torch.sigmoid(self.bu + x.matmul(self.Uu.t()) + h_prev.matmul(self.Wu.t()))

        # Reset gate
        r = torch.sigmoid(self.br + x.matmul(self.Ur.t()) + h_prev.matmul(self.Wr.t()))

        # Output
        h = u*h_prev + (torch.ones(self.hidden_features) - u) * torch.tanh(self.b + x.matmul(self.U.t()) +
                                                                           r*h_prev.matmul(self.W.t()))
        return h, None


class GRU(BaseRNN):
    def __init__(self, *args, **kwargs):
        super(GRU, self).__init__(*args, cell=False, **kwargs)

    def _init_layers(self) -> torch.nn.ModuleList:
        """Stack of LSTM layers"""
        initial_layer = GRULayer(self.input_features, self.hidden_features)
        layers = [initial_layer]
        for i in range(1, self.n_layers):
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            layers.append(GRULayer(self.hidden_features, self.hidden_features))
        layers = nn.ModuleList(layers)
        return layers


if __name__ == '__main__':
    net = GRU(100, 64, 128, 3)
    x = torch.tensor([[1, 0], [9, 3], [4, 5]])
    lengths = torch.tensor([1, 2, 2])
    y = net(x, lengths)
    print(y)
