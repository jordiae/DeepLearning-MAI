from torch import nn
import torch
from rnn.models.base_rnn import BaseRNN, BaseRNNLayer
from typing import Union
from typing import Tuple


class LSTMLayer(BaseRNNLayer):
    def __init__(self, *args, **kwargs):
        super(LSTMLayer, self).__init__(*args, **kwargs)
        # Forget gate, f
        self.bf = nn.Parameter(torch.Tensor(self.hidden_features), requires_grad=True)
        self.Uf = nn.Parameter(torch.Tensor(self.hidden_features, self.input_features), requires_grad=True)
        self.Wf = nn.Parameter(torch.Tensor(self.hidden_features, self.hidden_features), requires_grad=True)

        # Self-loop, s (cell)
        self.b = nn.Parameter(torch.Tensor(self.hidden_features), requires_grad=True)
        self.U = nn.Parameter(torch.Tensor(self.hidden_features, self.input_features), requires_grad=True)
        self.W = nn.Parameter(torch.Tensor(self.hidden_features, self.hidden_features), requires_grad=True)
        self.s_prev = torch.zeros(self.hidden_features)

        # Input gate, g
        self.bg = nn.Parameter(torch.Tensor(self.hidden_features), requires_grad=True)
        self.Ug = nn.Parameter(torch.Tensor(self.hidden_features, self.input_features), requires_grad=True)
        self.Wg = nn.Parameter(torch.Tensor(self.hidden_features, self.hidden_features), requires_grad=True)
        # Output gate, q
        self.bo = nn.Parameter(torch.Tensor(self.hidden_features), requires_grad=True)
        self.Uo = nn.Parameter(torch.Tensor(self.hidden_features, self.input_features), requires_grad=True)
        self.Wo = nn.Parameter(torch.Tensor(self.hidden_features, self.hidden_features), requires_grad=True)

        self._reset_parameters()

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, s_prev: Union[torch.Tensor, None] = None) ->\
            Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        Notation and equations from the Deep Learning book
        :param x: [batch, input_features]
        :param h_prev: [batch, hidden_features]
        :param s_prev: [batch, hidden_features]
        :return: [batch, hidden_features]
        """
        # Forget gate, f
        f = torch.sigmoid(self.bf + x.matmul(self.Uf.t()) + h_prev.matmul(self.Wf.t()))

        # Input gate, g
        g = torch.sigmoid(self.bg + x.matmul(self.Ug.t()) + h_prev.matmul(self.Wg.t()))

        # Self-loop, s. torch's implementation uses tanh, but the Deep Learning book suggests sigmoid
        s = f * s_prev + g * torch.tanh(self.b + x.matmul(self.U.t()) + h_prev.matmul(self.W.t()))

        # Output gate, q

        q = torch.sigmoid(self.bo + x.matmul(self.Uo.t()) + h_prev.matmul(self.Wo.t()))

        h = (torch.tanh(s)) * q

        return h, s


class LSTM(BaseRNN):
    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__(*args, cell=True, **kwargs)

    def _init_layers(self) -> torch.nn.ModuleList:
        """Stack of LSTM layers"""
        initial_layer = LSTMLayer(self.input_features, self.hidden_features)
        layers = [initial_layer]
        for i in range(1, self.n_layers):
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            layers.append(LSTMLayer(self.hidden_features, self.hidden_features))
        layers = nn.ModuleList(layers)
        return layers


if __name__ == '__main__':
    net = LSTM(100, 64, 128, 3)
    x = torch.tensor([[1, 0], [9, 3], [4, 5]])
    lengths = torch.tensor([1, 2, 2])
    y = net(x, lengths)
    print(y)
