from torch import nn
import torch
from rnn.models.base_rnn import BaseRNN, BaseRNNLayer


class VanillaRNNLayer(BaseRNNLayer):
    def __init__(self, *args, mode: str = 'jordan', **kwargs):
        """ Vanilla RNN layer. Unlike torch.nn's implementation, it supports both Jordan and Elman RNN layers
        :param mode: either Jordan or Elman"""
        super().__init__(*args, **kwargs)
        assert mode in ['elman', 'jordan']
        self.mode = mode
        self.b = nn.Parameter(torch.Tensor(self.hidden_features), requires_grad=True)
        self.W = nn.Parameter(torch.Tensor(self.hidden_features, self.hidden_features), requires_grad=True)
        self.U = nn.Parameter(torch.Tensor(self.hidden_features, self.input_features), requires_grad=True)
        if self.mode == 'jordan':
            self.V = nn.Parameter(torch.Tensor(self.hidden_features, self.hidden_features), requires_grad=True)
            self.c = nn.Parameter(torch.Tensor(self.hidden_features), requires_grad=True)
        self._reset_parameters()

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch, input_features]
        :param h_prev: [batch, hidden_features]
        :return: [batch, hidden_features]
        """
        a = self.b + h_prev.matmul(self.W.t()) + x.matmul(self.U.t())
        h = self.activation(a)
        if self.mode == 'elman':
            return h
        o = self.c + h.matmul(self.V.t())
        return o


class VanillaRNN(BaseRNN):
    def __init__(self, *args, mode: str = 'jordan', **kwargs):
        """ Vanilla RNN layer. Unlike torch.nn's implementation, it supports both Jordan and Elman RNN layers.
        :param mode: either Jordan or Elman"""
        assert mode in ['elman', 'jordan']
        self.mode = mode
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> torch.nn.ModuleList:
        """Stack of vanilla RNN layers"""
        initial_layer = VanillaRNNLayer(self.input_features, self.hidden_features, self.activation, mode=self.mode)
        layers = [initial_layer]
        for i in range(1, self.n_layers):
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            layers.append(VanillaRNNLayer(self.input_features, self.hidden_features, self.activation, mode=self.mode))
        layers = nn.ModuleList(layers)
        return layers

if __name__ == '__main__':
    net = VanillaRNN(100, 64, 128, 3)
    x = torch.tensor([[2,3], [4,5]])
    y = net(x)
    print()