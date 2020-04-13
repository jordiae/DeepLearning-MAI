from torch import nn
import torch
from rnn.models.base_rnn import BaseRNN, BaseRNNLayer
from typing import Union
from typing import Tuple


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

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, s_prev: Union[torch.Tensor, None] = None) -> \
            Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        :param x: [batch, input_features]
        :param h_prev: [batch, hidden_features]
        :param s_prev: None (not used)
        :return: [batch, hidden_features]
        """
        a = self.b + h_prev.matmul(self.W.t()) + x.matmul(self.U.t())
        h = torch.tanh(a)
        if self.mode == 'elman':
            return h, None
        o = torch.tanh(self.c + h.matmul(self.V.t()))
        return o, None


class VanillaRNN(BaseRNN):
    def __init__(self, *args, mode: str = 'jordan', **kwargs):
        """ Vanilla RNN layer. Unlike torch.nn's implementation, it supports both Jordan and Elman RNN layers.
        :param mode: either Jordan or Elman"""
        assert mode in ['elman', 'jordan']
        self.mode = mode
        super().__init__(*args, cell=False, **kwargs)

    def _init_layers(self) -> torch.nn.ModuleList:
        """Stack of vanilla RNN layers"""
        initial_layer = VanillaRNNLayer(self.input_features, self.hidden_features, mode=self.mode)
        layers = [initial_layer]
        for i in range(1, self.n_layers):
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            layers.append(VanillaRNNLayer(self.hidden_features, self.hidden_features, mode=self.mode))
        layers = nn.ModuleList(layers)
        return layers


if __name__ == '__main__':
    from rnn.models.base_rnn import Decoder
    net = VanillaRNN(torch.device('cpu'), 100, 64, 128, 1, mode='jordan')
    x = torch.tensor([[1, 0], [9, 3], [4, 5]])
    lengths = torch.tensor([1, 2, 2])
    encoder_x, encoder_hidden, _ = net(x, lengths)
    net2 = VanillaRNN(torch.device('cpu'), 100, 64, 128, 1, mode='jordan')
    decoder = Decoder(net2, 100)
    t = torch.tensor([[1], [9], [4]])
    l = torch.tensor([[1], [1], [1]])
    decoder_x, decoder_hidden, _ = decoder(t, l, encoder_hidden, None)
    print(decoder_x)
