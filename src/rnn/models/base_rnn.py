import torch.nn as nn
import torch
from ..utils import pack_right_padded_seq, unpack_seq


class BaseRNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_features: int, n_layers: int, output_features: int,
                 dropout: float = 0.0, bidirectional: bool = False, activation: str = 'tanh'):
        super(BaseRNN).__init__()
        assert vocab_size > 0
        self.vocab_size = vocab_size
        assert embedding_dim > 0
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.input_features = embedding_dim
        assert hidden_features > 0
        self.hidden_features = hidden_features
        assert n_layers > 0
        self.n_layers = n_layers
        assert output_features > 0
        self.output_features = output_features
        assert dropout >= 0.0 and dropout < 1
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.layers = self._init_layers()
        if self.bidirectional:
            self.layers_reverse = self._init_layers()
        self.activation = activation

    def _init_layers(self) -> torch.nn.ModuleList:
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch, seq_len, (right-padded) tokens]
        :return: [batch, features]
        """

        x, effective_batch_sizes = pack_right_padded_seq(x)

        x = self.embedding(x)

        if self.bidirectional:
            reverse_x = x.clone()
            reverse_x = torch.flip(reverse_x, dims=(-1, ))
            reverse_effective_batch_sizes = effective_batch_sizes.clone()
            reverse_effective_batch_sizes = torch.flip(reverse_effective_batch_sizes, dims=(-1, ))

        done_batches = 0
        hidden = torch.zeros(self.n_layers, self.hidden_features)
        for effective_batch_size in effective_batch_sizes:
            effective_batch = x[done_batches:effective_batch_size]
            for idx, layer in enumerate(self.layers):
                effective_batch = layer(effective_batch, hidden[idx])
                hidden[idx] = effective_batch
            done_batches += effective_batch_size
        x = hidden[-1]

        if self.bidirectional:
            done_batches = 0
            hidden = torch.zeros(self.n_layers, self.hidden_features)
            for effective_batch_size in reverse_effective_batch_sizes:
                effective_batch = reverse_x[done_batches:effective_batch_size]
                for idx, layer in enumerate(self.layers):
                    effective_batch = layer(effective_batch, hidden[idx])
                    hidden[idx] = effective_batch
                done_batches += effective_batch_size

        x = unpack_seq(x, effective_batch_sizes)
        if self.bidirectional:
            reverse_x = unpack_seq(reverse_x, reverse_effective_batch_sizes)
            x = torch.cat((x, reverse_x), dim=-1)
        return x
