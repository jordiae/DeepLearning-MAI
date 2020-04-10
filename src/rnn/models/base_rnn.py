import torch.nn as nn
import torch


class BaseRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, hidden_dim, n_layers):
        super(BaseRNN, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # RNN Layer
        self.rnn = nn.LSTM(embedding_size, hidden_dim, n_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Embedding
        x = self.embedding(x)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x)

        out = self.fc(out[:, -1, :])

        return out

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden

