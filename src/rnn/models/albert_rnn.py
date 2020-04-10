import torch.nn as nn
import torch

class AlbertRNN(nn.Module):

    def __init__(self, vocab_size, embed_size, num_output, rnn_model='LSTM', use_last=True,
                 hidden_size=64, num_layers=1):
        """
        Args:
            vocab_size: vocab size
            embed_size: embedding size
            num_output: number of output (classes)
            rnn_model:  LSTM or GRU
            use_last:  bool
            embedding_tensor:
            padding_index:
            hidden_size: hidden size of rnn module
            num_layers:  number of layers in rnn module
            batch_first: batch first option
        """

        super(AlbertRNN, self).__init__()
        self.use_last = use_last
        # embedding
        self.encoder = nn.Embedding(vocab_size, embed_size)

        self.drop_en = nn.Dropout(p=0.6)

        # rnn module
        if rnn_model == 'LSTM':
            self.rnn = nn.LSTM( input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.5,
                                batch_first=True, bidirectional=True)
        elif rnn_model == 'GRU':
            self.rnn = nn.GRU( input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.5,
                                batch_first=True, bidirectional=True)
        else:
            raise LookupError(' only support LSTM and GRU')


        self.bn2 = nn.BatchNorm1d(hidden_size*2)
        self.fc = nn.Linear(hidden_size*2, num_output)

    def forward(self, x):
        '''
        Args:
            x: (batch, time_step, input_size)
        Returns:
            num_output size
        '''

        x = self.encoder(x)
        x = self.drop_en(x)

        # r_out shape (batch, time_step, output_size)
        # None is for initial hidden state
        out_rnn, ht = self.rnn(x, None)

        if self.use_last:
            last_tensor=out_rnn[:, -1, :]
        else:
            # use mean
            last_tensor = out_rnn[:, :, :]
            last_tensor = torch.mean(last_tensor, dim=1)

        fc_input = self.bn2(last_tensor)
        out = self.fc(fc_input)
        return out