import torch.nn as nn
import torch.nn.functional as F
import torch


class PyramidCNN(nn.Module):
    def __init__(self, args):
        super(PyramidCNN, self).__init__()
        # channels in, channels out, kernel_size.
        # Defaults:  stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
        self.kernel_size = args.kernel_size
        self.dropout = args.dropout
        self.dropout_ = nn.Dropout()
        self.batch_norm = args.batch_norm
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_layers = nn.ModuleList([])
        self.n_classes = 67
        self.input_size = 256
        self.stride = 1
        self.padding = self.kernel_size//2
        self.channels_in = 3
        self.channels_first_in = args.initial_channels
        self.n_conv_layers = args.conv_layers

        channels_in = self.channels_in
        dims = self.input_size
        for i in range(0, args.conv_blocks):
            if i == 0:
                channels_out = self.channels_first_in
            else:
                channels_out = channels_in*2
            for j in range(0, args.conv_layers):
                dims = ((dims - self.kernel_size + 2 * self.padding) * self.stride + 1)
                conv = nn.Conv2d(channels_in, channels_out, self.kernel_size, stride=self.stride, padding=self.padding)
                if self.batch_norm:
                    self.conv_layers.append(nn.ModuleList([conv, nn.BatchNorm2d(channels_out)]))
                else:
                    self.conv_layers.append(conv)
                channels_in = channels_out
            dims //= 2
        self.pool_channels = nn.MaxPool2d(dims, dims)
        self.fc_layers = nn.ModuleList([])
        # square image, so same dimensions
        #dims = (self.input_size - self.kernel_size + 2*self.padding)*self.stride + 1
        #dims_in = ((dims//2)**2) * channels_out
        #dims_in = dims * channels_out
        #self.dims_in_fc = dims_in#//channels_out
        #self.conv1x1 = nn.Conv2d(dims_in, 1, 1)
        dims_in = channels_out
        self.dims_in_fc = dims_in
        for i in range(0, args.fc_layers-1):
            dims_out = self.dims_in_fc //2  #dims_in//4
            fc = nn.Linear(dims_in, dims_out)
            if self.batch_norm:
                self.fc_layers.append(nn.ModuleList([fc, nn.BatchNorm1d(dims_out)]))
            else:
                self.fc_layers.append(fc)
            dims_in = dims_out
        self.fc_layers.append(nn.Linear(dims_in, self.n_classes))

    def forward(self, x):
        for idx, conv_layer in enumerate(self.conv_layers):
            if self.batch_norm:
                conv, batch_norm = conv_layer
                x = conv(x)
                x = batch_norm(x)
                x = F.relu(x)
                #x = F.relu(batch_norm(conv(x)))
            else:
                x = F.relu(conv_layer(x))
            if idx % self.n_conv_layers == 0:
                x = self.pool(x)
        #x = self.conv1x1(x)
        x = torch.squeeze(self.pool_channels(x))
        x = x.view(-1, self.dims_in_fc)
        for fc_layer in self.fc_layers[:-1]:
            if self.batch_norm:
                fc, batch_norm = fc_layer
                x = F.relu(batch_norm(fc(x)))
            else:
                x = F.relu(fc_layer(x))
            x = self.dropout_(x)
        x = self.fc_layers[-1](x)
        return x
        # softmax not required (done by cross-entropy criterion):
        # "This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
        # https://pytorch.org/docs/stable/nn.html#crossentropyloss
