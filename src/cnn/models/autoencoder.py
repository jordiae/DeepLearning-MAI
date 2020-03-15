import torch.nn as nn
import torch.nn.functional as F
import torch


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        # channels in, channels out, kernel_size.
        # Defaults:  stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
        self.kernel_size = args.kernel_size
        self.dropout = True if args.dropout > 0.0 else False
        self.dropout_layer = nn.Dropout(args.dropout) if args.dropout > 0.0 else None
        self.batch_norm = not args.no_batch_norm
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_layers = nn.ModuleList([])
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
        self.dims = dims
        self.channels_out = channels_out

    def forward(self, x):
        for idx, conv_layer in enumerate(self.conv_layers):
            if self.batch_norm:
                conv, batch_norm = conv_layer
                x = F.relu(batch_norm(conv(x)))
            else:
                x = F.relu(conv_layer(x))
            if idx % self.n_conv_layers == 0:
                x = self.pool(x)
        return x


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        # channels in, channels out, kernel_size.
        # Defaults:  stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
        self.kernel_size = args.kernel_size
        self.dropout = True if args.dropout > 0.0 else False
        self.dropout_layer = nn.Dropout(args.dropout) if args.dropout > 0.0 else None
        self.batch_norm = not args.no_batch_norm
        self.conv_layers = nn.ModuleList([])
        self.input_size = args.encoder_dims_out
        self.stride = 2
        self.padding = self.kernel_size//2
        self.channels_in = args.encoder_channels_out
        #self.channels_first_in = args.initial_channels
        self.n_conv_layers = args.conv_layers
        self.channels_out = 3

        channels_in = self.channels_in
        dims = self.input_size
        for i in range(0, args.conv_blocks):
            if i == args.conv_blocks-1:
                channels_out = self.channels_out
            else:
                channels_out = channels_in//2
            #for j in range(0, args.conv_layers):
            dims = ((dims - self.kernel_size + 2 * self.padding) * self.stride + 1)
            conv = nn.ConvTranspose2d(channels_in, channels_out, self.kernel_size, stride=self.stride,
                                      padding=self.padding)
            if self.batch_norm:
                self.conv_layers.append(nn.ModuleList([conv, nn.BatchNorm2d(channels_out)]))
            else:
                self.conv_layers.append(conv)

            channels_in = channels_out
            dims *= 2

    def forward(self, x):
        for idx, conv_layer in enumerate(self.conv_layers):
            #if idx % self.n_conv_layers == self.n_conv_layers-1:
            output_size = list(map(lambda d: d * 2, list(x.shape)[-2:]))
            #else:
            #    output_size = list(x.shape)[-2:]
            if self.batch_norm:
                conv, batch_norm = conv_layer
                x = F.relu(batch_norm(conv(x, output_size=output_size)))
            else:
                x = F.relu(conv_layer(x, output_size=output_size))
        x = torch.tanh(x)
        return x


class PyramidAutoEncoder(nn.Module):
    def __init__(self, args):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(args)
        args.encoder_dims_out = self.encoder.dims
        args.encoder_channels_out = self.encoder.channels_out
        self.decoder = Decoder(args)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


'''
if __name__ == '__main__':
    x = torch.zeros((1, 3, 256, 256))

    class Args:
        pass
    args = Args()
    args.kernel_size = 3
    args.no_dropout = False
    args.no_batch_norm = False
    args.initial_channels = 8
    args.conv_layers = 2
    args.conv_blocks = 2
    ae = AutoEncoder(args)
    y = ae(x)
'''