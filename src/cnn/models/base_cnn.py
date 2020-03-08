import torch.nn as nn
import torch.nn.functional as F


class BaseCNN(nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()
        # channels in, channels out, kernel_size.
        # Defaults:  stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.fc1 = nn.Linear(12*61*61, 120)
        self.fc2 = nn.Linear(120, 67)  # 67 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 12*61*61)  # view -> reshape
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # softmax not required (done by cross-entropy criterion):
        # "This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
        # https://pytorch.org/docs/stable/nn.html#crossentropyloss
        return x
