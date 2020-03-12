import torch.nn as nn
import torch.nn.functional as F


class AlbertCNN(nn.Module):
    def __init__(self):
        super(AlbertCNN, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 32, 5)
        self.conv1_2 = nn.Conv2d(32, 32, 5)

        self.conv2_1 = nn.Conv2d(32, 64, 5)
        self.conv2_2 = nn.Conv2d(64, 64, 5)

        self.conv3_1 = nn.Conv2d(64, 128, 3)
        self.conv3_2 = nn.Conv2d(128, 128, 3)

        self.conv4_1 = nn.Conv2d(128, 256, 2)
        self.conv4_2 = nn.Conv2d(256, 256, 2)

        self.conv5_1 = nn.Conv2d(256, 384, 2)
        self.conv5_2 = nn.Conv2d(384, 384, 2)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(9600, 1500)
        self.fc2 = nn.Linear(1500, 120)
        self.fc3 = nn.Linear(120, 67)  # 67 classes

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = self.pool(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = self.pool(x)

        x = x.view(-1, 9600)  # view -> reshape

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
