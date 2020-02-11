import torch
import torch.nn as nn
import torch.nn.functional as F


class DispatchNet(nn.Module):

    def __init__(self):
        super(DispatchNet, self).__init__()

        self.conv1 = nn.Conv2d(4, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 1)
        self.conv5 = nn.Conv2d(128, 1, 1)

        self.fc = nn.Linear(15 * 15, 15 * 15)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
