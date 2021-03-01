import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, 3)  ## inchannel , outchannel , kernel size ..
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv1_bn = nn.BatchNorm2d(8)

        self.drp = nn.Dropout2d(0.1)

        self.conv2 = nn.Conv2d(8, 16, 3)
        self.pool2 = nn.MaxPool2d(4, stride=2)
        self.conv2_bn = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, 5)
        self.conv3_bn = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32 * 12 * 82, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv1_bn(self.pool1(F.relu(self.conv1(x))))
        x = self.drp(x)

        x = self.conv2_bn(self.pool1(F.relu(self.conv2(x))))
        x = self.drp(x)

        x = self.conv3_bn(self.pool2(F.relu(self.conv3(x))))
        x = self.drp(x)
        # x = self.drp(self.pool1(F.relu(self.conv4(x))))
        # x = self.drp(self.pool2(F.relu(self.conv5(x))))
        # size = torch.flatten(x).shape[0]
        x = x.view(-1, 32 * 12 * 82)
        # x = x.unsqueeze_(1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)