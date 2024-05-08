import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(self.dropout1(x), 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(self.dropout2(x))
        output = F.log_softmax(x, dim=1)
        return output


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 10)
        self.linear2 = nn.Linear(10, 10)

    def forward(self, x: torch.Tensor):
        x = self.linear1(torch.flatten(x, 1))
        x = F.relu(x)
        x = self.linear2(x)
        output = F.log_softmax(x, dim=1)
        return output


class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, x: torch.Tensor):
        x = self.linear(torch.flatten(x, 1))
        output = F.log_softmax(x, dim=1)
        return output
