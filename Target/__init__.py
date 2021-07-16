# Python 3.8.5


import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, out):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 32*32 -> 28*28
        self.pool = nn.MaxPool2d(2, 2) # 28*28 -> 14*14
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        self.fc1 = nn.Linear(400, 200) # 16*5*5 = 400
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, out)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.fc1(x.view(-1, 400)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_last_hidden_layer(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.fc1(x.view(-1, 400)))
        x = self.fc2(x)
        return x