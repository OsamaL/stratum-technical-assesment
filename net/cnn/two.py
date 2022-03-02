import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, net_name):
        super().__init__()
        self.net_name = net_name
        self.conv1 = nn.Conv2d(1, 20, 3, 1)
        self.conv2 = nn.Conv2d(20, 50, 3, 1)
        self.conv3 = nn.Conv2d(50, 100, 3, 1)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x = F.max_pool2d(x1, 2, 2)
        x2 = F.relu(self.conv2(x))
        x = F.max_pool2d(x2, 2, 2)
        x3 = F.relu(self.conv3(x))
        x = F.max_pool2d(x3, 2, 2)
        x = x.view(-1, 100)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, x1, x3
    
    def get_name(self):
        return self.net_name
    
def create_net(net_name="no_name_net"):
    return Net(net_name)