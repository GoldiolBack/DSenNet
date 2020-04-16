import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self, input_shape=((4, 32, 32), (6, 16, 16)), feature_size=128, kernel_size=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, feature_size, kernel_size, 1)
        self.conv2 = nn.conv2d(feature_size, input_shape[-1][0], kernel_size, 1)
        self.rBlock = ResBlock(feature_size, kernel_size)

    def forward(self, x, num_layers=6):
#       input10 = ...
#       input20 = ...
        x = torch.cat((input10, input20), 0)
        x = self.conv1(x)
        x = F.relu(x)
        for i in range(num_layers):
            x = self.rBlock(x)
        x = self.conv2(x)
        x += input20
        return x


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ResBlock, self).__init__()
        self.conv3 = nn.Conv2d(channels, channels, kernel_size, 1)

    def forward(self, x, scale=0.1):
        tmp = self.conv3(x)
        tmp = F.relu(tmp)
        tmp = self.conv3(tmp)
        tmp = tmp * scale
        tmp += x
        return tmp
    
    
