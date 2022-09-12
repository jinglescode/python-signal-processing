import torch
from torch import nn
from splearn.nn.modules.conv2d import Conv2d


class TimeDomainBasedCNN(nn.Module):
    """
    Filter Bank Convolutional Neural Network for Short Time-Window Steady-State Visual Evoked Potential Classification
    https://ieeexplore.ieee.org/abstract/document/9632600
    """
    def __init__(self, num_classes=4, signal_length=1000):
        super().__init__()

        self.conv1 = Conv2d(1, 16, (9, 1), padding=(0,0))
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = Conv2d(16, 16, (1, signal_length), padding="SAME", stride=5)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = Conv2d(16, 16, (1, 5), padding=(0,0))
        self.bn3 = nn.BatchNorm2d(16)

        k2 = int(signal_length/5)-4
        self.conv4 = Conv2d(16, 32, (1, k2), padding=(0,0))
        self.bn4 = nn.BatchNorm2d(32)

        self.fc = nn.Linear(32, out_features=num_classes)
        
        self.dropout = nn.Dropout(p=0.4)
        self.elu = nn.ELU()
        
    def forward(self, x):

        x = torch.unsqueeze(x,1)
        
        # the first convolution layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        
        # the second convolution layer
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu(x)
        
        # the third convolution layer
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu(x)
        
        # the fourth convolution layer
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.elu(x)
        # flatten used to reduce the dimension of the features
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x