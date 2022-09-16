import torch
import torch.nn as nn
import torch.nn.functional as F 
from splearn.nn.modules.conv2d import Conv2d

class SignalCNN(nn.Module):
    """
    Convolutional correlation analysis for enhancing the performance of SSVEP-based brain-computer interface
    https://ieeexplore.ieee.org/abstract/document/9261605/
    https://github.com/yaoli90/Conv-CA/blob/main/convca.py
    """
    def __init__(self, num_channel=10):
        super().__init__()
        self.conv1 = Conv2d(1, 16, (9, num_channel))
        self.conv2 = Conv2d(16, 1, (1, num_channel))
        self.conv3 = Conv2d(1, 1, (1, num_channel), padding=(0,0))
        self.dropout = nn.Dropout(p=0.75)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        return x

class ReferenceCNN(nn.Module):
    def __init__(self, num_channel=10, num_freq=40):
        super().__init__()
        self.conv1 = Conv2d(num_channel, num_freq, (9, 1))
        self.conv2 = Conv2d(num_freq, 1, (9, 1))
        self.dropout = nn.Dropout(p=0.15)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = torch.squeeze(x)
        return x

class Corr(nn.Module):
    def __init__(self):
        super().__init__()

    def batch_dot(self, x_batch, y_batch):
        return torch.sum(x_batch * y_batch, axis=1)
    
    def forward(self, x, t):
        corr_xt = self.batch_dot(x.unsqueeze(-1),t) # [?,cl]
        corr_xx = self.batch_dot(x,x) # [?]
        corr_xx = corr_xx.unsqueeze(-1) # [?,1]
        corr_tt = torch.sum(t*t,axis=1) # [?,cl]
        self.corr = corr_xt/torch.sqrt(corr_tt)/torch.sqrt(corr_xx)
        return self.corr

class ConvCA(nn.Module):
    def __init__(self, num_channel=10, num_classes=4, **kwargs):
        super().__init__()
        self.signal_cnn = SignalCNN(num_channel)
        self.reference_cnn = ReferenceCNN(num_channel, num_classes)
        self.correlation = Corr()
        self.dense = nn.Linear(in_features=num_classes, out_features=num_classes)
    
    def forward(self, x, ref):
        x = x.unsqueeze(-1)
        x = torch.transpose(x, 3, 1)

        x1 = self.signal_cnn(x)
        x2 = self.reference_cnn(ref)
        x = self.correlation(x1, x2)
        x = self.dense(x)
        return x
