# -*- coding: utf-8 -*-
"""EEGNet: Compact Convolutional Neural Network (Compact-CNN) https://arxiv.org/pdf/1803.04566.pdf
"""
import torch
from torch import nn
from splearn.nn.modules.conv2d import SeparableConv2d


class CompactEEGNet(nn.Module):
    """
    EEGNet: Compact Convolutional Neural Network (Compact-CNN)
    https://arxiv.org/pdf/1803.04566.pdf
    """
    def __init__(self, num_channel=10, num_classes=4, signal_length=1000, f1=96, f2=96, d=1):
        super().__init__()
        
        self.signal_length = signal_length
        
        # layer 1
        self.conv1 = nn.Conv2d(1, f1, (1, signal_length), padding=(0,signal_length//2))
        self.bn1 = nn.BatchNorm2d(f1)
        self.depthwise_conv = nn.Conv2d(f1, d*f1, (num_channel, 1), groups=f1)
        self.bn2 = nn.BatchNorm2d(d*f1)
        self.avgpool1 = nn.AvgPool2d((1,4))
        
        # layer 2
        self.separable_conv = SeparableConv2d(
            in_channels=f1, 
            out_channels=f2, 
            kernel_size=(1,16)
        )
        self.bn3 = nn.BatchNorm2d(f2)
        self.avgpool2 = nn.AvgPool2d((1,8))
        
        # layer 3
        self.fc = nn.Linear(in_features=f2*(signal_length//32), out_features=num_classes)
        
        self.dropout = nn.Dropout(p=0.5)
        self.elu = nn.ELU()
        
    def forward(self, x):
        
        # layer 1
        x = torch.unsqueeze(x,1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout(x)
                
        # layer 2
        x = self.separable_conv(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout(x)
                
        # layer 3
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        
        return x
