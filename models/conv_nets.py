#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 23:47:02 2020

@author: orivlin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np




class ResNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.identity_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.identity_bn(self.identity(x))
        out = F.relu(out)
        return out


class Resnet(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, final_activation='sigmoid'):
        super(Resnet, self).__init__()
        self.final_activation = final_activation
        self.net = nn.Sequential(ResNetBlock(in_channels, 32),
                                 nn.MaxPool2d((2, 2)),
                                 ResNetBlock(32, 64),
                                 nn.MaxPool2d((2, 2)),
                                 ResNetBlock(64, 64),
                                 ResNetBlock(64, 64),
                                 nn.Upsample(scale_factor=2, mode='bilinear'),
                                 ResNetBlock(64, 32),
                                 nn.Upsample(scale_factor=2, mode='bilinear'),
                                 ResNetBlock(32, out_channels)
                                 )

    def forward(self, x):
        out = self.net(x)
        if self.final_activation == 'sigmoid':
            return torch.sigmoid(out)
        elif self.final_activation == 'tanh':
            return torch.tanh(out)
        elif self.final_activation == 'none':
            return out
    
class Conv2Vec(nn.Module):
    
    def __init__(self, in_channels, out_dim):
        super(Conv2Vec, self).__init__()
        self.backbone = Resnet(in_channels=in_channels, out_channels=32, final_activation='none')
        self.frontend = torch.nn.Sequential(nn.MaxPool2d((2, 2)),
                                            nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                            nn.MaxPool2d((2, 2)),
                                            nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                            nn.Flatten(1),
                                            nn.Linear(800, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 1))
        
    def forward(self, x):
        x = self.backbone(x)
        return self.frontend(x)