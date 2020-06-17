#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 23:47:02 2020

@author: orivlin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical



def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


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
                                            nn.Linear(256, out_dim))
        
    def forward(self, x):
        x = self.backbone(x)
        return self.frontend(x)


class DiscreteActor(nn.Module):

    def __init__(self, in_channels, num_actions):
        super(DiscreteActor, self).__init__()
        self.model = Conv2Vec(in_channels, num_actions)

    def forward(self, x):
        return F.softmax(self.model(x), dim=1)

    def act(self, x, deterministic=False):
        with torch.no_grad():
            action_probs = self.forward(x)
            if deterministic:
                actions = torch.argmax(action_probs, dim=1, keepdim=True)
            else:
                action_dist = Categorical(action_probs)
                actions = action_dist.sample().view(-1, 1)
        return actions.squeeze().cpu().item()

    def sample(self, x):
        action_probs = self.forward(x)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        # Avoid numerical instability.
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs


class DiscreteCritic(nn.Module):
    def __init__(self, num_channels, num_actions):
        super().__init__()
        self.Q1 = Conv2Vec(num_channels, num_actions)
        self.Q2 = Conv2Vec(num_channels, num_actions)

    def forward(self, states):
        q1 = self.Q1(states)
        q2 = self.Q2(states)
        return q1, q2
