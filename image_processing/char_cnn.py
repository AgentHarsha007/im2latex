import numpy as np
import torch 
import torch.nn as nn
import os
import matplotlib
import torch.Functions as F
class Character_Cnn(nn.module):
    pass
class CNNModel(nn.Module):
    def __init__(self, num_classes, use_positional_encoding=True):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, return_logits=False):
        x = self.pool(F.relu(self.conv1(x)))  # → [B, 32, 12, 12]
        x = self.pool(F.relu(self.conv2(x)))  # → [B, 64, 6, 6]
        # Keep spatial info until now, then flatten for FC
        x = x.view(x.size(0), -1)  # → [B, 64*6*6]
        logits= F.relu(self.fc1(x))
        x = self.fc2(logits)  # → [B, num_classes]
        if return_logits:
            return x,logits
        return x