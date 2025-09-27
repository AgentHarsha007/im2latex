import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import copy # To save the best model weights
# =============================================================================
# 1. Data Loading & Transformation
# =============================================================================
# Define the transformations for the images.
# For a real project, you'd apply data augmentation transforms here for the
# training set to make the model more robust.
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure images are grayscale
    transforms.Resize((24, 24)),                  # Resize all images to a consistent 24x24
    transforms.ToTensor(),                        # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))          # Normalize tensors for better training stability
])
class ResidualBlock(nn.Module):
    """
    A residual block, the core component of ResNet architectures.
    It allows the model to learn an identity function, which helps with
    the vanishing gradient problem in deep networks.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # First convolutional layer in the block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Second convolutional layer in the block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to match dimensions if the stride or number of channels changes
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Add the input to the output (the "residual" connection)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ImprovedCharCNN(nn.Module):
    """ The main CNN model incorporating residual blocks for improved performance. """
    def __init__(self, num_classes):
        super(ImprovedCharCNN, self).__init__()
        # Initial convolutional layer to extract low-level features
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # A series of residual blocks to build a deep feature hierarchy
        self.res1 = ResidualBlock(64, 128, stride=2)   # Downsamples from 24x24 to 12x12
        self.res2 = ResidualBlock(128, 256, stride=2)  # Downsamples from 12x12 to 6x6
        self.res3 = ResidualBlock(256, 512, stride=2)  # Downsamples from 6x6 to 3x3

        # Adaptive pooling layer to flatten the feature maps for the classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4) # Dropout for regularization

        # Fully connected layers for classification
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1) # Flatten the output for the FC layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
