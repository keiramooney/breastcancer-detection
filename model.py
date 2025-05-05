"""
CNN model definition for BreakHis classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BreakHisClassifier(nn.Module):
    """
    A simple CNN architecture for classifying histopathology images
    from the BreakHis dataset.
    Expects input tensors of shape (batch_size, 3, 64, 64).

    Parameters:
        num_classes (int): the number of output classes.
    """

    def __init__(self, num_classes):
        super(BreakHisClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)

        # assuming input images are 64x64, after 3 poolings we get
        # 8x8 feature maps
        self.flatten_dim = 128 * 8 * 8

        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Forward pass of the CNN.

        Parameters:
            x (Tensor): Input tensor of shape (batch_size, 3, 64, 64)

        Returns:
            Tensor: Output logits of shape (batch_size, num_classes)
        """
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
