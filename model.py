import torch
from PIL import Image, ImageDraw
import torchvision.transforms as transforms

# Define the AdvancedCNN class here directly
import torch
import torch.nn as nn

# SEBlock definition
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = torch.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

# ResidualBlock with SEBlock for enhanced features
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.se = SEBlock(out_channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return torch.relu(out)

# Advanced CNN model definition
class AdvancedCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(AdvancedCNN, self).__init__()
        self.layer1 = self._make_layer(3, 64)
        self.layer2 = self._make_layer(64, 128)
        self.layer3 = self._make_layer(128, 256)
        self.layer4 = self._make_layer(256, 512)

        # Adjust the input size of the fully connected layer based on the flattened output
        self.fc = nn.Linear(512 * 45 * 45, num_classes)  # Adjusted from 512 * 4 * 4 to 512 * 80 * 80

    def _make_layer(self, in_channels, out_channels):
        downsample = None
        if in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, downsample=downsample),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Debugging shapes
        # print(f"Shape before flattening: {x.shape}")

        # Flatten the feature map
        x = x.view(x.size(0), -1)

        # print(f"Shape after flattening: {x.shape}")

        x = self.fc(x)
        return x