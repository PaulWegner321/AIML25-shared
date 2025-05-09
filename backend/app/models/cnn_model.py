import torch
import torch.nn as nn
import torch.nn.functional as F

class ASLCNNModel(nn.Module):
    def __init__(self):
        super(ASLCNNModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Input: 1 channel (grayscale)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(1152, 512)  # 1152 matches the saved weights
        self.fc2 = nn.Linear(512, 26)  # 26 classes for A-Z
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convert input to grayscale if it's RGB (3 channels)
        if x.size(1) == 3:
            # Convert RGB to grayscale using standard coefficients
            x = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        
        # Apply convolutional layers with ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        # Flatten the feature maps
        x = x.view(-1, 1152)  # Flatten to match fc1 input size

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x 