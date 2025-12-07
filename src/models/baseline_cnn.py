"""
Baseline CNN Model (LeNet-inspired) for Facial Emotion Recognition

Architecture:
    Input (48×48×1) → Conv → ReLU → MaxPool → Conv → ReLU → MaxPool → 
    Conv → ReLU → MaxPool → FC → ReLU → Dropout → FC → Softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """
    LeNet-inspired CNN for facial emotion classification.
    
    Args:
        num_classes (int): Number of emotion categories (default: 7)
        dropout (float): Dropout probability (default: 0.5)
        in_channels (int): Number of input channels (default: 1 for grayscale)
    """
    
    def __init__(self, num_classes: int = 7, dropout: float = 0.5, in_channels: int = 1):
        super(BaselineCNN, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        # After 4 pooling operations: 48 -> 24 -> 12 -> 6 -> 3
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 48, 48)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Conv block 1: 48x48 -> 24x24
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2: 24x24 -> 12x12
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3: 12x12 -> 6x6
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Conv block 4: 6x6 -> 3x3
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the final classification layer.
        Useful for visualization and analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor of shape (batch_size, 256)
        """
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class SimpleCNN(nn.Module):
    """
    Simpler CNN architecture for quick experiments.
    """
    
    def __init__(self, num_classes: int = 7, dropout: float = 0.5):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_baseline_model(num_classes: int = 7, dropout: float = 0.5, simple: bool = False) -> nn.Module:
    """
    Factory function to get baseline model.
    
    Args:
        num_classes: Number of output classes
        dropout: Dropout probability
        simple: Whether to use simpler architecture
        
    Returns:
        CNN model
    """
    if simple:
        return SimpleCNN(num_classes=num_classes, dropout=dropout)
    return BaselineCNN(num_classes=num_classes, dropout=dropout)


if __name__ == "__main__":
    # Test the model
    model = BaselineCNN(num_classes=7)
    print(model)
    
    # Test forward pass
    x = torch.randn(4, 1, 48, 48)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
