"""
Transfer Learning Models for Facial Emotion Recognition

Supports:
    - ResNet18
    - VGG16
    
Modes:
    - Feature Extraction: Freeze pretrained layers
    - Fine-tuning: Unfreeze and train all/some layers
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, List


class TransferModel(nn.Module):
    """
    Transfer learning model wrapper for facial emotion recognition.
    
    Args:
        model_name (str): Name of the pretrained model ('resnet18' or 'vgg16')
        num_classes (int): Number of emotion categories
        pretrained (bool): Whether to use pretrained weights
        dropout (float): Dropout probability for classifier
        freeze_features (bool): Whether to freeze feature extraction layers
    """
    
    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = 7,
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_features: bool = False
    ):
        super(TransferModel, self).__init__()
        
        self.model_name = model_name.lower()
        self.num_classes = num_classes
        
        # Load pretrained model
        if self.model_name == "resnet18":
            self.model = self._build_resnet18(pretrained, num_classes, dropout)
        elif self.model_name == "vgg16":
            self.model = self._build_vgg16(pretrained, num_classes, dropout)
        else:
            raise ValueError(f"Unsupported model: {model_name}. Choose 'resnet18' or 'vgg16'")
        
        # Freeze features if specified
        if freeze_features:
            self.freeze_features()
            
    def _build_resnet18(
        self, 
        pretrained: bool, 
        num_classes: int, 
        dropout: float
    ) -> nn.Module:
        """Build ResNet18 model with custom classifier."""
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        
        # Modify first conv layer to accept grayscale images
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        original_conv = model.conv1
        model.conv1 = nn.Conv2d(
            1, 64, 
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # Initialize new conv layer with mean of pretrained weights
        if pretrained:
            with torch.no_grad():
                model.conv1.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Replace final FC layer
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )
        
        return model
    
    def _build_vgg16(
        self, 
        pretrained: bool, 
        num_classes: int, 
        dropout: float
    ) -> nn.Module:
        """Build VGG16 model with custom classifier."""
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vgg16(weights=weights)
        
        # Modify first conv layer for grayscale
        original_conv = model.features[0]
        model.features[0] = nn.Conv2d(
            1, 64,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding
        )
        
        if pretrained:
            with torch.no_grad():
                model.features[0].weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Replace classifier
        model.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),  # Adjusted for 48x48 input
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes)
        )
        
        # Adjust adaptive pool for smaller input
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        return model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def freeze_features(self) -> None:
        """Freeze all feature extraction layers."""
        if self.model_name == "resnet18":
            # Freeze everything except fc layer
            for name, param in self.model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False
        elif self.model_name == "vgg16":
            # Freeze features
            for param in self.model.features.parameters():
                param.requires_grad = False
                
    def unfreeze_features(self) -> None:
        """Unfreeze all layers for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True
            
    def unfreeze_top_layers(self, num_layers: int = 2) -> None:
        """
        Unfreeze top N layers for gradual fine-tuning.
        
        Args:
            num_layers: Number of top layers to unfreeze
        """
        if self.model_name == "resnet18":
            layers = [
                self.model.layer4,
                self.model.layer3,
                self.model.layer2,
                self.model.layer1,
            ]
            for layer in layers[:num_layers]:
                for param in layer.parameters():
                    param.requires_grad = True
                    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def count_parameters(self) -> dict:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable
        }


def get_transfer_model(
    model_name: str = "resnet18",
    num_classes: int = 7,
    pretrained: bool = True,
    mode: str = "finetune",
    dropout: float = 0.5
) -> TransferModel:
    """
    Factory function to get transfer learning model.
    
    Args:
        model_name: Name of pretrained model
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        mode: 'feature_extraction' or 'finetune'
        dropout: Dropout probability
        
    Returns:
        Configured TransferModel
    """
    freeze = mode == "feature_extraction"
    model = TransferModel(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        freeze_features=freeze
    )
    return model


if __name__ == "__main__":
    # Test ResNet18
    print("Testing ResNet18...")
    model = get_transfer_model("resnet18", num_classes=7, mode="finetune")
    x = torch.randn(4, 1, 48, 48)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {model.count_parameters()}")
    
    print("\n" + "="*50 + "\n")
    
    # Test VGG16
    print("Testing VGG16...")
    model = get_transfer_model("vgg16", num_classes=7, mode="feature_extraction")
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {model.count_parameters()}")
