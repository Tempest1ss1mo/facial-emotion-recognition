"""
Data augmentation transforms for FER-2013 dataset.

Augmentation strategies:
    - Horizontal flip
    - Random rotation
    - Random zoom/crop
    - Brightness/contrast adjustment
"""

from torchvision import transforms
from typing import Optional


def get_train_transforms(
    image_size: int = 48,
    rotation_degrees: int = 10,
    brightness: float = 0.2,
    contrast: float = 0.2,
    horizontal_flip_prob: float = 0.5,
    zoom_range: float = 0.1
) -> transforms.Compose:
    """
    Get training data transforms with augmentation.
    
    Args:
        image_size: Target image size
        rotation_degrees: Max rotation degrees
        brightness: Brightness adjustment factor
        contrast: Contrast adjustment factor
        horizontal_flip_prob: Probability of horizontal flip
        zoom_range: Random zoom range
        
    Returns:
        Composed transforms
    """
    # Calculate crop size for zoom effect
    crop_size = int(image_size * (1 - zoom_range))
    
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
        transforms.RandomRotation(degrees=rotation_degrees),
        transforms.RandomResizedCrop(
            size=image_size,
            scale=(1 - zoom_range, 1.0),
            ratio=(0.95, 1.05)
        ),
        transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ]
    
    return transforms.Compose(transform_list)


def get_test_transforms(image_size: int = 48) -> transforms.Compose:
    """
    Get test/validation data transforms (no augmentation).
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transforms
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]
    
    return transforms.Compose(transform_list)


def get_gradcam_transforms(image_size: int = 48) -> transforms.Compose:
    """
    Get transforms for Grad-CAM visualization.
    Similar to test transforms but without normalization for visualization.
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transforms
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    
    return transforms.Compose(transform_list)


class CustomAugmentation:
    """
    Custom augmentation class for more control over transforms.
    """
    
    def __init__(
        self,
        image_size: int = 48,
        config: Optional[dict] = None
    ):
        self.image_size = image_size
        self.config = config or {}
        
        # Default config
        self.rotation = self.config.get("rotation_degrees", 10)
        self.h_flip = self.config.get("horizontal_flip", True)
        self.brightness = self.config.get("brightness", 0.2)
        self.contrast = self.config.get("contrast", 0.2)
        self.zoom = self.config.get("zoom_range", 0.1)
        
    def get_train_transform(self) -> transforms.Compose:
        """Get training transforms based on config."""
        transform_list = [transforms.Resize((self.image_size, self.image_size))]
        
        if self.h_flip:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
            
        if self.rotation > 0:
            transform_list.append(transforms.RandomRotation(degrees=self.rotation))
            
        if self.zoom > 0:
            transform_list.append(
                transforms.RandomResizedCrop(
                    size=self.image_size,
                    scale=(1 - self.zoom, 1.0)
                )
            )
            
        if self.brightness > 0 or self.contrast > 0:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=self.brightness,
                    contrast=self.contrast
                )
            )
            
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        return transforms.Compose(transform_list)
    
    def get_test_transform(self) -> transforms.Compose:
        """Get test transforms."""
        return get_test_transforms(self.image_size)


if __name__ == "__main__":
    # Test transforms
    print("Train transforms:")
    print(get_train_transforms())
    print("\nTest transforms:")
    print(get_test_transforms())
