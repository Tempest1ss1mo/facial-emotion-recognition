"""
Dataset classes and data loading utilities for FER-2013.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from .augmentation import get_train_transforms, get_test_transforms


# Emotion labels
EMOTION_LABELS = {
    0: "angry",
    1: "disgust", 
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

LABEL_TO_IDX = {v: k for k, v in EMOTION_LABELS.items()}


class FERDataset(Dataset):
    """
    FER-2013 Dataset for facial emotion recognition.
    
    Expects data organized as:
        data/
        ├── train/
        │   ├── angry/
        │   ├── disgust/
        │   ├── fear/
        │   ├── happy/
        │   ├── sad/
        │   ├── surprise/
        │   └── neutral/
        └── test/
            └── ...
    
    Args:
        root_dir (str): Root directory of the dataset
        split (str): 'train', 'test', or 'val'
        transform: Optional transforms to apply
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Load image paths and labels
        self.images: List[Path] = []
        self.labels: List[int] = []
        
        split_dir = self.root_dir / split
        if not split_dir.exists():
            raise ValueError(f"Directory not found: {split_dir}")
        
        for emotion_name, label_idx in LABEL_TO_IDX.items():
            emotion_dir = split_dir / emotion_name
            if emotion_dir.exists():
                for img_path in emotion_dir.glob("*.jpg"):
                    self.images.append(img_path)
                    self.labels.append(label_idx)
                for img_path in emotion_dir.glob("*.png"):
                    self.images.append(img_path)
                    self.labels.append(label_idx)
                    
        print(f"Loaded {len(self.images)} images from {split} split")
        
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform
            image = transforms.ToTensor()(image)
            
        return image, label
    
    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of labels in dataset."""
        distribution = {}
        for label_idx in self.labels:
            emotion = EMOTION_LABELS[label_idx]
            distribution[emotion] = distribution.get(emotion, 0) + 1
        return distribution
    
    def get_sample_weights(self) -> torch.Tensor:
        """
        Calculate sample weights for weighted sampling.
        Useful for handling class imbalance.
        """
        class_counts = [0] * len(EMOTION_LABELS)
        for label in self.labels:
            class_counts[label] += 1
            
        weights = [1.0 / class_counts[label] for label in self.labels]
        return torch.tensor(weights)


class FERDatasetFromCSV(Dataset):
    """
    Load FER-2013 from the original CSV format.
    
    Args:
        csv_path (str): Path to fer2013.csv
        split (str): 'Training', 'PublicTest', or 'PrivateTest'
        transform: Optional transforms
    """
    
    def __init__(
        self,
        csv_path: str,
        split: str = "Training",
        transform: Optional[transforms.Compose] = None
    ):
        import pandas as pd
        
        self.transform = transform
        
        # Load CSV
        df = pd.read_csv(csv_path)
        df = df[df["Usage"] == split]
        
        # Parse pixels
        self.images = []
        self.labels = df["emotion"].values.tolist()
        
        for pixels in df["pixels"]:
            img_array = np.array([int(p) for p in pixels.split()], dtype=np.uint8)
            img_array = img_array.reshape(48, 48)
            self.images.append(img_array)
            
        print(f"Loaded {len(self.images)} images from {split}")
        
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = Image.fromarray(self.images[idx])
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
            
        return image, label


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    augment: bool = True,
    image_size: int = 48
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Root data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        augment: Whether to apply data augmentation
        image_size: Target image size
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Get transforms
    train_transform = get_train_transforms(image_size) if augment else get_test_transforms(image_size)
    test_transform = get_test_transforms(image_size)
    
    # Create datasets
    train_dataset = FERDataset(data_dir, split="train", transform=train_transform)
    
    # Check if validation split exists
    val_loader = None
    val_dir = Path(data_dir) / "val"
    if val_dir.exists():
        val_dataset = FERDataset(data_dir, split="val", transform=test_transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    test_dataset = FERDataset(data_dir, split="test", transform=test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # If no validation set, use test as validation
    if val_loader is None:
        val_loader = test_loader
        
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset classes...")
    
    # Test transforms
    from augmentation import get_train_transforms
    transform = get_train_transforms(48)
    print(f"Train transforms: {transform}")
