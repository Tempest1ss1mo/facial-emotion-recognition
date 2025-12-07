"""
Visualization utilities for training analysis and results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: tuple = (14, 5)
) -> None:
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary with keys 'train_loss', 'val_loss', 
                 'train_acc', 'val_acc'
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Training Loss', markersize=4)
    axes[0].plot(epochs, history['val_loss'], 'r-o', label='Validation Loss', markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy', markersize=4)
    axes[1].plot(epochs, history['val_acc'], 'r-o', label='Validation Accuracy', markersize=4)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
    normalize: bool = True,
    cmap: str = 'Blues'
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_path: Path to save the plot
        figsize: Figure size
        normalize: Whether to normalize values
        cmap: Colormap
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        cbar_kws={'shrink': 0.8}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    plt.close()


def plot_sample_predictions(
    images: np.ndarray,
    true_labels: List[int],
    pred_labels: List[int],
    class_names: List[str],
    confidences: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    num_samples: int = 16,
    figsize: tuple = (16, 16)
) -> None:
    """
    Plot sample predictions with true and predicted labels.
    
    Args:
        images: Array of images
        true_labels: Ground truth labels
        pred_labels: Predicted labels
        class_names: List of class names
        confidences: Prediction confidences
        save_path: Path to save the plot
        num_samples: Number of samples to display
        figsize: Figure size
    """
    num_samples = min(num_samples, len(images))
    nrows = int(np.ceil(np.sqrt(num_samples)))
    ncols = int(np.ceil(num_samples / nrows))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    for idx in range(num_samples):
        ax = axes[idx]
        
        # Handle different image formats
        img = images[idx]
        if len(img.shape) == 3 and img.shape[0] in [1, 3]:
            img = img.transpose(1, 2, 0)
        if img.shape[-1] == 1:
            img = img.squeeze()
            
        ax.imshow(img, cmap='gray')
        
        true_name = class_names[true_labels[idx]]
        pred_name = class_names[pred_labels[idx]]
        correct = true_labels[idx] == pred_labels[idx]
        
        title = f"True: {true_name}\nPred: {pred_name}"
        if confidences is not None:
            title += f"\nConf: {confidences[idx]:.2%}"
            
        color = 'green' if correct else 'red'
        ax.set_title(title, color=color, fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Sample Predictions', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Sample predictions saved to {save_path}")
    
    plt.show()
    plt.close()


def plot_label_distribution(
    distribution: Dict[str, int],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
    title: str = "Label Distribution"
) -> None:
    """
    Plot distribution of labels in dataset.
    
    Args:
        distribution: Dictionary mapping labels to counts
        save_path: Path to save the plot
        figsize: Figure size
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    labels = list(distribution.keys())
    counts = list(distribution.values())
    
    colors = sns.color_palette("husl", len(labels))
    bars = ax.bar(labels, counts, color=colors, edgecolor='black', alpha=0.8)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    ax.set_xlabel('Emotion', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Label distribution saved to {save_path}")
    
    plt.show()
    plt.close()


def plot_per_class_accuracy(
    accuracies: Dict[str, float],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot per-class accuracy.
    
    Args:
        accuracies: Dictionary mapping class names to accuracy
        save_path: Path to save plot
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    classes = list(accuracies.keys())
    accs = list(accuracies.values())
    
    colors = ['green' if a >= 0.7 else 'orange' if a >= 0.5 else 'red' for a in accs]
    bars = ax.barh(classes, accs, color=colors, edgecolor='black', alpha=0.8)
    
    # Add value labels
    for bar, acc in zip(bars, accs):
        ax.text(
            acc + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{acc:.1%}',
            ha='left',
            va='center',
            fontsize=10
        )
    
    ax.set_xlim(0, 1.1)
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_ylabel('Emotion', fontsize=12)
    ax.set_title('Per-Class Accuracy', fontsize=14)
    ax.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5, label='70% threshold')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Per-class accuracy saved to {save_path}")
    
    plt.show()
    plt.close()


def plot_learning_rate_schedule(
    lrs: List[float],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 5)
) -> None:
    """
    Plot learning rate schedule over epochs.
    
    Args:
        lrs: List of learning rates
        save_path: Path to save plot
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    epochs = range(1, len(lrs) + 1)
    ax.plot(epochs, lrs, 'b-o', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    plt.close()


if __name__ == "__main__":
    # Test visualizations
    print("Visualization module loaded successfully")
    
    # Test label distribution
    test_dist = {
        "angry": 3995,
        "disgust": 436,
        "fear": 4097,
        "happy": 7215,
        "sad": 4830,
        "surprise": 3171,
        "neutral": 4965
    }
    plot_label_distribution(test_dist, title="FER-2013 Label Distribution")
