"""
Evaluation metrics for facial emotion recognition.
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging method ('weighted', 'macro', 'micro')
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    return metrics


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = False
) -> np.ndarray:
    """
    Calculate confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        normalize: Whether to normalize
        
    Returns:
        Confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    return cm


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_dict: bool = False
) -> str:
    """
    Generate classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        output_dict: Return dictionary instead of string
        
    Returns:
        Classification report
    """
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=output_dict,
        zero_division=0
    )


def per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> Dict[str, float]:
    """
    Calculate per-class accuracy.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Dictionary mapping class names to accuracy
    """
    cm = confusion_matrix(y_true, y_pred)
    per_class = cm.diagonal() / cm.sum(axis=1)
    
    return {name: acc for name, acc in zip(class_names, per_class)}


def top_k_accuracy(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    k: int = 3
) -> float:
    """
    Calculate top-k accuracy.
    
    Args:
        y_true: Ground truth labels
        y_pred_probs: Predicted probabilities
        k: Top k predictions to consider
        
    Returns:
        Top-k accuracy
    """
    top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -k:]
    correct = np.array([y in top_k for y, top_k in zip(y_true, top_k_preds)])
    return correct.mean()


class MetricsTracker:
    """
    Track metrics during training.
    """
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
    def update(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float = None
    ) -> bool:
        """
        Update metrics for an epoch.
        
        Args:
            epoch: Current epoch
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss
            val_acc: Validation accuracy
            lr: Current learning rate
            
        Returns:
            True if validation accuracy improved
        """
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        
        if lr is not None:
            self.history['learning_rate'].append(lr)
            
        improved = val_acc > self.best_val_acc
        if improved:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            
        return improved
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get training history."""
        return self.history
    
    def get_best(self) -> Tuple[int, float]:
        """Get best epoch and accuracy."""
        return self.best_epoch, self.best_val_acc
    
    def print_epoch_summary(self, epoch: int) -> None:
        """Print summary for an epoch."""
        print(f"Epoch {epoch}:")
        print(f"  Train Loss: {self.history['train_loss'][-1]:.4f} | "
              f"Train Acc: {self.history['train_acc'][-1]:.2f}%")
        print(f"  Val Loss: {self.history['val_loss'][-1]:.4f} | "
              f"Val Acc: {self.history['val_acc'][-1]:.2f}%")
        
    def print_best_summary(self) -> None:
        """Print best results summary."""
        print(f"\nBest Results:")
        print(f"  Epoch: {self.best_epoch}")
        print(f"  Validation Accuracy: {self.best_val_acc:.2f}%")


if __name__ == "__main__":
    # Test metrics
    y_true = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 3, 4, 5, 6, 1, 1, 2])
    
    metrics = calculate_metrics(y_true, y_pred)
    print("Metrics:", metrics)
    
    class_names = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    report = get_classification_report(y_true, y_pred, class_names)
    print("\nClassification Report:\n", report)
