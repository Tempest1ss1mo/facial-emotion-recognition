"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

Visualizes which regions of the face the model focuses on for predictions.
Reference: https://arxiv.org/abs/1610.02391
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import cv2


class GradCAM:
    """
    Grad-CAM implementation for visualizing CNN attention.
    
    Args:
        model: PyTorch model
        target_layer: Target convolutional layer for visualization
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward and backward hooks."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
        
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (None = use predicted class)
            
        Returns:
            CAM heatmap as numpy array
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Calculate weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Generate CAM
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input size
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        return cam.squeeze().cpu().numpy()
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Alias for generate_cam."""
        return self.generate_cam(input_tensor, target_class)


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Get the target layer for Grad-CAM based on model architecture.
    
    Args:
        model: PyTorch model
        model_name: Name of the model architecture
        
    Returns:
        Target layer for visualization
    """
    model_name = model_name.lower()
    
    if model_name == "baseline":
        # Last conv layer before FC
        return model.conv4
    elif model_name == "resnet18":
        # Last layer of ResNet
        return model.model.layer4[-1]
    elif model_name == "vgg16":
        # Last conv layer of VGG
        return model.model.features[-1]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def visualize_gradcam(
    model: nn.Module,
    image: torch.Tensor,
    original_image: np.ndarray,
    target_layer: nn.Module,
    target_class: Optional[int] = None,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate and visualize Grad-CAM overlay.
    
    Args:
        model: PyTorch model
        image: Preprocessed input tensor
        original_image: Original image for overlay
        target_layer: Target layer for CAM
        target_class: Target class (None = predicted)
        alpha: Overlay transparency
        colormap: OpenCV colormap
        save_path: Path to save visualization
        
    Returns:
        Tuple of (heatmap, overlay)
    """
    # Generate CAM
    gradcam = GradCAM(model, target_layer)
    cam = gradcam(image, target_class)
    
    # Convert to heatmap
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        colormap
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Resize original image if needed
    if original_image.shape[:2] != cam.shape:
        original_image = cv2.resize(
            original_image,
            (cam.shape[1], cam.shape[0])
        )
    
    # Handle grayscale images
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    elif original_image.shape[2] == 1:
        original_image = cv2.cvtColor(original_image.squeeze(), cv2.COLOR_GRAY2RGB)
        
    # Normalize original image to 0-255
    if original_image.max() <= 1:
        original_image = (original_image * 255).astype(np.uint8)
        
    # Create overlay
    overlay = cv2.addWeighted(
        original_image, 1 - alpha,
        heatmap, alpha,
        0
    )
    
    if save_path:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(original_image, cmap='gray' if len(original_image.shape) == 2 else None)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap)
        plt.title("Grad-CAM Heatmap")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title("Overlay")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    return heatmap, overlay


def batch_visualize_gradcam(
    model: nn.Module,
    images: torch.Tensor,
    original_images: List[np.ndarray],
    target_layer: nn.Module,
    labels: List[int],
    predictions: List[int],
    emotion_labels: dict,
    save_path: str,
    num_samples: int = 16
) -> None:
    """
    Visualize Grad-CAM for a batch of images.
    
    Args:
        model: PyTorch model
        images: Batch of preprocessed images
        original_images: List of original images
        target_layer: Target layer for CAM
        labels: Ground truth labels
        predictions: Model predictions
        emotion_labels: Dictionary mapping indices to emotion names
        save_path: Path to save visualization
        num_samples: Number of samples to visualize
    """
    num_samples = min(num_samples, len(images))
    nrows = int(np.ceil(num_samples / 4))
    
    fig, axes = plt.subplots(nrows, 4, figsize=(16, 4 * nrows))
    axes = axes.flatten()
    
    gradcam = GradCAM(model, target_layer)
    
    for idx in range(num_samples):
        image = images[idx:idx+1]
        cam = gradcam(image, predictions[idx])
        
        # Create overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        orig = original_images[idx]
        if len(orig.shape) == 2:
            orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)
        if orig.max() <= 1:
            orig = (orig * 255).astype(np.uint8)
        orig = cv2.resize(orig, (cam.shape[1], cam.shape[0]))
        
        overlay = cv2.addWeighted(orig, 0.5, heatmap, 0.5, 0)
        
        axes[idx].imshow(overlay)
        
        true_label = emotion_labels[labels[idx]]
        pred_label = emotion_labels[predictions[idx]]
        color = 'green' if labels[idx] == predictions[idx] else 'red'
        
        axes[idx].set_title(
            f"True: {true_label}\nPred: {pred_label}",
            color=color,
            fontsize=10
        )
        axes[idx].axis('off')
        
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("Grad-CAM module loaded successfully")
