"""
Evaluation script for Facial Emotion Recognition.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth
    python evaluate.py --checkpoint checkpoints/best_model.pth --gradcam
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import yaml

from models import BaselineCNN, get_transfer_model
from data import get_dataloaders, EMOTION_LABELS
from utils.metrics import (
    calculate_metrics,
    get_confusion_matrix,
    get_classification_report,
    per_class_accuracy
)
from utils.visualization import (
    plot_confusion_matrix,
    plot_sample_predictions,
    plot_per_class_accuracy
)
from utils.gradcam import GradCAM, get_target_layer, batch_visualize_gradcam


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Facial Emotion Recognition Model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['baseline', 'resnet18', 'vgg16'],
                        help='Model architecture')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--gradcam', action='store_true',
                        help='Generate Grad-CAM visualizations')
    parser.add_argument('--num-samples', type=int, default=16,
                        help='Number of samples for visualization')
    
    return parser.parse_args()


def get_model(model_name: str, checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    num_classes = 7
    
    if model_name == 'baseline':
        model = BaselineCNN(num_classes=num_classes)
    else:
        model = get_transfer_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False  # We'll load weights from checkpoint
        )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def evaluate(model, test_loader, device):
    """
    Evaluate model on test set.
    
    Returns:
        all_labels: Ground truth labels
        all_preds: Predicted labels
        all_probs: Prediction probabilities
        all_images: Sample images for visualization
    """
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    all_images = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Store some images for visualization
            if len(all_images) < 100:
                all_images.extend(images.cpu().numpy())
    
    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        np.array(all_images[:100])
    )


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Create results directory
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = get_model(args.model, args.checkpoint, device)
    
    # Data loader
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False
    )
    
    # Evaluate
    print("\nEvaluating model...")
    labels, preds, probs, images = evaluate(model, test_loader, device)
    
    # Class names
    class_names = list(EMOTION_LABELS.values())
    
    # Calculate metrics
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    metrics = calculate_metrics(labels, preds)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    
    # Per-class accuracy
    class_acc = per_class_accuracy(labels, preds, class_names)
    print(f"\nPer-Class Accuracy:")
    for emotion, acc in class_acc.items():
        print(f"  {emotion:10s}: {acc:.4f} ({acc*100:.2f}%)")
    
    # Classification report
    print(f"\nClassification Report:")
    report = get_classification_report(labels, preds, class_names)
    print(report)
    
    # Save metrics
    metrics_path = Path(args.results_dir) / 'evaluation_metrics.yaml'
    with open(metrics_path, 'w') as f:
        yaml.dump({
            'overall': metrics,
            'per_class': class_acc,
            'report': get_classification_report(labels, preds, class_names, output_dict=True)
        }, f)
    print(f"\nMetrics saved to {metrics_path}")
    
    # Plot confusion matrix
    cm = get_confusion_matrix(labels, preds)
    cm_path = Path(args.results_dir) / 'confusion_matrix.png'
    plot_confusion_matrix(cm, class_names, save_path=str(cm_path))
    
    # Plot per-class accuracy
    acc_path = Path(args.results_dir) / 'per_class_accuracy.png'
    plot_per_class_accuracy(class_acc, save_path=str(acc_path))
    
    # Plot sample predictions
    confidences = [probs[i][preds[i]] for i in range(len(preds))]
    samples_path = Path(args.results_dir) / 'sample_predictions.png'
    plot_sample_predictions(
        images[:args.num_samples],
        labels[:args.num_samples].tolist(),
        preds[:args.num_samples].tolist(),
        class_names,
        confidences[:args.num_samples],
        save_path=str(samples_path),
        num_samples=args.num_samples
    )
    
    # Grad-CAM visualization
    if args.gradcam:
        print("\nGenerating Grad-CAM visualizations...")
        try:
            target_layer = get_target_layer(model, args.model)
            gradcam_path = Path(args.results_dir) / 'gradcam_visualization.png'
            
            # Prepare tensors
            sample_images = torch.tensor(images[:args.num_samples]).to(device)
            original_images = [(img.transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1) 
                              for img in images[:args.num_samples]]
            
            batch_visualize_gradcam(
                model=model,
                images=sample_images,
                original_images=original_images,
                target_layer=target_layer,
                labels=labels[:args.num_samples].tolist(),
                predictions=preds[:args.num_samples].tolist(),
                emotion_labels=EMOTION_LABELS,
                save_path=str(gradcam_path),
                num_samples=args.num_samples
            )
            print(f"Grad-CAM visualization saved to {gradcam_path}")
        except Exception as e:
            print(f"Error generating Grad-CAM: {e}")
    
    print("\n" + "="*50)
    print("Evaluation complete!")
    print(f"Results saved to {args.results_dir}/")
    print("="*50)
    
    return metrics


if __name__ == '__main__':
    main()
