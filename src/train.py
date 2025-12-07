"""
Training script for Facial Emotion Recognition.

Usage:
    python train.py --model baseline --epochs 50
    python train.py --model resnet18 --mode finetune --epochs 30
"""

import os
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import yaml

from models import BaselineCNN, get_transfer_model
from data import get_dataloaders
from utils.metrics import MetricsTracker, calculate_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Facial Emotion Recognition Model')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['baseline', 'resnet18', 'vgg16'],
                        help='Model architecture')
    parser.add_argument('--mode', type=str, default='finetune',
                        choices=['finetune', 'feature_extraction'],
                        help='Transfer learning mode')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Use data augmentation')
    
    # Output arguments
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory to save results')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(args, device):
    """Create model based on arguments."""
    num_classes = 7
    
    if args.model == 'baseline':
        model = BaselineCNN(
            num_classes=num_classes,
            dropout=args.dropout
        )
    else:
        model = get_transfer_model(
            model_name=args.model,
            num_classes=num_classes,
            pretrained=True,
            mode=args.mode,
            dropout=args.dropout
        )
    
    return model.to(device)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # Override args with config
        for key, value in config.get('training', {}).items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Set seed
    set_seed(args.seed)
    
    # Create directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment
    )
    
    # Model
    print(f"Creating {args.model} model...")
    model = get_model(args, device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    # Metrics tracker
    tracker = MetricsTracker()
    
    # Resume from checkpoint
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print("\nStarting training...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print('='*50)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Update tracker
        current_lr = optimizer.param_groups[0]['lr']
        improved = tracker.update(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)
        
        # Print summary
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if improved:
            best_path = Path(args.checkpoint_dir) / f'best_{args.model}_{timestamp}.pth'
            save_checkpoint(model, optimizer, scheduler, epoch, tracker.get_history(), best_path)
            print(f"New best model! Val Acc: {val_acc:.2f}%")
    
    # Save final model
    final_path = Path(args.checkpoint_dir) / f'final_{args.model}_{timestamp}.pth'
    save_checkpoint(model, optimizer, scheduler, args.epochs, tracker.get_history(), final_path)
    
    # Print best results
    print("\n" + "="*50)
    tracker.print_best_summary()
    
    # Save training history
    history_path = Path(args.results_dir) / f'history_{args.model}_{timestamp}.yaml'
    with open(history_path, 'w') as f:
        yaml.dump(tracker.get_history(), f)
    print(f"\nTraining history saved to {history_path}")
    
    return tracker.get_best()


if __name__ == '__main__':
    main()
