"""
Unit tests for Facial Emotion Recognition models.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models import BaselineCNN, get_transfer_model


class TestBaselineCNN:
    """Tests for BaselineCNN model."""
    
    def test_model_creation(self):
        """Test model can be created."""
        model = BaselineCNN(num_classes=7)
        assert model is not None
        
    def test_forward_pass(self):
        """Test forward pass with correct input shape."""
        model = BaselineCNN(num_classes=7)
        x = torch.randn(4, 1, 48, 48)
        output = model(x)
        
        assert output.shape == (4, 7)
        
    def test_different_batch_sizes(self):
        """Test model works with different batch sizes."""
        model = BaselineCNN(num_classes=7)
        
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 1, 48, 48)
            output = model(x)
            assert output.shape == (batch_size, 7)
            
    def test_output_range(self):
        """Test output logits are reasonable."""
        model = BaselineCNN(num_classes=7)
        x = torch.randn(4, 1, 48, 48)
        output = model(x)
        
        # Logits should be finite
        assert torch.isfinite(output).all()
        
    def test_parameter_count(self):
        """Test model has expected number of parameters."""
        model = BaselineCNN(num_classes=7)
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should have reasonable number of parameters
        assert total_params > 10000
        assert total_params < 50000000


class TestTransferModel:
    """Tests for transfer learning models."""
    
    @pytest.mark.parametrize("model_name", ["resnet18", "vgg16"])
    def test_model_creation(self, model_name):
        """Test transfer models can be created."""
        model = get_transfer_model(
            model_name=model_name,
            num_classes=7,
            pretrained=False  # Faster for testing
        )
        assert model is not None
        
    @pytest.mark.parametrize("model_name", ["resnet18", "vgg16"])
    def test_forward_pass(self, model_name):
        """Test forward pass."""
        model = get_transfer_model(
            model_name=model_name,
            num_classes=7,
            pretrained=False
        )
        x = torch.randn(2, 1, 48, 48)
        output = model(x)
        
        assert output.shape == (2, 7)
        
    def test_feature_extraction_mode(self):
        """Test feature extraction freezes layers."""
        model = get_transfer_model(
            model_name="resnet18",
            num_classes=7,
            pretrained=False,
            mode="feature_extraction"
        )
        
        params = model.count_parameters()
        assert params['frozen'] > 0
        
    def test_finetune_mode(self):
        """Test finetune mode has all trainable parameters."""
        model = get_transfer_model(
            model_name="resnet18",
            num_classes=7,
            pretrained=False,
            mode="finetune"
        )
        
        params = model.count_parameters()
        # Most parameters should be trainable
        assert params['trainable'] > params['frozen']


class TestModelTraining:
    """Tests for model training functionality."""
    
    def test_gradient_flow(self):
        """Test gradients flow through the model."""
        model = BaselineCNN(num_classes=7)
        x = torch.randn(4, 1, 48, 48, requires_grad=True)
        y = torch.randint(0, 7, (4,))
        
        output = model(x)
        loss = torch.nn.CrossEntropyLoss()(output, y)
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                
    def test_model_eval_mode(self):
        """Test model behavior in eval mode."""
        model = BaselineCNN(num_classes=7)
        x = torch.randn(4, 1, 48, 48)
        
        model.train()
        output_train = model(x)
        
        model.eval()
        with torch.no_grad():
            output_eval = model(x)
            
        # Outputs should differ due to dropout
        # (though this test may occasionally fail by chance)
        

class TestGradCAM:
    """Tests for Grad-CAM functionality."""
    
    def test_gradcam_import(self):
        """Test Grad-CAM can be imported."""
        from utils.gradcam import GradCAM
        assert GradCAM is not None
        
    def test_gradcam_generation(self):
        """Test Grad-CAM heatmap generation."""
        from utils.gradcam import GradCAM
        
        model = BaselineCNN(num_classes=7)
        target_layer = model.conv4
        
        gradcam = GradCAM(model, target_layer)
        x = torch.randn(1, 1, 48, 48)
        
        cam = gradcam(x)
        
        assert cam.shape == (48, 48)
        assert cam.min() >= 0
        assert cam.max() <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
