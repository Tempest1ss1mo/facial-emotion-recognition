"""
Models module for Facial Emotion Recognition
"""

from .baseline_cnn import BaselineCNN
from .transfer_model import TransferModel, get_transfer_model

__all__ = ["BaselineCNN", "TransferModel", "get_transfer_model"]
