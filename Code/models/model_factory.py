"""
Model factory for creating models by order
"""

import torch
from torch import nn

from models.zoo.MI.EEGNet import EEGNet
from models.zoo.MI.SimpleNet import SimpleNet
# from models.zoo.SSVEP import ...

def create_mi_model(config, channels, samples, dropout_rate=0.5):
    """
    Factory function to create MI model based on configuration
    
    Args:
        config: Configuration object
        channels: Number of EEG channels
        samples: Number of time samples
        
    Returns:
        PyTorch model instance
    """
    architecture = config.get('model.architecture', 'SimpleNet')
    num_classes = config.get('model.num_classes', 2)
    
    if architecture == 'EEGNet':
        return EEGNet(num_classes, channels, samples, dropout_rate)
    elif architecture == 'SimpleNet':
        return SimpleNet(num_classes, channels, samples, dropout_rate)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


# Test the model factory
if __name__ == "__main__":
    from configs.mi_config import MIConfig

    # Load configuration
    mi_config = MIConfig()
    
    # Create a model instance
    model = create_mi_model(mi_config, channels=3, samples=500)
    
    print(f"Created model: {model.__class__.__name__}")
    print(f"Model architecture: {model}")
    
    # Test forward pass with dummy data - Add unsqueeze for channel dimension
    dummy_input = torch.randn(1, 1, 3, 500)  # Batch size 1, 1 input channel, 3 EEG channels, 500 samples
    output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")