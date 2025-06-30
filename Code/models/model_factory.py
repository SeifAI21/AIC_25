"""
Model factory for creating models by order
"""

import torch
from torch import nn

from models.zoo.MI.EEGNet import EEGNet
from models.zoo.MI.SimpleNet import SimpleNet
from models.zoo.SSVEP.VotingClassifier import SSVEPVotingClassifier

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


def create_ssvep_model(config, num_classes):
    """
    Factory function to create SSVEP model based on configuration
    
    Args:
        config: Configuration object
        num_classes: Number of SSVEP classes
        
    Returns:
        SSVEP model instance
    """
    model_type = config.get('model.type', 'classical')
    
    if model_type == 'classical':
        voting_type = config.get('model.voting_type', 'soft')
        random_state = config.get('seed', 42)
        
        return SSVEPVotingClassifier(
            num_classes=num_classes,
            voting=voting_type,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown SSVEP model type: {model_type}")


# Test the model factory
if __name__ == "__main__":
    from configs.mi_config import MIConfig
    from configs.ssvep_config import SSVEPConfig

    # Test MI model creation
    mi_config = MIConfig()
    mi_model = create_mi_model(mi_config, channels=3, samples=500)
    print(f"Created MI model: {mi_model.__class__.__name__}")
    
    # Test SSVEP model creation
    ssvep_config = SSVEPConfig()
    ssvep_model = create_ssvep_model(ssvep_config, num_classes=12)
    print(f"Created SSVEP model: {ssvep_model.__class__.__name__}")
    
    # Test forward pass with dummy data for MI model
    dummy_input = torch.randn(1, 1, 3, 500)  # Batch size 1, 1 input channel, 3 EEG channels, 500 samples
    output = mi_model(dummy_input)
    print(f"MI model output shape: {output.shape}")
    
    # Test SSVEP model with dummy data
    import numpy as np
    dummy_features = np.random.randn(10, 100)  # 10 samples, 100 features
    dummy_labels = np.random.randint(0, 12, 10)  # 10 labels
    
    ssvep_model.fit(dummy_features, dummy_labels)
    predictions = ssvep_model.predict(dummy_features)
    print(f"SSVEP model predictions shape: {predictions.shape}")