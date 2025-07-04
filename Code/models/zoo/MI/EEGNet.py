"""
EEGNet model Implementation according to the published paper @https://arxiv.org/pdf/1611.08024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self, num_classes, channels=64, samples=128, 
                 dropout_rate=0.5, kernel_length=64, F1=8, D=2, F2=16):
        """
        EEGNet implementation according to Lawhern et al. 2018
        
        Args:
            num_classes: Number of output classes
            channels: Number of EEG channels  
            samples: Number of time samples
            dropout_rate: Dropout probability
            kernel_length: Length of temporal convolution in first layer
            F1: Number of temporal filters
            D: Number of spatial filters (per temporal filter)
            F2: Number of pointwise filters
        """
        super(EEGNet, self).__init__()
        
        # Block 1: Temporal Convolution
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length//2), bias=False),
            nn.BatchNorm2d(F1),
        )
        
        # Block 2: Depthwise Convolution (Spatial filtering)
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1*D, (channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )
        
        # Block 3: Separable Convolution (Temporal filtering)
        self.block3 = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(F1*D, F1*D, (1, 16), padding=(0, 8), groups=F1*D, bias=False),
            # Pointwise convolution  
            nn.Conv2d(F1*D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )
        
        # Classification layer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._get_conv_output_size(channels, samples), num_classes)
        )
    
    def _get_conv_output_size(self, channels, samples):
        """Calculate the output size after convolution layers"""
        # Simulate forward pass to get size
        x = torch.zeros(1, 1, channels, samples)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)  # Add channel dimension
            
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        
        return x
    

# Sanity check for model
if __name__ == "__main__":
    model = EEGNet(num_classes=2, channels=3, samples=500, dropout_rate=0.2)
    x = torch.randn(1, 1, 3, 500)  # Batch size of 1, 3 channels, 500 samples
    output = model(x)
    
    assert output.shape == (1, 2), "Output shape mismatch"