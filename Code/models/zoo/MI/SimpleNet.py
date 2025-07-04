"""
A simple convolutional neural network (CNN) for Motor Imagery (MI) classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self, num_classes, channels, samples, dropout_rate=0.5):
        super(SimpleNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 3), padding=(0,1)),
            nn.ELU(),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(16 * channels * samples, num_classes)
        )
    def forward(self, x):
        return self.net(x)
    

# Sanity check for model
if __name__ == "__main__":
    model = SimpleNet(num_classes=2, channels=3, samples=500, dropout_rate=0.2)
    x = torch.randn(1, 1, 3, 500)
    output = model(x)
    
    assert output.shape == (1, 2), "Output shape mismatch"