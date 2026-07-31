"""Model definitions for the PyTorch keyword-spotting custom learning block.

Audio processing blocks (MFE / MFCC) hand the learning block a flat feature
vector per one-second window. The training script reshapes that vector back into
a ``(1, frames, columns)`` single-channel "spectrogram image", which this
compact 2D CNN consumes.

The network is intentionally small so it trains quickly and fits on
resource-constrained edge targets. Adaptive average pooling keeps the classifier
independent of the exact number of frames/columns, so the same architecture
works for MFE (40 filters) or MFCC (13 coefficients) front ends.
"""

from __future__ import annotations

import torch
from torch import nn


class KwsCNN(nn.Module):
    """A compact 2D CNN for keyword spotting on MFE/MFCC spectrograms.

    Args:
        in_channels: Number of input channels (1 for an MFE/MFCC spectrogram).
        num_classes: Number of output classes (keywords + noise/unknown).
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        # Adaptive pooling makes the classifier independent of the spectrogram
        # dimensions (frame count / coefficient count).
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ClassifierExport(nn.Module):
    """Wraps a trained classifier with a softmax for export.

    Edge Impulse's classifier evaluator expects the model to output per-class
    probabilities (it normalises predictions by their sum, which blows up on
    raw logits). We keep training in logit space (CrossEntropyLoss) and only
    add the softmax at export time.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(x), dim=1)


def build_model(in_channels: int, num_classes: int) -> nn.Module:
    """Factory used by the training script."""
    return KwsCNN(in_channels=in_channels, num_classes=num_classes)
