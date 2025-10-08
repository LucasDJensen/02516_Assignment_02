import torch
import torch.nn as nn


class Simple3DConvNet(nn.Module):
    def __init__(self, in_ch: int = 3, num_classes: int = 100):
        super().__init__()
        # Input: [B, C, T, H, W]
        self.features = nn.Sequential(
            nn.Conv3d(in_ch, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)
