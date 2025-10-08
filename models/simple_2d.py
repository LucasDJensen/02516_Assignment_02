import torch
import torch.nn as nn


class Simple2DConvNet(nn.Module):
    """
    A small 2D CNN image classifier.
    Expects input of shape [B, C, H, W] and returns [B, num_classes].
    """
    def __init__(self, in_ch: int = 3, num_classes: int = 100, base_ch: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(base_ch * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)  # [B, C, 1, 1]
        x = x.flatten(1)      # [B, C]
        return self.classifier(x)
