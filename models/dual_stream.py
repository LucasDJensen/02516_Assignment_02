from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_sobel_kernel(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    sobel_x = torch.tensor(
        [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
        device=device,
        dtype=dtype,
    )
    sobel_y = torch.tensor(
        [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
        device=device,
        dtype=dtype,
    )
    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)
    return sobel_x, sobel_y


class _SpatialStream(nn.Module):
    """
    AlexNet-style spatial stream as in Simonyan & Zisserman (2014).
    Processes a single RGB frame.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)


class _TemporalStream(nn.Module):
    """
    Temporal stream mirroring the architecture in the paper.
    Input is a stack of optical-flow fields (2*(T-1) channels).
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)


@dataclass
class TwoStreamOutputs:
    spatial_logits: torch.Tensor
    temporal_logits: torch.Tensor
    fused_logits: torch.Tensor


class TwoStreamConvNet(nn.Module):
    """
    Dual-stream architecture with dedicated convolutional stacks for spatial and temporal pathways.
    """

    def __init__(self, *, num_classes: int, n_frames: int, fusion: str = "mean"):
        super().__init__()
        if n_frames < 2:
            raise ValueError("TwoStreamConvNet requires at least two frames.")
        if fusion not in {"mean", "sum", "max"}:
            raise ValueError("Fusion must be one of {'mean', 'sum', 'max'}.")

        self.n_frames = n_frames
        self.fusion = fusion

        flow_channels = 2 * (n_frames - 1)
        self.spatial_stream = _SpatialStream(num_classes=num_classes)
        self.temporal_stream = _TemporalStream(
            in_channels=flow_channels, num_classes=num_classes
        )

        # Helper buffers to compute simple optical-flow proxies (Sobel gradients of frame differences).
        rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32).view(1, 3, 1, 1, 1)
        self.register_buffer("rgb_weights", rgb_weights, persistent=False)

        sobel_x, sobel_y = _make_sobel_kernel(torch.device("cpu"), torch.float32)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def forward(self, x: torch.Tensor, *, return_outputs: bool = False):
        if x.dim() != 5:
            raise ValueError("Expected input of shape [B, C, T, H, W].")

        b, c, t, h, w = x.shape
        if t != self.n_frames:
            raise ValueError(f"Expected {self.n_frames} frames, received {t}.")
        if c != 3:
            raise ValueError("Input must have 3 channels (RGB).")

        spatial_frame = x[:, :, t // 2, :, :]
        spatial_logits = self.spatial_stream(spatial_frame)

        flow_stack = self._compute_optical_flow_stack(x)
        temporal_logits = self.temporal_stream(flow_stack)

        if self.fusion == "mean":
            fused = 0.5 * (spatial_logits + temporal_logits)
        elif self.fusion == "sum":
            fused = spatial_logits + temporal_logits
        else:  # 'max'
            fused = torch.max(spatial_logits, temporal_logits)

        if return_outputs:
            return TwoStreamOutputs(
                spatial_logits=spatial_logits,
                temporal_logits=temporal_logits,
                fused_logits=fused,
            )
        return fused

    def _compute_optical_flow_stack(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        b, _, t, h, w = x.shape
        weights = self.rgb_weights.to(dtype=x.dtype, device=x.device)
        gray = (x * weights).sum(dim=1)  # [B, T, H, W]

        frame_diffs = gray[:, 1:, :, :] - gray[:, :-1, :, :]  # [B, T-1, H, W]
        diff = frame_diffs.reshape(-1, 1, h, w)  # [B*(T-1), 1, H, W]

        sobel_x = self.sobel_x.to(dtype=x.dtype, device=x.device)
        sobel_y = self.sobel_y.to(dtype=x.dtype, device=x.device)

        grad_x = F.conv2d(diff, sobel_x, padding=1)
        grad_y = F.conv2d(diff, sobel_y, padding=1)

        grad_x = grad_x.view(b, t - 1, h, w)
        grad_y = grad_y.view(b, t - 1, h, w)

        flow = torch.cat([grad_x, grad_y], dim=1)  # [B, 2*(T-1), H, W]
        return flow
