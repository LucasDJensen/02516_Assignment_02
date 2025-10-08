import torch
import torch.nn as nn
from typing import List


class PerFrameAggregator(nn.Module):
    """
    Wraps a 2D frame model and aggregates per-frame predictions into a video prediction.
    Input: x of shape [B, C, T, H, W]
    Aggregation: average logits (or max) over T.
    """
    def __init__(self, frame_model: nn.Module, agg: str = "mean"):
        super().__init__()
        self.frame_model = frame_model
        assert agg in {"mean", "max"}
        self.agg = agg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W] -> [B, T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        logits = self.frame_model(x)  # [B*T, num_classes]
        num_classes = logits.shape[-1]
        logits = logits.view(B, T, num_classes)
        if self.agg == "mean":
            out = logits.mean(dim=1)
        else:
            out = logits.max(dim=1).values
        return out


class EarlyFusion2D(nn.Module):
    """
    Early fusion: concatenate frames along channel dim, run a 2D model once.
    Input: [B, C, T, H, W]. The underlying model must accept in_ch = C*T.
    """
    def __init__(self, model_2d: nn.Module, n_frames: int):
        super().__init__()
        self.model_2d = model_2d
        self.n_frames = n_frames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W] -> [B, C*T, H, W]
        B, C, T, H, W = x.shape
        assert T == self.n_frames, f"Expected {self.n_frames} frames, got {T}"
        x = x.permute(0, 2, 1, 3, 4).reshape(B, C * T, H, W)
        return self.model_2d(x)


class LateFusionEnsemble(nn.Module):
    """
    Late fusion of multiple models by averaging logits.
    Each submodel must accept the same input as provided (e.g., [B,C,T,H,W] for 3D models).
    """
    def __init__(self, submodels: List[nn.Module], agg: str = "mean"):
        super().__init__()
        self.submodels = nn.ModuleList(submodels)
        assert agg in {"mean", "max"}
        self.agg = agg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits_list = [m(x) for m in self.submodels]
        logits = torch.stack(logits_list, dim=0)  # [M, B, num_classes]
        if self.agg == "mean":
            return logits.mean(dim=0)
        else:
            return logits.max(dim=0).values
