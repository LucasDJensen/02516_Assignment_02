#!/usr/bin/env python3
import argparse
import os
import time
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T


class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root_dir: Path,
            split: str = "train",
            transform=None,
            *,
            device: str | torch.device = "cpu",
            preload_to_device: bool = False,
    ):
        self.frame_paths = sorted(glob(str(root_dir / "frames" / split / "*" / "*" / "*.jpg")))
        self.df = pd.read_csv(root_dir / "metadata" / f"{split}.csv")
        self.split = split
        self.transform = transform
        self.device = torch.device(device)
        self.preload_to_device = preload_to_device

        # Optional: preload to device immediately
        self._frames_gpu: list[torch.Tensor] | None = None
        self._labels_gpu: list[torch.Tensor] | None = None
        if self.preload_to_device:
            self._preload_all_images_to_device()

    def __len__(self):
        return len(self.frame_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def _load_one(self, idx: int):
        frame_path = self.frame_paths[idx]
        video_name = Path(frame_path).parent.name
        video_meta = self._get_meta("video_name", video_name)
        label = int(video_meta["label"].item())

        frame = Image.open(frame_path).convert("RGB")
        if self.transform:
            frame = self.transform(frame)
        else:
            frame = T.ToTensor()(frame)

        return frame, label

    def _preload_all_images_to_device(self):
        frames_gpu, labels_gpu = [], []
        for i in range(len(self.frame_paths)):
            frame, label = self._load_one(i)
            frames_gpu.append(frame.to(self.device, non_blocking=False))
            labels_gpu.append(torch.tensor(label, device=self.device, dtype=torch.long))
        self._frames_gpu = frames_gpu
        self._labels_gpu = labels_gpu

    def __getitem__(self, idx):
        if self._frames_gpu is not None:
            # Already on device
            return self._frames_gpu[idx], self._labels_gpu[idx]
        # Fallback: load on demand (CPU), caller can .to(device)
        frame, label = self._load_one(idx)
        return frame, label


class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root_dir: Path,
            split: str = "train",
            transform=None,
            stack_frames: bool = True,
            *,
            device: str | torch.device = "cpu",
            preload_to_device: bool = False,
            n_sampled_frames: int = 10,
    ):
        self.video_paths = sorted(glob(str(root_dir / "videos" / split / "*" / "*.avi")))
        self.df = pd.read_csv(root_dir / "metadata" / f"{split}.csv")
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames

        self.n_sampled_frames = n_sampled_frames
        self.device = torch.device(device)
        self.preload_to_device = preload_to_device

        # Optional: preload to device immediately
        self._videos_gpu: list[torch.Tensor] | None = None
        self._labels_gpu: list[torch.Tensor] | None = None
        if self.preload_to_device:
            self._preload_all_videos_to_device()

    def __len__(self):
        return len(self.video_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def load_frames(self, frames_dir: str):
        frames = []
        # assumes frames exist as frame_1.jpg ... frame_n.jpg
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)
        return frames

    def _load_one(self, idx: int):
        video_path = self.video_paths[idx]
        video_name = Path(video_path).name.split(".avi")[0]
        video_meta = self._get_meta("video_name", video_name)
        label = int(video_meta["label"].item())

        video_frames_dir = video_path.split(".avi")[0].replace("videos", "frames")
        video_frames = self.load_frames(video_frames_dir)

        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [T.ToTensor()(frame) for frame in video_frames]

        if self.stack_frames:
            # frames: list[T, C, H, W] -> [C, T, H, W]
            frames = torch.stack(frames).permute(1, 0, 2, 3)

        return frames, label

    def _preload_all_videos_to_device(self):
        vids_gpu, labels_gpu = [], []
        for i in range(len(self.video_paths)):
            x, y = self._load_one(i)  # CPU tensors
            vids_gpu.append(x.to(self.device, non_blocking=False))
            labels_gpu.append(torch.tensor(y, device=self.device, dtype=torch.long))
        self._videos_gpu = vids_gpu
        self._labels_gpu = labels_gpu

    def __getitem__(self, idx):
        if self._videos_gpu is not None:
            return self._videos_gpu[idx], self._labels_gpu[idx]
        x, y = self._load_one(idx)
        return x, y


def _maybe_to(x, device):
    # no-op if already on the same device
    return x if isinstance(x, torch.Tensor) and x.device == device else x.to(device, non_blocking=True)


# -----------------------------
# Model: a tiny 3D CNN baseline
# -----------------------------
class Simple3DConvNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=100):
        super().__init__()
        # Input: [B, C, T, H, W]
        self.features = nn.Sequential(
            nn.Conv3d(in_ch, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),  # -> halve H,W

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),  # -> halve T,H,W

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # -> [B, 128, 1, 1, 1]
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)  # [B, 128, 1, 1, 1]
        x = x.flatten(1)  # [B, 128]
        return self.classifier(x)  # [B, num_classes]


# -----------------------------
# Train / Eval helpers
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for vids, labels in loader:
        # vids: [B, C, T, H, W]
        vids = _maybe_to(vids, device)
        labels = _maybe_to(labels, device)
        logits = model(vids)
        loss = criterion(logits, labels)
        loss_sum += loss.item() * labels.size(0)
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def train_one_epoch(model, loader, optimizer, device, scaler=None, grad_clip=None):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for vids, labels in loader:
        vids = _maybe_to(vids, device)
        labels = _maybe_to(labels, device)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(vids)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(vids)
            loss = criterion(logits, labels)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        loss_sum += loss.item() * labels.size(0)
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return loss_sum / max(total, 1), correct / max(total, 1)


def plot_curves(history, out_path="training_curves.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss vs Epochs")
    plt.grid(True)
    plt.savefig(out_path.replace(".png", "_loss.png"), bbox_inches="tight", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs, [100 * x for x in history["train_acc"]], label="Train Acc (%)")
    plt.plot(epochs, [100 * x for x in history["val_acc"]], label="Val Acc (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Accuracy vs Epochs")
    plt.grid(True)
    plt.savefig(out_path.replace(".png", "_acc.png"), bbox_inches="tight", dpi=150)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Modular video classifier")
    parser.add_argument("--data_dir", type=str, default="/dtu/datasets1/02516/ucf101_noleakage")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--img_size", type=int, nargs=2, default=[112, 112], help="H W")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_amp", action="store_true", help="Mixed precision")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_dir", type=str, default=None)
    # New options
    parser.add_argument("--model", type=str, default="3d", choices=["3d", "2d_per_frame_avg", "early_fusion_2d", "late_fusion"],
                        help="Model architecture / fusion strategy")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--n_sampled_frames", type=int, default=10)
    parser.add_argument("--fusion_agg", type=str, default="mean", choices=["mean", "max"],
                        help="Aggregation method for per-frame or late fusion")

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = Path("runs") / f"{args.model}_{int(time.time())}"
        save_dir.mkdir(parents=True, exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Import modular components lazily to avoid name clashes with legacy code above
    from dataset.datasets import FrameVideoDataset as _FrameVideoDataset
    from models import create_model
    from utils.train_utils import evaluate as _evaluate, train_one_epoch as _train_one_epoch, plot_curves as _plot_curves

    # Transforms
    transform = T.Compose([
        T.Resize(tuple(args.img_size)),
        T.ToTensor(),
    ])

    # Preload datasets to GPU if using CUDA
    preload = (device.type == "cuda")
    train_set = _FrameVideoDataset(
        root_dir=Path(args.data_dir),
        split=args.train_split,
        transform=transform,
        stack_frames=True,
        device=device,
        preload_to_device=preload,
        n_sampled_frames=args.n_sampled_frames,
    )
    val_set = _FrameVideoDataset(
        root_dir=Path(args.data_dir),
        split=args.val_split,
        transform=transform,
        stack_frames=True,
        device=device,
        preload_to_device=preload,
        n_sampled_frames=args.n_sampled_frames,
    )

    # If data are on GPU, don't use workers/pin_memory
    loader_workers = 0 if preload else args.num_workers
    pin_mem = False if preload else True

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=loader_workers, pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_set, batch_size=max(1, args.batch_size), shuffle=False,
        num_workers=loader_workers, pin_memory=pin_mem
    )

    # Create model via factory
    model = create_model(
        name=args.model,
        num_classes=args.num_classes,
        n_frames=args.n_sampled_frames,
        fusion_agg=args.fusion_agg,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=(args.use_amp and device.type == "cuda"))

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_path = save_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = _train_one_epoch(
            model, train_loader, optimizer, device,
            scaler=scaler if scaler.is_enabled() else None, grad_clip=args.grad_clip
        )
        val_loss, val_acc = _evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        took = time.time() - t0
        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"Train: loss {train_loss:.4f}, acc {train_acc * 100:.2f}% | "
              f"Val: loss {val_loss:.4f}, acc {val_acc * 100:.2f}% | "
              f"{took:.1f}s")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "val_acc": val_acc}, best_path)

        # Save curves progressively
        _plot_curves(history, out_path=str(save_dir / "training_curves.png"))

    # Final save
    last_path = save_dir / "last.pt"
    torch.save({"model": model.state_dict(),
                "epoch": args.epochs,
                "val_acc": history["val_acc"][-1]}, last_path)

    print(f"Best checkpoint: {best_path} (val acc {best_val_acc * 100:.2f}%)")
    print(f"Curves saved to: {save_dir / 'training_curves_acc.png'} and {save_dir / 'training_curves_loss.png'}")


if __name__ == "__main__":
    main()
