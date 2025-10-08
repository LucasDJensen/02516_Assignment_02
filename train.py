#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T


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
