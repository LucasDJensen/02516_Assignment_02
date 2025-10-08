import argparse, os, sys, logging
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms as T

# Make assignment folder importable when running this file directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets import FrameVideoDataset


def setup_logger(log_file=None, level=logging.INFO):
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt = "%H:%M:%S"
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)
    return logging.getLogger("train")


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root_dir', type=str, default='C:/Users/owner/Documents/DTU/Semester_1/comp_vision/ucf101')
    ap.add_argument('--split', type=str, default='train')
    ap.add_argument('--val_split', type=str, default='val')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--num_classes', type=int, default=101)
    ap.add_argument('--frames', type=int, default=10)
    ap.add_argument('--size', type=int, default=112)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--prefetch_factor', type=int, default=2)
    ap.add_argument('--persistent_workers', action='store_true')
    ap.add_argument('--log_file', type=str, default=None)
    return ap.parse_args()


def make_loaders(root, split, val_split, size, batch_size, frames, device, num_workers, prefetch_factor, persistent_workers):
    tfm = T.Compose([T.Resize((size, size)), T.ToTensor()])

    # We want tensors shaped [C, T, H, W] so we can apply a 2D CNN per frame
    train_ds = FrameVideoDataset(root_dir=root, split=split, transform=tfm, stack_frames=True)
    val_ds   = FrameVideoDataset(root_dir=root, split=val_split, transform=tfm, stack_frames=True)
    train_ds.n_sampled_frames = frames
    val_ds.n_sampled_frames   = frames

    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin,
                              prefetch_factor=prefetch_factor if num_workers>0 else None,
                              persistent_workers=persistent_workers and num_workers>0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin,
                              prefetch_factor=prefetch_factor if num_workers>0 else None,
                              persistent_workers=persistent_workers and num_workers>0)
    return train_loader, val_loader


def build_model(num_classes):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def train_one_epoch(model, loader, criterion, optim, device, epoch):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    pbar = tqdm(loader, desc=f"Train {epoch}", leave=False)
    for vids, labels in pbar:  # vids: [B, C, T, H, W]
        vids, labels = vids.to(device), labels.to(device)
        B, C, T, H, W = vids.shape
        # reshape to per-frame batch: [B*T, C, H, W]
        frames = vids.permute(0, 2, 1, 3, 4).contiguous().view(B*T, C, H, W)
        logits_f = model(frames)  # [B*T, K]
        # average logits per video across frames
        logits = logits_f.view(B, T, -1).mean(dim=1)  # [B, K]

        loss = criterion(logits, labels)
        optim.zero_grad(); loss.backward(); optim.step()

        preds = logits.argmax(1)
        total += B
        correct += (preds == labels).sum().item()
        loss_sum += loss.item() * B
        pbar.set_postfix(loss=f"{loss_sum/total:.4f}", acc=f"{correct/total:.3f}")

    return loss_sum/total, correct/total


def evaluate(model, loader, criterion, device, epoch):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    pbar = tqdm(loader, desc=f"Eval  {epoch}", leave=False)
    with torch.no_grad():
        for vids, labels in pbar:
            vids, labels = vids.to(device), labels.to(device)
            B, C, T, H, W = vids.shape
            frames = vids.permute(0, 2, 1, 3, 4).contiguous().view(B*T, C, H, W)
            logits_f = model(frames)
            logits = logits_f.view(B, T, -1).mean(dim=1)
            loss = criterion(logits, labels)

            preds = logits.argmax(1)
            total += B
            correct += (preds == labels).sum().item()
            loss_sum += loss.item() * B
            pbar.set_postfix(loss=f"{loss_sum/total:.4f}", acc=f"{correct/total:.3f}")

    return loss_sum/total, correct/total


def main():
    args = get_args()
    log = setup_logger(args.log_file)

    log.info("==== Stage: parse args -> OK")
    log.info(f"root_dir={args.root_dir}  split(train/val)={args.split}/{args.val_split}  "
             f"frames={args.frames}  size={args.size}  batch={args.batch_size}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"==== Stage: choose device -> {device}")

    log.info("==== Stage: build datasets/dataloaders")
    train_loader, val_loader = make_loaders(args.root_dir, args.split, args.val_split,
                                            args.size, args.batch_size, args.frames,
                                            device, args.num_workers, args.prefetch_factor,
                                            args.persistent_workers)
    log.info(f"Datasets ready -> train: {len(train_loader.dataset)} videos, val: {len(val_loader.dataset)} videos")

    # Warmup probe to confirm sample shapes (optional)
    try:
        _xb, _yb = next(iter(val_loader))
        log.info(f"Warmup batch OK -> val sample: x={tuple(_xb.shape)}, y={tuple(_yb.shape)}")
    except StopIteration:
        log.warning("Validation loader is empty!")

    log.info("==== Stage: build model/optimizer")
    model = build_model(args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    log.info(f"Model: ResNet18, classes={args.num_classes} | Optim: AdamW lr={args.lr}")

    best = 0.0
    for epoch in range(1, args.epochs+1):
        log.info(f"---- Epoch {epoch}/{args.epochs} | Train start")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optim, device, epoch)
        log.info(f"---- Epoch {epoch}/{args.epochs} | Train done  : loss={tr_loss:.4f} acc={tr_acc:.3f}")

        log.info(f"---- Epoch {epoch}/{args.epochs} | Eval  start")
        va_loss, va_acc = evaluate(model, val_loader, criterion, device, epoch)
        log.info(f"---- Epoch {epoch}/{args.epochs} | Eval  done  : loss={va_loss:.4f} acc={va_acc:.3f}")

        print(f"[Epoch {epoch}] train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")

        if va_acc > best:
            best = va_acc
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'best_acc': best}, 'agg2d_best.pt')
            log.info(f"** Checkpoint saved: agg2d_best.pt (val_acc={best:.3f})")

    log.info("==== Training finished")


if __name__ == '__main__':
    main()