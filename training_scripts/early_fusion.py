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


class EarlyFusionResNet(nn.Module):
    def __init__(self, num_classes, t_frames):
        super().__init__()
        base = models.resnet18(weights=None)
        in_ch = 3 * t_frames
        base.conv1 = nn.Conv2d(in_ch, base.conv1.out_channels,
                               kernel_size=base.conv1.kernel_size,
                               stride=base.conv1.stride,
                               padding=base.conv1.padding, bias=False)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.net = base
        self.t = t_frames

    def forward(self, x):  # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        assert T == self.t, f"Expected {self.t} frames, got {T}"
        x = x.view(B, C*T, H, W)
        return self.net(x)


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


def main():
    args = get_args()
    log = setup_logger(args.log_file)
    log.info("==== Stage: parse args -> OK")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"==== Stage: choose device -> {device}")

    tfm = T.Compose([T.Resize((args.size,args.size)), T.ToTensor()])
    train_ds = FrameVideoDataset(root_dir=args.root_dir, split=args.split, transform=tfm, stack_frames=True)
    val_ds   = FrameVideoDataset(root_dir=args.root_dir, split=args.val_split, transform=tfm, stack_frames=True)
    train_ds.n_sampled_frames = args.frames
    val_ds.n_sampled_frames = args.frames
    log.info("==== Stage: datasets built")

    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin,
                              prefetch_factor=args.prefetch_factor if args.num_workers>0 else None,
                              persistent_workers=args.persistent_workers and args.num_workers>0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin,
                              prefetch_factor=args.prefetch_factor if args.num_workers>0 else None,
                              persistent_workers=args.persistent_workers and args.num_workers>0)
    log.info(f"==== Stage: dataloaders ready (train={len(train_ds)}, val={len(val_ds)})")

    model = EarlyFusionResNet(args.num_classes, args.frames).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    log.info(f"==== Stage: model/optim ready (lr={args.lr})")

    def run_epoch(loader, train=True, epoch=0, tag="Train"):
        model.train(train)
        tot=0; ok=0; loss_sum=0.0
        pbar = tqdm(loader, desc=f"{tag} {epoch}", leave=False)
        for x,y in pbar:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            bs = y.size(0)
            loss_sum += loss.item()*bs
            tot += bs
            ok  += (logits.argmax(1)==y).sum().item()
            pbar.set_postfix(loss=f"{loss_sum/tot:.4f}", acc=f"{ok/tot:.3f}")
        return loss_sum/tot, ok/tot

    best=0.0
    for e in range(1, args.epochs+1):
        log.info(f"---- Epoch {e}/{args.epochs} | Train start")
        tr_loss,tr_acc = run_epoch(train_loader, True, e, "Train")
        log.info(f"---- Epoch {e}/{args.epochs} | Train done  : loss={tr_loss:.4f} acc={tr_acc:.3f}")

        log.info(f"---- Epoch {e}/{args.epochs} | Eval  start")
        va_loss,va_acc = run_epoch(val_loader, False, e, "Eval ")
        log.info(f"---- Epoch {e}/{args.epochs} | Eval  done  : loss={va_loss:.4f} acc={va_acc:.3f}")

        print(f"[Epoch {e}] train {tr_loss:.4f}/{tr_acc:.3f}  val {va_loss:.4f}/{va_acc:.3f}")
        if va_acc>best:
            best=va_acc
            torch.save({'model': model.state_dict(), 'epoch': e, 'best_acc': best}, 'early_fusion_best.pt')
            log.info(f"** Checkpoint saved: early_fusion_best.pt (val_acc={best:.3f})")

    log.info("==== Training finished")


if __name__=='__main__':
    main()