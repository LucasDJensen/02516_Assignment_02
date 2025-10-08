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


class LateFusionNet(nn.Module):
    def __init__(self, num_classes, w1=0.5):
        super().__init__()
        self.backbone1 = models.resnet18(weights=None)
        self.backbone1.fc = nn.Linear(self.backbone1.fc.in_features, num_classes)
        self.backbone2 = models.resnet18(weights=None)
        self.backbone2.fc = nn.Linear(self.backbone2.fc.in_features, num_classes)
        self.w1 = w1

    def forward(self, x_frames, x_frames_aug):  # each: [B, C, T, H, W]
        B, C, T, H, W = x_frames.shape
        f1 = x_frames.permute(0,2,1,3,4).reshape(B*T, C, H, W)
        l1 = self.backbone1(f1).view(B, T, -1).mean(1)  # [B,K]
        f2 = x_frames_aug.permute(0,2,1,3,4).reshape(B*T, C, H, W)
        l2 = self.backbone2(f2).view(B, T, -1).mean(1)  # [B,K]
        return self.w1*l1 + (1-self.w1)*l2


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root_dir', type=str, default='C:/Users/owner/Documents/DTU/Semester_1/comp_vision/ucf101')
    ap.add_argument('--split', type=str, default='train')
    ap.add_argument('--val_split', type=str, default='val')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=6)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--num_classes', type=int, default=101)
    ap.add_argument('--frames', type=int, default=10)
    ap.add_argument('--size', type=int, default=112)
    ap.add_argument('--w1', type=float, default=0.5)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--prefetch_factor', type=int, default=2)
    ap.add_argument('--persistent_workers', action='store_true')
    ap.add_argument('--log_file', type=str, default=None)
    return ap.parse_args()


def make_loaders(root, split, val_split, size, batch_size, frames, num_workers, prefetch_factor, persistent_workers, device):
    base = [T.Resize((size,size)), T.ToTensor()]
    tfm1 = T.Compose(base)
    tfm2 = T.Compose([T.Resize((size,size)), T.RandomHorizontalFlip(0.5),
                      T.ColorJitter(0.2,0.2,0.2,0.1), T.ToTensor()])

    # paired datasets with same indexing
    train_ds1 = FrameVideoDataset(root_dir=root, split=split, transform=tfm1, stack_frames=True); train_ds1.n_sampled_frames=frames
    train_ds2 = FrameVideoDataset(root_dir=root, split=split, transform=tfm2, stack_frames=True); train_ds2.n_sampled_frames=frames
    val_ds1   = FrameVideoDataset(root_dir=root, split=val_split, transform=tfm1, stack_frames=True); val_ds1.n_sampled_frames=frames
    val_ds2   = FrameVideoDataset(root_dir=root, split=val_split, transform=tfm2, stack_frames=True); val_ds2.n_sampled_frames=frames

    pin = (device.type == "cuda")
    ltrain1 = DataLoader(train_ds1, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                         pin_memory=pin, prefetch_factor=prefetch_factor if num_workers>0 else None,
                         persistent_workers=persistent_workers and num_workers>0)
    ltrain2 = DataLoader(train_ds2, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                         pin_memory=pin, prefetch_factor=prefetch_factor if num_workers>0 else None,
                         persistent_workers=persistent_workers and num_workers>0)
    lval1   = DataLoader(val_ds1,   batch_size=batch_size, shuffle=False, num_workers=num_workers,
                         pin_memory=pin, prefetch_factor=prefetch_factor if num_workers>0 else None,
                         persistent_workers=persistent_workers and num_workers>0)
    lval2   = DataLoader(val_ds2,   batch_size=batch_size, shuffle=False, num_workers=num_workers,
                         pin_memory=pin, prefetch_factor=prefetch_factor if num_workers>0 else None,
                         persistent_workers=persistent_workers and num_workers>0)
    return (ltrain1, ltrain2), (lval1, lval2)


def run():
    args = get_args()
    log = setup_logger(args.log_file)
    log.info("==== Stage: parse args -> OK")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"==== Stage: choose device -> {device}")

    (tr1, tr2), (va1, va2) = make_loaders(args.root_dir, args.split, args.val_split,
                                          args.size, args.batch_size, args.frames,
                                          args.num_workers, args.prefetch_factor,
                                          args.persistent_workers, device)
    log.info("==== Stage: dataloaders ready")

    model = LateFusionNet(args.num_classes, args.w1).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    log.info(f"==== Stage: model/optim ready (w1={args.w1}, lr={args.lr})")

    def _iterate(l1, l2, train=True, epoch=0, tag="Train"):
        total, correct, loss_sum = 0, 0, 0.0
        model.train(train)
        for (x1, y1), (x2, y2) in tqdm(zip(l1, l2), total=len(l1), desc=f"{tag} {epoch}", leave=False):
            if not torch.equal(y1, y2):
                raise RuntimeError("Mismatched labels between paired loaders")
            x1, x2, y = x1.to(device), x2.to(device), y1.to(device)
            logits = model(x1, x2)
            loss = crit(logits, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            bs = y.size(0)
            loss_sum += loss.item()*bs
            total += bs
            correct += (logits.argmax(1)==y).sum().item()
        return loss_sum/total, correct/total

    best = 0.0
    for e in range(1, args.epochs+1):
        log.info(f"---- Epoch {e}/{args.epochs} | Train start")
        tr_loss, tr_acc = _iterate(tr1, tr2, train=True, epoch=e, tag="Train")
        log.info(f"---- Epoch {e}/{args.epochs} | Train done  : loss={tr_loss:.4f} acc={tr_acc:.3f}")

        log.info(f"---- Epoch {e}/{args.epochs} | Eval  start")
        va_loss, va_acc = _iterate(va1, va2, train=False, epoch=e, tag="Eval ")
        log.info(f"---- Epoch {e}/{args.epochs} | Eval  done  : loss={va_loss:.4f} acc={va_acc:.3f}")

        print(f"[Epoch {e}] train {tr_loss:.4f}/{tr_acc:.3f}  val {va_loss:.4f}/{va_acc:.3f}")
        if va_acc > best:
            best = va_acc
            torch.save({'model': model.state_dict(), 'epoch': e, 'best_acc': best}, 'late_fusion_best.pt')
            log.info(f"** Checkpoint saved: late_fusion_best.pt (val_acc={best:.3f})")

    log.info("==== Training finished")


if __name__ == '__main__':
    run()
