
import argparse, sys, os, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms as T
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import FrameVideoDataset


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
    return ap.parse_args()


def make_loaders(root, split, val_split, size, batch_size, frames):
    tfm = T.Compose([T.Resize((size, size)), T.ToTensor()])
    # We want tensors shaped [C, T, H, W] so we can apply a 2D CNN per frame
    train_ds = FrameVideoDataset(root_dir=root, split=split, transform=tfm, stack_frames=True)
    train_ds.n_sampled_frames = frames
    val_ds = FrameVideoDataset(root_dir=root, split=val_split, transform=tfm, stack_frames=True)
    val_ds.n_sampled_frames = frames


    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader

def build_model(num_classes):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def train_one_epoch(model, loader, criterion, optim, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for vids, labels in loader: # vids: [B, C, T, H, W]
        vids, labels = vids.to(device), labels.to(device)
        B, C, T, H, W = vids.shape
        # reshape to per-frame batch: [B*T, C, H, W]
        frames = vids.permute(0, 2, 1, 3, 4).contiguous().view(B*T, C, H, W)
        logits_f = model(frames) # [B*T, K]
        # average logits per video across frames
        logits = logits_f.view(B, T, -1).mean(dim=1) # [B, K]

        loss = criterion(logits, labels)
        optim.zero_grad(); loss.backward(); optim.step()


        loss_sum += loss.item() * B
        preds = logits.argmax(1)
        total += B
        correct += (preds == labels).sum().item()
    return loss_sum/total, correct/total

def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for vids, labels in loader:
            vids, labels = vids.to(device), labels.to(device)
            B, C, T, H, W = vids.shape
            frames = vids.permute(0, 2, 1, 3, 4).contiguous().view(B*T, C, H, W)
            logits_f = model(frames)
            logits = logits_f.view(B, T, -1).mean(dim=1)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * B
            preds = logits.argmax(1)
            total += B
            correct += (preds == labels).sum().item()
    return loss_sum/total, correct/total

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = make_loaders(args.root_dir, args.split, args.val_split, args.size, args.batch_size, args.frames)
    model = build_model(args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)


    best = 0.0
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optim, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        print(f"[Epoch {epoch}] train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")
    if va_acc > best:
        best = va_acc
        torch.save({'model': model.state_dict(), 'epoch': epoch}, 'agg2d_best.pt')


if __name__ == '__main__':
    main()