import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def _maybe_to(x, device):
    # no-op if already on the same device
    return x if isinstance(x, torch.Tensor) and x.device == device else x.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for vids, labels in loader:
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
