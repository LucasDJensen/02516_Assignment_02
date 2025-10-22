#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
import re
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
import matplotlib.pyplot as plt
import numpy as np


def infer_model_name_from_run(run_dir: Path) -> str | None:
    # e.g., runs/3d_1699999999 -> "3d" prefix
    m = re.match(r"^(\w+)_", run_dir.name)
    return m.group(1) if m else None


def build_dataloader(data_dir: Path, split: str, img_size: Tuple[int, int], batch_size: int,
                     num_workers: int, device: torch.device, n_sampled_frames: int) -> DataLoader:
    from dataset.datasets import FrameVideoDataset
    transform = T.Compose([
        T.Resize(tuple(img_size)),
        T.ToTensor(),
    ])
    preload = (device.type == "cuda")
    ds = FrameVideoDataset(
        root_dir=data_dir,
        split=split,
        transform=transform,
        stack_frames=True,
        device=device,
        preload_to_device=preload,
        n_sampled_frames=n_sampled_frames,
    )
    loader_workers = 0 if preload else num_workers
    pin_mem = False if preload else True
    return DataLoader(ds, batch_size=max(1, batch_size), shuffle=False,
                      num_workers=loader_workers, pin_memory=pin_mem)


def build_model(name: str, num_classes: int, n_frames: int, fusion_agg: str, device: torch.device):
    from models import create_model
    model = create_model(name=name, num_classes=num_classes, n_frames=n_frames, fusion_agg=fusion_agg)
    return model.to(device)


@torch.no_grad()
def eval_with_preds(model, loader, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    import torch.nn as nn
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    all_preds: List[int] = []
    all_labels: List[int] = []
    for vids, labels in loader:
        vids = vids if isinstance(vids, torch.Tensor) and vids.device == device else vids.to(device, non_blocking=True)
        labels = labels if isinstance(labels, torch.Tensor) and labels.device == device else labels.to(device, non_blocking=True)
        logits = model(vids)
        loss = criterion(logits, labels)
        loss_sum += loss.item() * labels.size(0)
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(pred.detach().tolist())
        all_labels.extend(labels.detach().tolist())
    loss = loss_sum / max(total, 1)
    acc = correct / max(total, 1)
    return loss, acc, np.array(all_preds, dtype=np.int64), np.array(all_labels, dtype=np.int64)


def confusion_matrix(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(preds, labels):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm: np.ndarray, out_path: Path, class_names: List[str] | None = None):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    if class_names is not None:
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90)
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_yticklabels(class_names)
    ax.set_title('Confusion Matrix')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_metrics(checkpoint_names: List[str], losses: List[float], accs: List[float], out_dir: Path):
    # Line plot
    x = np.arange(len(checkpoint_names))
    plt.figure()
    plt.plot(x, losses, marker='o')
    plt.xticks(x, checkpoint_names, rotation=45, ha='right')
    plt.ylabel('Loss')
    plt.title('Loss per checkpoint')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(out_dir / 'eval_loss_per_checkpoint.png'), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(x, [a * 100 for a in accs], marker='o')
    plt.xticks(x, checkpoint_names, rotation=45, ha='right')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy per checkpoint')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(out_dir / 'eval_acc_per_checkpoint.png'), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints or runs on the test set")
    parser.add_argument('--run_dir', type=str, default=None, help='Path to a single run directory containing .pt checkpoints')
    parser.add_argument('--runs_dir', type=str, default=None, help='Path to the runs directory containing multiple run subfolders')
    parser.add_argument('--data_dir', type=str, default='/dtu/datasets1/02516/ucf101_noleakage')
    parser.add_argument('--test_split', type=str, default='test')
    parser.add_argument('--img_size', type=int, nargs=2, default=[112, 112], help='H W')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model', type=str, default=None, choices=[None, '3d', '2d_per_frame_avg', 'early_fusion_2d', 'late_fusion'],
                        help='If not provided for single-run mode, attempt to infer from run folder name prefix')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--n_sampled_frames', type=int, default=10)
    parser.add_argument('--fusion_agg', type=str, default='mean', choices=['mean', 'max'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out_dir', type=str, default=None, help='Where to save evaluation outputs; default run_dir/eval or runs_dir/eval_all')

    args = parser.parse_args()

    # validate mode
    if (args.run_dir is None) == (args.runs_dir is None):
        raise SystemExit("Provide exactly one of --run_dir or --runs_dir")

    device = torch.device(args.device)

    # Build shared loader
    loader = build_dataloader(
        data_dir=Path(args.data_dir),
        split=args.test_split,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        n_sampled_frames=args.n_sampled_frames,
    )

    if args.run_dir is not None:
        # Single run mode (backward compatible)
        run_dir = Path(args.run_dir)
        assert run_dir.exists() and run_dir.is_dir(), f"Run directory not found: {run_dir}"
        out_dir = Path(args.out_dir) if args.out_dir else (run_dir / 'eval')
        out_dir.mkdir(parents=True, exist_ok=True)

        # Infer model if not specified
        model_name = args.model
        if model_name is None:
            inferred = infer_model_name_from_run(run_dir)
            if inferred is not None:
                model_name = inferred
            else:
                raise SystemExit("--model was not provided and could not be inferred from run directory name.")

        # Find checkpoints
        ckpts = sorted([p for p in run_dir.glob('*.pt') if p.is_file()], key=lambda p: p.name)
        if not ckpts:
            raise SystemExit(f"No .pt checkpoints found in {run_dir}")

        metrics: List[Dict] = []
        for ckpt_path in ckpts:
            model = build_model(name=model_name, num_classes=args.num_classes,
                                n_frames=args.n_sampled_frames, fusion_agg=args.fusion_agg, device=device)
            state = torch.load(ckpt_path, map_location=device)
            state_dict = state.get('model', state)
            model.load_state_dict(state_dict)

            loss, acc, preds, labels = eval_with_preds(model, loader, device)

            rec = {
                'checkpoint': ckpt_path.name,
                'loss': float(loss),
                'acc': float(acc),
                'n_samples': int(len(labels)),
            }
            for k in ('epoch', 'val_acc'):
                if isinstance(state, dict) and k in state:
                    rec[k] = float(state[k]) if isinstance(state[k], (int, float)) else state[k]
            metrics.append(rec)

            cm = confusion_matrix(preds, labels, num_classes=args.num_classes)
            plot_confusion_matrix(cm, out_dir / f'confusion_matrix_{ckpt_path.stem}.png')

        (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        try:
            import pandas as pd
            df = pd.DataFrame(metrics)
            df.to_csv(out_dir / 'metrics.csv', index=False)
        except Exception:
            headers = list(metrics[0].keys())
            with open(out_dir / 'metrics.tsv', 'w', encoding='utf-8') as f:
                f.write('\t'.join(headers) + '\n')
                for m in metrics:
                    f.write('\t'.join(str(m[h]) for h in headers) + '\n')

        names = [m['checkpoint'] for m in metrics]
        losses = [m['loss'] for m in metrics]
        accs = [m['acc'] for m in metrics]
        plot_metrics(names, losses, accs, out_dir)

        best_idx = int(np.argmax(accs))
        best = metrics[best_idx]
        (out_dir / 'best_summary.txt').write_text(
            f"Best checkpoint: {best['checkpoint']}\n"
            f"Accuracy: {best['acc'] * 100:.2f}%\n"
            f"Loss: {best['loss']:.4f}\n"
        )
        print(f"Evaluated {len(metrics)} checkpoints. Best: {best['checkpoint']} (acc {best['acc'] * 100:.2f}%)")
        return

    # Multi-run mode: evaluate best.pt from each subdirectory of runs_dir
    runs_dir = Path(args.runs_dir)
    # If a parent path is given that contains a 'runs' subdirectory, use it
    if runs_dir.name.lower() != 'runs' and (runs_dir / 'runs').is_dir():
        runs_dir = runs_dir / 'runs'
    assert runs_dir.exists() and runs_dir.is_dir(), f"Runs directory not found: {runs_dir}"
    out_all = Path(args.out_dir) if args.out_dir else (runs_dir / 'eval_all')
    out_all.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []
    run_subdirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    for run in sorted(run_subdirs, key=lambda p: p.name):
        ckpt_path = run / 'best.pt'
        if not ckpt_path.exists():
            print(f"[WARN] Skipping {run.name}: best.pt not found")
            continue
        model_name = infer_model_name_from_run(run)
        if not model_name:
            print(f"[WARN] Skipping {run.name}: could not infer model name from folder name")
            continue
        try:
            model = build_model(name=model_name, num_classes=args.num_classes,
                                n_frames=args.n_sampled_frames, fusion_agg=args.fusion_agg, device=device)
            state = torch.load(ckpt_path, map_location=device)
            state_dict = state.get('model', state)
            model.load_state_dict(state_dict)

            loss, acc, preds, labels = eval_with_preds(model, loader, device)

            rec = {
                'run': run.name,
                'model': model_name,
                'checkpoint': 'best.pt',
                'loss': float(loss),
                'acc': float(acc),
                'n_samples': int(len(labels)),
            }
            for k in ('epoch', 'val_acc'):
                if isinstance(state, dict) and k in state:
                    rec[k] = float(state[k]) if isinstance(state[k], (int, float)) else state[k]
            results.append(rec)

            cm = confusion_matrix(preds, labels, num_classes=args.num_classes)
            plot_confusion_matrix(cm, out_all / f'confusion_matrix_{run.name}.png')
        except Exception as e:
            print(f"[ERROR] Failed evaluating {run.name}: {e}")
            continue

    if not results:
        raise SystemExit("No runs evaluated. Ensure runs_dir contains subfolders with best.pt and valid names.")

    # Save cross-run metrics
    (out_all / 'metrics_all.json').write_text(json.dumps(results, indent=2))
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(out_all / 'metrics_all.csv', index=False)
    except Exception:
        headers = list(results[0].keys())
        with open(out_all / 'metrics_all.tsv', 'w', encoding='utf-8') as f:
            f.write('\t'.join(headers) + '\n')
            for m in results:
                f.write('\t'.join(str(m[h]) for h in headers) + '\n')

    # Plot cross-run accuracy and loss (bar charts)
    names = [r['run'] for r in results]
    accs = [r['acc'] for r in results]
    losses = [r['loss'] for r in results]

    x = np.arange(len(names))
    plt.figure(figsize=(max(6, len(names) * 0.8), 4))
    plt.bar(x, [a * 100 for a in accs])
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy per run (best.pt)')
    plt.tight_layout()
    plt.grid(axis='y')
    plt.savefig(str(out_all / 'cross_run_acc.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(max(6, len(names) * 0.8), 4))
    plt.bar(x, losses, color='orange')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('Loss')
    plt.title('Loss per run (best.pt)')
    plt.tight_layout()
    plt.grid(axis='y')
    plt.savefig(str(out_all / 'cross_run_loss.png'), dpi=150)
    plt.close()

    best_idx = int(np.argmax(accs))
    best = results[best_idx]
    (out_all / 'best_overall.txt').write_text(
        f"Best run: {best['run']} ({best['model']})\n"
        f"Accuracy: {best['acc'] * 100:.2f}%\n"
        f"Loss: {best['loss']:.4f}\n"
    )
    print(f"Evaluated {len(results)} runs. Best: {best['run']} ({best['model']}) acc {best['acc'] * 100:.2f}%")


if __name__ == '__main__':
    main()
