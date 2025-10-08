# 02516_Assignment_02
02516 Introduction to Deep Learning in Computer Vision. Assignment 2

# venv setup on HPC
cd ~/
module load python3/3.12.7
python3 -m venv 02516_venv
source 02516_venv/bin/activate
python3 -m pip install torch torchvision pandas matplotlib



python train_video_classifier.py --data_dir C:\Users\lucas\PycharmProjects\02516_Assignment_02\data\ufc101 --train_split train --val_split val --img_size 112 112 --batch_size 8 --epochs 20 --lr 1e-3 --weight_decay 1e-4 --num_workers 4 --use_amp


## Modular Training Pipeline

This project has been refactored to make it easy to add new models and fusion strategies.

### New Project Structure
- data/
  - __init__.py
  - datasets.py (FrameImageDataset, FrameVideoDataset)
- models/
  - __init__.py (create_model factory)
  - simple_2d.py (Simple2DConvNet)
  - simple_3d.py (Simple3DConvNet)
  - fusion.py (PerFrameAggregator, EarlyFusion2D, LateFusionEnsemble)
- utils/
  - __init__.py
  - train_utils.py (_maybe_to, train_one_epoch, evaluate, plot_curves)
- train.py (CLI, data loading, training loop)

### Supported Models / Strategies
- 3d: 3D CNN on stacked frames [B, C, T, H, W]
- 2d_per_frame_avg: Per-frame 2D CNN with logits averaged over time (aggregation)
- early_fusion_2d: Concatenate frames along channels and apply a 2D CNN once
- late_fusion: Ensemble multiple sub-models (default: two 3D CNNs) with mean/max over logits

### Key CLI Arguments
- --model {3d,2d_per_frame_avg,early_fusion_2d,late_fusion}
- --num_classes <int>
- --n_sampled_frames <int> (number of frames T per video)
- --fusion_agg {mean,max} (aggregation for per-frame and late fusion)
- Common training args: --batch_size, --epochs, --lr, --weight_decay, --use_amp, --grad_clip

### Examples
- 3D CNN (default behavior):
  python train.py --model 3d --num_classes 10

- Per-frame 2D with average aggregation:
  python train.py --model 2d_per_frame_avg --num_classes 10 --n_sampled_frames 10

- Early fusion with 2D CNN (C*T channels):
  python train.py --model early_fusion_2d --num_classes 10 --n_sampled_frames 10

- Late fusion of two 3D CNNs (mean logits):
  python train.py --model late_fusion --num_classes 10 --fusion_agg mean

### Extending
- Add new models under models/ (e.g., models/my_model.py)
- Register them in models/__init__.py inside create_model()
- Reuse FrameVideoDataset for [C, T, H, W] tensors and the train_utils for training
