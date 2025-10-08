from pathlib import Path
from glob import glob
import os
from typing import List

import pandas as pd
import torch
from PIL import Image
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
        self.frame_paths = sorted(glob(str(Path(root_dir) / "frames" / split / "*" / "*" / "*.jpg")))
        self.df = pd.read_csv(Path(root_dir) / "metadata" / f"{split}.csv")
        self.split = split
        self.transform = transform
        self.device = torch.device(device)
        self.preload_to_device = preload_to_device

        self._frames_gpu: List[torch.Tensor] | None = None
        self._labels_gpu: List[torch.Tensor] | None = None
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
            return self._frames_gpu[idx], self._labels_gpu[idx]
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
        self.video_paths = sorted(glob(str(Path(root_dir) / "videos" / split / "*" / "*.avi")))
        self.df = pd.read_csv(Path(root_dir) / "metadata" / f"{split}.csv")
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames

        self.n_sampled_frames = n_sampled_frames
        self.device = torch.device(device)
        self.preload_to_device = preload_to_device

        self._videos_gpu: List[torch.Tensor] | None = None
        self._labels_gpu: List[torch.Tensor] | None = None
        if self.preload_to_device:
            self._preload_all_videos_to_device()

    def __len__(self):
        return len(self.video_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def load_frames(self, frames_dir: str):
        frames = []
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
            # [T, C, H, W] -> [C, T, H, W]
            frames = torch.stack(frames).permute(1, 0, 2, 3)

        return frames, label

    def _preload_all_videos_to_device(self):
        vids_gpu, labels_gpu = [], []
        for i in range(len(self.video_paths)):
            x, y = self._load_one(i)
            vids_gpu.append(x.to(self.device, non_blocking=False))
            labels_gpu.append(torch.tensor(y, device=self.device, dtype=torch.long))
        self._videos_gpu = vids_gpu
        self._labels_gpu = labels_gpu

    def __getitem__(self, idx):
        if self._videos_gpu is not None:
            return self._videos_gpu[idx], self._labels_gpu[idx]
        x, y = self._load_one(idx)
        return x, y
