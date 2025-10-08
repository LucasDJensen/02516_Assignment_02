from glob import glob
import os
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T


class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_dir: Path,
                 split='train',
                 transform=None
                 ):
        self.frame_paths = sorted(glob(str(root_dir / 'frames' / split / '*' / '*' / '*.jpg')))
        self.df = pd.read_csv(root_dir / 'metadata' / f'{split}.csv')
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.frame_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        video_name = Path(frame_path).parent.name
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        frame = Image.open(frame_path).convert('RGB')

        if self.transform:
            frame = self.transform(frame)
        else:
            frame = T.ToTensor()(frame)

        return frame, label


class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_dir: Path,
                 split='train',
                 transform=None,
                 stack_frames=True
                 ):

        self.video_paths = sorted(glob(str(root_dir / "videos" / split / "*" / "*.avi")))
        self.df = pd.read_csv(root_dir / "metadata" / f"{split}.csv")
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames

        self.n_sampled_frames = 10

    def __len__(self):
        return len(self.video_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = Path(video_path).name.split('.avi')[0]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        video_frames_dir = self.video_paths[idx].split('.avi')[0].replace('videos', 'frames')
        video_frames = self.load_frames(video_frames_dir)

        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [T.ToTensor()(frame) for frame in video_frames]

        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)

        return frames, label

    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f'frame_{i}.jpg')
            frame = Image.open(frame_file).convert('RGB')
            frames.append(frame)

        return frames
