from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T

from pathlib import Path

class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir='/work3/ppar/data/ucf101', split='train', transform=None):
        self.root = Path(root_dir)
        self.frame_paths = sorted(map(Path, glob(f'{root_dir}/frames/{split}/*/*/*.jpg')))
        df = pd.read_csv(self.root / f'metadata/{split}.csv')

        # Ensure one row per video_name
        if df['video_name'].duplicated().any():
            dups = df[df['video_name'].duplicated(keep=False)].sort_values('video_name')
            raise ValueError(f"Duplicated video_name in CSV for split={split}:\n{dups[['video_name','label']].head()}")

        self.label_map = dict(zip(df['video_name'], df['label']))
        self.transform = transform
        self.split = split

    def __len__(self): return len(self.frame_paths)

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        video_name = frame_path.parent.name  # the folder just above the frame file
        try:
            label = int(self.label_map[video_name])
        except KeyError:
            raise KeyError(f"No label for video_name={video_name!r} in metadata/{self.split}.csv")

        img = Image.open(frame_path).convert('RGB')
        img = self.transform(img) if self.transform else T.ToTensor()(img)
        return img, label


class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir='/work3/ppar/data/ucf101', split='train', transform=None, stack_frames=True):
        self.root = Path(root_dir)
        self.video_paths = sorted(map(Path, glob(f'{root_dir}/videos/{split}/*/*.avi')))
        df = pd.read_csv(self.root / f'metadata/{split}.csv')

        if df['video_name'].duplicated().any():
            dups = df[df['video_name'].duplicated(keep=False)].sort_values('video_name')
            raise ValueError(f"Duplicated video_name in CSV for split={split}:\n{dups[['video_name','label']].head()}")

        self.label_map = dict(zip(df['video_name'], df['label']))
        self.transform = transform
        self.stack_frames = stack_frames
        self.n_sampled_frames = 10
        self.split = split

    def __len__(self): return len(self.video_paths)

    def load_frames(self, frames_dir: Path):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            fp = frames_dir / f"frame_{i}.jpg"
            if not fp.exists():
                raise FileNotFoundError(f"Missing frame: {fp}")
            frames.append(Image.open(fp).convert('RGB'))
        return frames

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = video_path.stem  # filename without .avi

        try:
            label = int(self.label_map[video_name])
        except KeyError:
            raise KeyError(f"No label for video_name={video_name!r} in metadata/{self.split}.csv")

        frames_dir = (video_path.with_suffix('')).parent.parent  # up from videos/... to frames/...
        frames_dir = Path(str(video_path.with_suffix(''))).as_posix().replace('/videos/', '/frames/')
        frames_dir = Path(frames_dir)

        video_frames = self.load_frames(frames_dir)

        frames = [self.transform(f) if self.transform else T.ToTensor()(f) for f in video_frames]
        if self.stack_frames:
            # [T, C, H, W] -> [C, T, H, W]
            frames = torch.stack(frames).permute(1, 0, 2, 3)
        return frames, label



if __name__ == '__main__':
    from torch.utils.data import DataLoader

    root_dir = 'C:/Users/owner/Documents/DTU/Semester_1/comp_vision/ucf101'

    transform = T.Compose([T.Resize((64, 64)),T.ToTensor()])
    frameimage_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)
    framevideostack_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = True)
    framevideolist_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames = False)


    frameimage_loader = DataLoader(frameimage_dataset,  batch_size=8, shuffle=False)
    framevideostack_loader = DataLoader(framevideostack_dataset,  batch_size=8, shuffle=False)
    framevideolist_loader = DataLoader(framevideolist_dataset,  batch_size=8, shuffle=False)

    for frames, labels in frameimage_loader:
        print(frames.shape, labels.shape) # [batch, channels, height, width]

    for video_frames, labels in framevideolist_loader:
        print(45*'-')
        for frame in video_frames: # loop through number of frames
            print(frame.shape, labels.shape)# [batch, channels, height, width]

    for video_frames, labels in framevideostack_loader:
        print(video_frames.shape, labels.shape) # [batch, channels, number of frames, height, width]
            
