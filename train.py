from datasets import *
from torch.utils.data import DataLoader
import argparse

if __name__ == "__main__":
    import torch

    print("Is CUDA available?", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    parser = argparse.ArgumentParser(description='Training script for video classification')
    parser.add_argument('--data_dir', type=str, default='/dtu/datasets1/02516/ucf101_noleakage', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--split', type=str, default='val', help='Dataset split to use (train/val/test)')
    args = parser.parse_args()

    print("Parsed args:", args)
    root_dir = Path(args.data_dir)

    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    frameimage_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)
    framevideostack_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames=True)
    framevideolist_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames=False)

    frameimage_loader = DataLoader(frameimage_dataset, batch_size=args.batch_size, shuffle=False)
    framevideostack_loader = DataLoader(framevideostack_dataset, batch_size=args.batch_size, shuffle=False)
    framevideolist_loader = DataLoader(framevideolist_dataset, batch_size=args.batch_size, shuffle=False)

    for frames, labels in frameimage_loader:
        print(frames.shape, labels.shape)  # [batch, channels, height, width]

    for video_frames, labels in framevideolist_loader:
        print(45 * '-')
        for frame in video_frames:  # loop through number of frames
            print(frame.shape, labels.shape)  # [batch, channels, height, width]

    for video_frames, labels in framevideostack_loader:
        print(video_frames.shape, labels.shape)  # [batch, channels, number of frames, height, width]
