"""
VisionGuard - DataLoader & Augmentation Pipeline
Day 2 Milestone: Image resizing, normalization, augmentation transforms.

Deliverables:
- Torchvision ImageFolder data loading
- Augmentation pipeline: RandomHorizontalFlip, RandomVerticalFlip, RandomRotation,
  ColorJitter, RandomResizedCrop, Normalization
- Validation and Test transforms (Resize, CenterCrop, Normalize)
- Configurable DataLoaders with batching and shuffling
- Single-image inference transform
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

# ImageNet statistics for transfer learning and CNN normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size: int = 224) -> Dict[str, transforms.Compose]:
    """Return data augmentation transforms for training and standard transforms for val/test."""
    train_transform = transforms.Compose([
        transforms.Resize((int(image_size * 1.15), int(image_size * 1.15))),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return {
        "train": train_transform,
        "val": eval_transform,
        "test": eval_transform,
        "inference": eval_transform,
    }


def get_single_image_transform(image_size: int = 224) -> transforms.Compose:
    """Transform for single-image inference in predict.py."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_and_preprocess_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    """Load an arbitrary image from disk and apply preprocessing for model inference."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    img = Image.open(path).convert("RGB")
    transform = get_single_image_transform(image_size=image_size)
    tensor = transform(img).unsqueeze(0)  # Shape: [1, 3, H, W]
    return tensor


def get_dataloaders(
    data_dir: str = "data",
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 0,
) -> Tuple[Dict[str, DataLoader], Dict[str, int], List[str]]:
    """Create DataLoaders for train, val, and test splits using ImageFolder.
    
    Returns:
        dataloaders: Dict containing 'train', 'val', and 'test' DataLoaders.
        dataset_sizes: Dict containing sample counts per split.
        class_names: List of class names sorted by class index.
    """
    base_path = Path(data_dir)
    data_transforms = get_transforms(image_size=image_size)

    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}

    splits = ["train", "val", "test"]
    class_names: List[str] = []

    for split in splits:
        split_dir = base_path / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}. "
                f"Please run prepare_data.py first to create the dataset splits."
            )

        dataset = datasets.ImageFolder(
            root=str(split_dir),
            transform=data_transforms[split]
        )
        image_datasets[split] = dataset
        dataset_sizes[split] = len(dataset)

        shuffle = (split == "train")
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

        if not class_names:
            class_names = dataset.classes

    print(f"[VisionGuard DataLoader] Loaded splits from '{data_dir}':")
    print(f"  - Train samples : {dataset_sizes['train']}")
    print(f"  - Val samples   : {dataset_sizes['val']}")
    print(f"  - Test samples  : {dataset_sizes['test']}")
    print(f"  - Classes       : {class_names} (indices: {image_datasets['train'].class_to_idx})")

    return dataloaders, dataset_sizes, class_names


if __name__ == "__main__":
    # Test DataLoader setup if data directory exists
    try:
        loaders, sizes, classes = get_dataloaders("data", batch_size=16)
        print("\nTesting single batch from Train Loader:")
        for images, labels in loaders["train"]:
            print(f"  Batch images shape: {images.shape}")
            print(f"  Batch labels shape: {labels.shape}")
            print(f"  Sample labels     : {labels[:5].tolist()}")
            break
        print("DataLoaders initialized successfully!")
    except Exception as e:
        print(f"Dataset check: {e}")
