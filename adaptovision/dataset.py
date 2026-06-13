"""Dataset and dataloader utilities for CIFAR experiments."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_transforms(config: dict, train: bool = True) -> transforms.Compose:
    """Build torchvision transforms."""
    image_size = config["data"]["image_size"]

    if train:
        aug_cfg = config.get("augmentation", {})
        transform_list = []

        if aug_cfg.get("random_crop_padding", 0) > 0:
            transform_list.append(
                transforms.RandomCrop(
                    image_size,
                    padding=aug_cfg["random_crop_padding"],
                )
            )

        if aug_cfg.get("random_horizontal_flip", True):
            transform_list.append(transforms.RandomHorizontalFlip())

        rotation = aug_cfg.get("random_rotation_degrees", 0)
        if rotation > 0:
            transform_list.append(transforms.RandomRotation(rotation))

        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
        return transforms.Compose(transform_list)

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    data_dir = Path(config["data"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    batch_size = config["training"]["batch_size"]
    num_workers = config["data"].get("num_workers", 4)
    pin_memory = config["data"].get("pin_memory", True)

    train_transform = build_transforms(config, train=True)
    test_transform = build_transforms(config, train=False)

    full_train_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        transform=train_transform,
        download=True,
    )

    test_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        transform=test_transform,
        download=True,
    )

    val_ratio = config["data"].get("val_ratio", 0.1)
    val_size = int(len(full_train_dataset) * val_ratio)
    train_size = len(full_train_dataset) - val_size

    generator = torch.Generator().manual_seed(config["project"].get("seed", 42))
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
