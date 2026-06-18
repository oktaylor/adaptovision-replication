"""Dataset and dataloader utilities for CIFAR experiments."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_transforms(config: dict, train: bool = True) -> transforms.Compose:
    """Build torchvision transforms."""
    image_size = config["data"]["image_size"]

    if train:
        aug_cfg = config.get("augmentation", {})
        transform_list = []

        # Paper-style random crop:
        # CIFAR 32x32 -> random crop 26x26 -> resize back to 32x32.
        crop_size = aug_cfg.get("random_resized_crop_size", None)
        if crop_size is not None:
            transform_list.extend(
                [
                    transforms.RandomCrop(crop_size),
                    transforms.Resize((image_size, image_size)),
                ]
            )
        elif aug_cfg.get("random_crop_padding", 0) > 0:
            transform_list.append(
                transforms.RandomCrop(
                    image_size,
                    padding=aug_cfg["random_crop_padding"],
                )
            )

        if aug_cfg.get("random_horizontal_flip", True):
            transform_list.append(transforms.RandomHorizontalFlip())

        rotation = aug_cfg.get("random_rotation_degrees", 0)
        shear = aug_cfg.get("random_affine_shear", None)

        # Use RandomAffine so rotation and affine shear are applied together.
        # Note: torchvision's shear argument is interpreted in degrees.
        if rotation > 0 or shear is not None:
            transform_list.append(
                transforms.RandomAffine(
                    degrees=(-rotation, rotation) if rotation > 0 else 0,
                    shear=shear,
                )
            )

        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )

        return transforms.Compose(transform_list)

    # Validation/test: no augmentation.
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
    eval_transform = build_transforms(config, train=False)

    # Make two CIFAR-10 train datasets with different transforms.
    # The train split uses augmentation.
    train_dataset_full = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        transform=train_transform,
        download=True,
    )

    # The validation split uses eval transform only.
    val_dataset_full = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        transform=eval_transform,
        download=True,
    )

    test_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        transform=eval_transform,
        download=True,
    )

    val_ratio = config["data"].get("val_ratio", 0.1)
    num_train_total = len(train_dataset_full)

    val_size = int(num_train_total * val_ratio)
    train_size = num_train_total - val_size

    generator = torch.Generator().manual_seed(config["project"].get("seed", 42))
    indices = torch.randperm(num_train_total, generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    print(
        f"Dataset split: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
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