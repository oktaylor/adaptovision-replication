"""Training script for AdaptoVision on CIFAR-10."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from tqdm import tqdm

from adaptovision.dataset import build_dataloaders
from adaptovision.models.adaptovision import AdaptoVision, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AdaptoVision on CIFAR-10.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name. Overrides project.run_name in the config.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_name(text: str) -> str:
    """Make a string safe for folder names."""
    return (
        text.replace(" ", "_")
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
    )


def create_run_dirs(config: dict, config_path: str, run_name_override: str | None = None) -> dict[str, Path]:
    """Create a unique run directory and return important output paths."""
    dataset_name = config["data"]["dataset"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = run_name_override or config.get("project", {}).get("run_name", "")
    run_name = safe_name(run_name)

    if run_name:
        folder_name = f"{timestamp}_{dataset_name}_{run_name}"
    else:
        folder_name = f"{timestamp}_{dataset_name}"

    runs_dir = Path(config["project"].get("runs_dir", "outputs/runs"))
    run_dir = runs_dir / folder_name
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"

    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=True)

    copied_config_path = run_dir / "config.yaml"
    shutil.copy2(config_path, copied_config_path)

    paths = {
        "run_dir": run_dir,
        "checkpoint_dir": checkpoint_dir,
        "log_dir": log_dir,
        "metrics_path": run_dir / "metrics.csv",
        "summary_path": log_dir / "train_summary.txt",
        "config_copy_path": copied_config_path,
    }

    return paths


def build_model(config: dict) -> AdaptoVision:
    model_cfg = config["model"]

    return AdaptoVision(
        in_channels=model_cfg["in_channels"],
        num_classes=model_cfg["num_classes"],
        base_channels=model_cfg["base_channels"],
        stage_channels=model_cfg["stage_channels"],
        blocks_per_stage=model_cfg["blocks_per_stage"],
        dropout_rates=model_cfg["dropout_rates"],
        activation=model_cfg.get("activation", "elu"),
        depthwise_kernel_sizes=model_cfg.get("depthwise_kernel_sizes", (3, 5, 7, 7)),
        learnable_skip_weights=model_cfg.get("learnable_skip_weights", True),
    )


def build_optimizer(config: dict, model: nn.Module) -> torch.optim.Optimizer:
    train_cfg = config["training"]

    if train_cfg.get("optimizer", "sgd").lower() != "sgd":
        raise ValueError("This replication script currently supports SGD only.")

    return torch.optim.SGD(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        momentum=train_cfg["momentum"],
        weight_decay=train_cfg["weight_decay"],
        nesterov=train_cfg.get("nesterov", False),
    )


def build_scheduler(config: dict, optimizer: torch.optim.Optimizer):
    scheduler_cfg = config["scheduler"]

    if scheduler_cfg.get("name", "exponential").lower() != "exponential":
        return None

    decay_factor = scheduler_cfg["decay_factor"]
    decay_every_epochs = scheduler_cfg["decay_every_epochs"]
    gamma_per_epoch = decay_factor ** (1.0 / decay_every_epochs)

    return torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=gamma_per_epoch,
    )


def set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    """Set learning rate for all optimizer parameter groups."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def compute_epoch_lr(config: dict, epoch: int) -> float:
    """Compute epoch-level learning rate with optional linear warmup.

    If training.warmup_epochs is not set or is 0, this function should not be used.
    Warmup is applied first, then the configured exponential decay starts after warmup.
    """
    train_cfg = config["training"]
    scheduler_cfg = config.get("scheduler", {})

    base_lr = float(train_cfg["learning_rate"])
    warmup_epochs = int(train_cfg.get("warmup_epochs", 0) or 0)
    warmup_start_lr = float(train_cfg.get("warmup_start_lr", base_lr * 0.01))

    if warmup_epochs <= 0:
        return base_lr

    if epoch <= warmup_epochs:
        if warmup_epochs == 1:
            return base_lr

        alpha = (epoch - 1) / (warmup_epochs - 1)
        return warmup_start_lr + alpha * (base_lr - warmup_start_lr)

    scheduler_name = scheduler_cfg.get("name", "exponential").lower()

    if scheduler_name == "exponential":
        decay_factor = float(scheduler_cfg["decay_factor"])
        decay_every_epochs = float(scheduler_cfg["decay_every_epochs"])
        gamma_per_epoch = decay_factor ** (1.0 / decay_every_epochs)

        # Epoch immediately after warmup gets one decay step.
        decay_steps = epoch - warmup_epochs - 1
        return base_lr * (gamma_per_epoch ** decay_steps)

    if scheduler_name in {"none", "constant", "null"}:
        return base_lr

    raise ValueError(
        f"Warmup is currently implemented for exponential/constant schedules only, "
        f"but got scheduler.name={scheduler_name!r}."
    )


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float | None = None,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(loader, desc="train", leave=False)

    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, targets)

        if torch.isnan(loss):
            raise FloatingPointError("NaN loss detected. Try lowering the learning rate.")

        loss.backward()

        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

        progress.set_postfix(
            loss=total_loss / total_samples,
            acc=total_correct / total_samples,
        )

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    split: str = "val",
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(loader, desc=split, leave=False)

    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

        progress.set_postfix(
            loss=total_loss / total_samples,
            acc=total_correct / total_samples,
        )

    return total_loss / total_samples, total_correct / total_samples


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_val_acc: float,
    config: dict,
    run_dir: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "config": config,
        "run_dir": str(run_dir),
    }

    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(payload, path)


def append_summary(summary_path: Path, text: str) -> None:
    with open(summary_path, "a", encoding="utf-8") as file:
        file.write(text + "\n")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_seed(config["project"].get("seed", 42))
    device = get_device()

    paths = create_run_dirs(config, args.config, args.run_name)
    run_dir = paths["run_dir"]
    checkpoint_dir = paths["checkpoint_dir"]
    metrics_path = paths["metrics_path"]
    summary_path = paths["summary_path"]

    train_loader, val_loader, _ = build_dataloaders(config)

    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    warmup_epochs = int(config["training"].get("warmup_epochs", 0) or 0)
    use_manual_lr_schedule = warmup_epochs > 0

    if use_manual_lr_schedule:
        scheduler = None

    num_params = count_parameters(model)

    print("=" * 80)
    print("AdaptoVision CIFAR-10 Training")
    print(f"Device: {device}")
    print(f"Run directory: {run_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Metrics CSV: {metrics_path}")
    print(f"Trainable parameters: {num_params:,}")
    print("=" * 80)

    append_summary(summary_path, "AdaptoVision CIFAR-10 Training")
    append_summary(summary_path, f"Run directory: {run_dir}")
    append_summary(summary_path, f"Device: {device}")
    append_summary(summary_path, f"Trainable parameters: {num_params:,}")
    append_summary(summary_path, f"Config copy: {paths['config_copy_path']}")
    if use_manual_lr_schedule:
        append_summary(
            summary_path,
            (
                "LR schedule: manual linear warmup + configured decay | "
                f"warmup_epochs={warmup_epochs}, "
                f"warmup_start_lr={config['training'].get('warmup_start_lr')}, "
                f"base_lr={config['training']['learning_rate']}"
            ),
        )
    
    with open(metrics_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "epoch",
                "learning_rate",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "best_val_acc",
                "best_checkpoint",
                "last_checkpoint",
            ]
        )

    best_val_acc = 0.0
    epochs = config["training"]["epochs"]
    grad_clip_norm = config["training"].get("grad_clip_norm", None)

    best_checkpoint_path = checkpoint_dir / "best.pt"
    last_checkpoint_path = checkpoint_dir / "last.pt"

    for epoch in range(1, epochs + 1):
        if use_manual_lr_schedule:
            current_lr = compute_epoch_lr(config, epoch)
            set_optimizer_lr(optimizer, current_lr)
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch}/{epochs} | lr={current_lr:.6f}")

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=grad_clip_norm,
        )

        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            split="val",
        )

        if scheduler is not None and not use_manual_lr_schedule:
            scheduler.step()

        save_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_acc,
            config,
            run_dir,
        )

        is_best = val_acc > best_val_acc
        if is_best:
            previous_best = best_val_acc
            best_val_acc = val_acc

            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_acc,
                config,
                run_dir,
            )

            message = (
                "[BEST CHECKPOINT SAVED] "
                f"epoch={epoch}, "
                f"previous_best_val_acc={previous_best:.4f}, "
                f"new_best_val_acc={best_val_acc:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"path={best_checkpoint_path}"
            )
            print(message)
            append_summary(summary_path, message)

        print(
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
            f"best_val_acc={best_val_acc:.4f}"
        )

        with open(metrics_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    epoch,
                    current_lr,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                    best_val_acc,
                    str(best_checkpoint_path) if best_checkpoint_path.exists() else "",
                    str(last_checkpoint_path),
                ]
            )

    print("\nTraining finished.")
    print(f"Run directory: {run_dir}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Last checkpoint: {last_checkpoint_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Summary saved to: {summary_path}")

    append_summary(summary_path, "Training finished.")
    append_summary(summary_path, f"Best validation accuracy: {best_val_acc:.4f}")
    append_summary(summary_path, f"Best checkpoint: {best_checkpoint_path}")
    append_summary(summary_path, f"Last checkpoint: {last_checkpoint_path}")
    append_summary(summary_path, f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
