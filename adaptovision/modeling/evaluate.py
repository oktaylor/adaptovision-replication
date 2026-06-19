"""Evaluation script for trained AdaptoVision checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn

from adaptovision.dataset import build_dataloaders
from adaptovision.models.adaptovision import AdaptoVision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AdaptoVision checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. If omitted, best.pt is found from project.run_name.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def find_checkpoint_from_config(config: dict) -> Path:
    """Find the newest best.pt matching project.run_name."""
    project_cfg = config["project"]
    data_cfg = config["data"]

    runs_dir = Path(project_cfg.get("runs_dir", "outputs/runs"))
    dataset_name = data_cfg["dataset"]
    run_name = project_cfg.get("run_name", "")

    if not run_name:
        raise ValueError(
            "project.run_name is empty. Provide --checkpoint manually or set project.run_name."
        )

    pattern = f"*_{dataset_name}_{run_name}/checkpoints/best.pt"
    candidates = list(runs_dir.glob(pattern))

    if not candidates:
        raise FileNotFoundError(
            "No checkpoint found.\n"
            f"Search directory: {runs_dir}\n"
            f"Pattern: {pattern}\n"
            "Either check project.run_name or pass --checkpoint manually."
        )

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


@torch.no_grad()
def collect_predictions(model: nn.Module, loader, device: torch.device):
    model.eval()

    all_preds = []
    all_targets = []
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * targets.size(0)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    accuracy = total_correct / total_samples
    avg_loss = total_loss / total_samples
    return avg_loss, accuracy, all_preds, all_targets


def infer_output_dir(checkpoint_path: Path) -> Path:
    """Infer evaluation output directory from checkpoint path."""
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent / "evaluation"
    return checkpoint_path.parent / "evaluation"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = find_checkpoint_from_config(config)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = infer_output_dir(checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    _, _, test_loader = build_dataloaders(config)

    model = build_model(config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    test_loss, test_acc, preds, targets = collect_predictions(model, test_loader, device)

    report = classification_report(targets, preds, digits=4)
    cm = confusion_matrix(targets, preds)

    result_text = "\n".join(
        [
            "=" * 80,
            "AdaptoVision CIFAR-10 Test Evaluation",
            f"Config: {args.config}",
            f"Run name: {config['project'].get('run_name', 'unknown')}",
            f"Checkpoint: {checkpoint_path}",
            f"Device: {device}",
            f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}",
            f"Best validation accuracy in checkpoint: {checkpoint.get('best_val_acc', 'unknown')}",
            f"Test loss: {test_loss:.4f}",
            f"Test accuracy: {test_acc:.4f}",
            "=" * 80,
            "",
            "Classification report:",
            report,
            "",
            "Confusion matrix:",
            str(cm),
            "",
        ]
    )

    print(result_text)

    result_path = output_dir / "test_results.txt"
    cm_path = output_dir / "confusion_matrix.txt"

    result_path.write_text(result_text, encoding="utf-8")
    cm_path.write_text(str(cm), encoding="utf-8")

    print(f"Saved test results to: {result_path}")
    print(f"Saved confusion matrix to: {cm_path}")


if __name__ == "__main__":
    main()
