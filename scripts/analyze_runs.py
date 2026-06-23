"""Analyze AdaptoVision experiment runs.

This script collects:
- epoch-wise train/validation metrics from metrics.csv
- test accuracy/loss from evaluation/test_results.txt
- confusion matrix from evaluation/confusion_matrix.txt

Outputs:
- outputs/analysis/summary.csv
- outputs/analysis/summary.md
- plots for LR vs accuracy, test accuracy ranking, train/val curves
- normalized confusion matrix for the best run
- top class confusions
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze AdaptoVision run outputs.")
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="outputs/runs",
        help="Directory containing run folders.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/analysis",
        help="Directory for analysis outputs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of top runs to include in curve plots.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data or {}


def get_nested(config: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def parse_test_results(path: Path) -> dict[str, Any]:
    result = {
        "test_loss": np.nan,
        "test_acc": np.nan,
        "checkpoint_epoch": np.nan,
        "checkpoint_best_val_acc": np.nan,
    }

    if not path.exists():
        return result

    text = path.read_text(encoding="utf-8", errors="replace")

    patterns = {
        "test_loss": r"Test loss:\s*([0-9.]+)",
        "test_acc": r"Test accuracy:\s*([0-9.]+)",
        "checkpoint_epoch": r"Checkpoint epoch:\s*([0-9]+)",
        "checkpoint_best_val_acc": r"Best validation accuracy in checkpoint:\s*([0-9.]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value = match.group(1)
            result[key] = float(value) if "." in value else int(value)

    return result


def parse_confusion_matrix(path: Path, num_classes: int = 10) -> np.ndarray | None:
    if not path.exists():
        return None

    text = path.read_text(encoding="utf-8", errors="replace")
    numbers = [int(x) for x in re.findall(r"-?\d+", text)]

    expected = num_classes * num_classes
    if len(numbers) != expected:
        print(f"[WARN] Could not parse {path}: expected {expected} numbers, found {len(numbers)}")
        return None

    return np.array(numbers, dtype=int).reshape(num_classes, num_classes)


def run_label(run_dir: Path) -> str:
    """Make run folder name shorter for plots."""
    name = run_dir.name

    # Example: 20260618_123456_CIFAR10_lr0050 -> lr0050
    parts = name.split("_")
    if len(parts) >= 4:
        return "_".join(parts[3:])
    return name


def collect_one_run(run_dir: Path) -> dict[str, Any] | None:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None

    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        return None

    for col in ["epoch", "learning_rate", "train_loss", "train_acc", "val_loss", "val_acc", "best_val_acc"]:
        if col in metrics.columns:
            metrics[col] = pd.to_numeric(metrics[col], errors="coerce")

    config = load_yaml(run_dir / "config.yaml")
    eval_result = parse_test_results(run_dir / "evaluation" / "test_results.txt")

    best_idx = metrics["val_acc"].idxmax()
    best_row = metrics.loc[best_idx]
    final_row = metrics.iloc[-1]

    final_epoch = int(metrics["epoch"].max())
    best_epoch = int(best_row["epoch"])
    best_val_acc = float(best_row["val_acc"])

    # Last-20-epoch val slope: positive slope means validation may still be improving.
    tail = metrics.tail(min(20, len(metrics))).dropna(subset=["epoch", "val_acc"])
    if len(tail) >= 2:
        slope = float(np.polyfit(tail["epoch"].to_numpy(), tail["val_acc"].to_numpy(), deg=1)[0])
    else:
        slope = np.nan

    # Heuristic: possibly needs more epochs if best epoch is near the end
    # or validation accuracy is still increasing in the tail.
    best_epoch_ratio = best_epoch / final_epoch if final_epoch > 0 else np.nan
    best_near_end = best_epoch_ratio >= 0.90
    tail_still_rising = bool(slope > 0.0002) if not np.isnan(slope) else False
    possible_epoch_limited = bool(best_near_end or tail_still_rising)

    lr_from_config = get_nested(config, ["training", "learning_rate"], np.nan)
    epochs_from_config = get_nested(config, ["training", "epochs"], final_epoch)
    batch_size = get_nested(config, ["training", "batch_size"], np.nan)
    weight_decay = get_nested(config, ["training", "weight_decay"], np.nan)
    grad_clip = get_nested(config, ["training", "grad_clip_norm"], None)
    seed = get_nested(config, ["project", "seed"], np.nan)

    dropout_rates = get_nested(config, ["model", "dropout_rates"], None)
    stage_channels = get_nested(config, ["model", "stage_channels"], None)
    rotation = get_nested(config, ["augmentation", "random_rotation_degrees"], None)
    shear = get_nested(config, ["augmentation", "random_affine_shear"], None)
    crop_size = get_nested(config, ["augmentation", "random_resized_crop_size"], None)
    crop_padding = get_nested(config, ["augmentation", "random_crop_padding"], None)

    return {
        "run_dir": str(run_dir),
        "run": run_label(run_dir),
        "lr_config": lr_from_config,
        "lr_first_epoch": float(metrics["learning_rate"].iloc[0]),
        "epochs_config": epochs_from_config,
        "final_epoch": final_epoch,
        "best_epoch": best_epoch,
        "best_epoch_ratio": best_epoch_ratio,
        "best_val_acc": best_val_acc,
        "final_val_acc": float(final_row["val_acc"]),
        "final_train_acc": float(final_row["train_acc"]),
        "final_train_loss": float(final_row["train_loss"]),
        "final_val_loss": float(final_row["val_loss"]),
        "generalization_gap_final": float(final_row["train_acc"] - final_row["val_acc"]),
        "last20_val_slope": slope,
        "possible_epoch_limited": possible_epoch_limited,
        "test_acc": eval_result["test_acc"],
        "test_loss": eval_result["test_loss"],
        "checkpoint_epoch": eval_result["checkpoint_epoch"],
        "checkpoint_best_val_acc": eval_result["checkpoint_best_val_acc"],
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "grad_clip_norm": grad_clip,
        "seed": seed,
        "dropout_rates": str(dropout_rates),
        "stage_channels": str(stage_channels),
        "rotation": rotation,
        "shear": str(shear),
        "crop_size": crop_size,
        "crop_padding": crop_padding,
    }


def collect_runs(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(runs_dir.glob("*/metrics.csv")):
        row = collect_one_run(metrics_path.parent)
        if row is not None:
            rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No metrics.csv files found under {runs_dir}")

    df = pd.DataFrame(rows)

    # Sort primarily by test accuracy if available, otherwise best validation accuracy.
    if df["test_acc"].notna().any():
        df = df.sort_values(["test_acc", "best_val_acc"], ascending=False)
    else:
        df = df.sort_values(["best_val_acc"], ascending=False)

    return df.reset_index(drop=True)


def save_markdown_summary(df: pd.DataFrame, path: Path) -> None:
    cols = [
        "run",
        "lr_config",
        "best_val_acc",
        "test_acc",
        "best_epoch",
        "final_epoch",
        "possible_epoch_limited",
        "generalization_gap_final",
        "last20_val_slope",
        "grad_clip_norm",
    ]

    available_cols = [col for col in cols if col in df.columns]
    top = df[available_cols].copy()

    text = []
    text.append("# AdaptoVision Experiment Summary")
    text.append("")
    text.append("## Top runs")
    text.append("")
    text.append(top.to_markdown(index=False))
    text.append("")
    text.append("## Notes")
    text.append("")
    text.append("- `possible_epoch_limited=True` means the best validation epoch was near the end, or the validation curve was still rising.")
    text.append("- `generalization_gap_final = final_train_acc - final_val_acc`.")
    text.append("- If test accuracy is NaN, run `adaptovision.modeling.evaluate` for that run first.")
    text.append("")

    path.write_text("\n".join(text), encoding="utf-8")


def plot_test_accuracy(df: pd.DataFrame, out_dir: Path) -> None:
    data = df.dropna(subset=["test_acc"]).copy()
    if data.empty:
        return

    data = data.sort_values("test_acc", ascending=True)

    plt.figure(figsize=(10, max(4, 0.35 * len(data))))
    plt.barh(data["run"], data["test_acc"])
    plt.xlabel("Test accuracy")
    plt.ylabel("Run")
    plt.title("Test accuracy by run")
    plt.tight_layout()
    plt.savefig(out_dir / "test_accuracy_by_run.png", dpi=200)
    plt.close()


def plot_lr_vs_accuracy(df: pd.DataFrame, out_dir: Path) -> None:
    data = df.dropna(subset=["lr_config", "test_acc"]).copy()
    if data.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.scatter(data["lr_config"], data["test_acc"])
    for _, row in data.iterrows():
        plt.annotate(row["run"], (row["lr_config"], row["test_acc"]), fontsize=8)
    plt.xlabel("Initial learning rate")
    plt.ylabel("Test accuracy")
    plt.title("Learning rate vs test accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "lr_vs_test_accuracy.png", dpi=200)
    plt.close()


def plot_train_val_curves(df: pd.DataFrame, out_dir: Path, top_k: int) -> None:
    # Prefer top test accuracy; if missing, use best val acc.
    if df["test_acc"].notna().any():
        top_df = df.dropna(subset=["test_acc"]).head(top_k)
    else:
        top_df = df.head(top_k)

    plt.figure(figsize=(10, 6))
    for _, row in top_df.iterrows():
        metrics_path = Path(row["run_dir"]) / "metrics.csv"
        metrics = pd.read_csv(metrics_path)
        plt.plot(metrics["epoch"], metrics["val_acc"], label=f"{row['run']} val")
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title(f"Validation accuracy curves, top {len(top_df)} runs")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "top_val_accuracy_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    for _, row in top_df.iterrows():
        metrics_path = Path(row["run_dir"]) / "metrics.csv"
        metrics = pd.read_csv(metrics_path)
        plt.plot(metrics["epoch"], metrics["train_acc"], label=f"{row['run']} train")
    plt.xlabel("Epoch")
    plt.ylabel("Training accuracy")
    plt.title(f"Training accuracy curves, top {len(top_df)} runs")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "top_train_accuracy_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    for _, row in top_df.iterrows():
        metrics_path = Path(row["run_dir"]) / "metrics.csv"
        metrics = pd.read_csv(metrics_path)
        plt.plot(metrics["epoch"], metrics["val_loss"], label=f"{row['run']} val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title(f"Validation loss curves, top {len(top_df)} runs")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "top_val_loss_curves.png", dpi=200)
    plt.close()


def plot_best_confusion_matrix(df: pd.DataFrame, out_dir: Path) -> None:
    data = df.dropna(subset=["test_acc"]).copy()
    if data.empty:
        data = df.copy()

    if data.empty:
        return

    best = data.iloc[0]
    cm_path = Path(best["run_dir"]) / "evaluation" / "confusion_matrix.txt"
    cm = parse_confusion_matrix(cm_path, num_classes=10)
    if cm is None:
        print(f"[WARN] No confusion matrix found for best run: {best['run']}")
        return

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.maximum(row_sums, 1)

    np.savetxt(out_dir / "best_confusion_matrix_raw.csv", cm, fmt="%d", delimiter=",")
    np.savetxt(out_dir / "best_confusion_matrix_normalized.csv", cm_norm, fmt="%.6f", delimiter=",")

    plt.figure(figsize=(8, 7))
    plt.imshow(cm_norm)
    plt.colorbar(label="Recall-normalized count")
    plt.xticks(range(10), CIFAR10_CLASSES, rotation=45, ha="right")
    plt.yticks(range(10), CIFAR10_CLASSES)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title(f"Normalized confusion matrix: {best['run']}")

    for i in range(10):
        for j in range(10):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_dir / "best_confusion_matrix_normalized.png", dpi=200)
    plt.close()

    recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)

    plt.figure(figsize=(8, 5))
    plt.bar(CIFAR10_CLASSES, recall)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Recall")
    plt.title(f"Per-class recall: {best['run']}")
    plt.tight_layout()
    plt.savefig(out_dir / "best_per_class_recall.png", dpi=200)
    plt.close()

    # Top off-diagonal mistakes.
    mistakes = []
    for i, true_name in enumerate(CIFAR10_CLASSES):
        for j, pred_name in enumerate(CIFAR10_CLASSES):
            if i == j:
                continue
            mistakes.append(
                {
                    "true_class": true_name,
                    "predicted_class": pred_name,
                    "count": int(cm[i, j]),
                    "rate_within_true_class": float(cm_norm[i, j]),
                }
            )

    mistakes_df = pd.DataFrame(mistakes).sort_values(
        ["count", "rate_within_true_class"],
        ascending=False,
    )
    mistakes_df.to_csv(out_dir / "best_top_confusions.csv", index=False)


def main() -> None:
    args = parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_runs(runs_dir)

    summary_path = out_dir / "summary.csv"
    df.to_csv(summary_path, index=False)

    save_markdown_summary(df, out_dir / "summary.md")

    plot_test_accuracy(df, out_dir)
    plot_lr_vs_accuracy(df, out_dir)
    plot_train_val_curves(df, out_dir, top_k=args.top_k)
    plot_best_confusion_matrix(df, out_dir)

    print("=" * 80)
    print(f"Saved summary to: {summary_path}")
    print(f"Saved markdown summary to: {out_dir / 'summary.md'}")
    print(f"Saved plots to: {out_dir}")
    print("=" * 80)

    display_cols = [
        "run",
        "lr_config",
        "best_val_acc",
        "test_acc",
        "best_epoch",
        "final_epoch",
        "possible_epoch_limited",
        "generalization_gap_final",
    ]
    display_cols = [col for col in display_cols if col in df.columns]
    print(df[display_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
