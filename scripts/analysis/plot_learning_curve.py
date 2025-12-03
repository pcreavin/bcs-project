"""
Plot learning curves for the ablation_full_enhanced run using seaborn.

Parses epoch summaries from logs/full_enhanced_20251106_213521.log and plots
train loss alongside validation accuracy and macro-F1.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

LOG_PATH = Path("logs/full_enhanced_20251106_213521.log")

LINE_RE = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+Train Loss:\s+([\d.]+)\s+\|\s+Val Acc:\s+([\d.]+)\s+\|\s+Val Macro-F1:\s+([\d.]+)\s+\|\s+Underweight Recall:\s+([\d.]+)"
)


def parse_log(path: Path) -> pd.DataFrame:
    """Parse training log and return DataFrame indexed by epoch."""
    records = []
    for line in path.read_text().splitlines():
        match = LINE_RE.search(line)
        if match:
            epoch, train_loss, val_acc, val_macro_f1, underweight_recall = match.groups()
            records.append(
                {
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "val_acc": float(val_acc),
                    "val_macro_f1": float(val_macro_f1),
                    "underweight_recall": float(underweight_recall),
                }
            )

    df = pd.DataFrame(records).set_index("epoch")
    if df.empty:
        raise RuntimeError(f"No epoch records parsed from {path}")
    return df


def plot_learning_curves(df: pd.DataFrame) -> None:
    """Plot train loss and validation metrics."""
    sns.set_theme(style="whitegrid", context="talk")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    sns.lineplot(
        data=df,
        x=df.index,
        y="train_loss",
        marker="o",
        ax=ax1,
        label="Train Loss",
        color="#1f77b4",
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")

    ax2 = ax1.twinx()
    sns.lineplot(
        data=df,
        x=df.index,
        y="val_macro_f1",
        marker="s",
        ax=ax2,
        label="Val Macro-F1",
        color="#ff7f0e",
    )
    sns.lineplot(
        data=df,
        x=df.index,
        y="val_acc",
        marker="^",
        ax=ax2,
        label="Val Accuracy",
        color="#2ca02c",
    )
    ax2.set_ylabel("Validation Metrics")

    ax1.set_title("EfficientNet-B0 (Enhanced) Learning Curves")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot learning curves for ablation_full_enhanced.")
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("outputs/ablation_full_enhanced/learning_curve.png"),
        help="Path to save the figure (default: outputs/ablation_full_enhanced/learning_curve.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure interactively (may require DISPLAY).",
    )
    args = parser.parse_args()

    df = parse_log(LOG_PATH)
    fig = plot_learning_curves(df)

    args.save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=300, bbox_inches="tight")
    print(f"Saved learning curve to {args.save.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()

