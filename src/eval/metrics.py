"""Comprehensive evaluation metrics for BCS classification."""
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    recall_score, precision_score
)
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


@torch.no_grad()
def evaluate(model, loader, device, class_names: Optional[List[str]] = None) -> Dict:
    """
    Comprehensive evaluation with multiple metrics.
    
    Args:
        model: PyTorch model
        loader: DataLoader for evaluation
        device: Device to run evaluation on
        class_names: Optional list of class names for labeling
    
    Returns:
        Dictionary with:
            - acc: Accuracy
            - macro_f1: Macro-averaged F1 score
            - weighted_f1: Weighted F1 score
            - underweight_recall: Recall for class 0 (assumed underweight)
            - per_class_recall: List of recall per class
            - per_class_precision: List of precision per class
            - confusion_matrix: 2D numpy array
            - class_names: List of class names
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    per_class_recall = recall_score(all_labels, all_preds, average=None)
    per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Underweight recall (class 0) - adjust if your mapping differs
    underweight_recall = per_class_recall[0] if len(per_class_recall) > 0 else 0.0
    
    if class_names is None:
        num_classes = len(per_class_recall)
        # Default: BCS values [3.25, 3.5, 3.75, 4.0, 4.25]
        default_names = ["3.25", "3.5", "3.75", "4.0", "4.25"]
        class_names = default_names[:num_classes]
    
    results = {
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "underweight_recall": float(underweight_recall),
        "per_class_recall": per_class_recall.tolist(),
        "per_class_precision": per_class_precision.tolist(),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names
    }
    
    return results


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Optional[str] = None):
    """
    Plot and optionally save confusion matrix.
    
    Args:
        cm: Confusion matrix (numpy array or list)
        class_names: List of class names
        save_path: Optional path to save the figure
    """
    if isinstance(cm, list):
        cm = np.array(cm)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix to {save_path}")
    
    plt.close()

