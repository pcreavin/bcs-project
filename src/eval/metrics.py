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
def evaluate(model, loader, device, class_names: Optional[List[str]] = None, head_type: str = "classification", ordinal_decoding_method: str = "threshold_count") -> Dict:
    """
    Comprehensive evaluation with multiple metrics.
    
    Args:
        model: PyTorch model
        loader: DataLoader for evaluation
        device: Device to run evaluation on
        class_names: Optional list of class names for labeling
        head_type: Type of model head ("classification" or "ordinal")
    
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
            - mae: Mean Absolute Error (if ordinal head)
            - ordinal_accuracy_1: Accuracy within ±1 class (if ordinal head)
    """
    from ..models.heads import OrdinalHead
    
    model.eval()
    all_preds = []
    all_labels = []
    
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        outputs = model(xb)
        
        if head_type == "ordinal":
            # Decode ordinal logits to class predictions
            pred = OrdinalHead.decode(outputs, method=ordinal_decoding_method)
        else:
            # Standard classification
            pred = outputs.argmax(1)
        
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
    
    # Add ordinal-specific metrics if using ordinal head
    if head_type == "ordinal":
        # Mean Absolute Error in class space
        mae_class = np.mean(np.abs(all_preds - all_labels))
        results["mae_class"] = float(mae_class)
        
        # Mean Absolute Error in BCS space (if class_names are numeric)
        try:
            bcs_values = np.array([float(name) for name in class_names])
            pred_bcs = bcs_values[all_preds]
            true_bcs = bcs_values[all_labels]
            mae_bcs = np.mean(np.abs(pred_bcs - true_bcs))
            results["mae_bcs"] = float(mae_bcs)
        except (ValueError, IndexError):
            pass  # Skip if class names aren't numeric
        
        # Ordinal accuracy: within ±1 class
        ordinal_acc_1 = np.mean(np.abs(all_preds - all_labels) <= 1)
        results["ordinal_accuracy_1"] = float(ordinal_acc_1)
        
        # Within ±0.25 BCS (if applicable)
        try:
            bcs_values = np.array([float(name) for name in class_names])
            pred_bcs = bcs_values[all_preds]
            true_bcs = bcs_values[all_labels]
            ordinal_acc_025 = np.mean(np.abs(pred_bcs - true_bcs) <= 0.25)
            results["ordinal_accuracy_025"] = float(ordinal_acc_025)
        except (ValueError, IndexError):
            pass
    
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

