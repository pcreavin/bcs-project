#!/usr/bin/env python3
"""Evaluate ensemble of any two classification models (general version)."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import os
import json
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    recall_score, precision_score, mean_absolute_error
)

from src.train.dataset import BcsDataset
from src.models import create_model
from src.eval.metrics import plot_confusion_matrix


def load_model(checkpoint_path: str, config_path: str, device: str):
    """Load a classification model from checkpoint."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    model_cfg = cfg.get("model", {})
    
    backbone = model_cfg.get("backbone", "efficientnet_b0")
    num_classes = int(model_cfg.get("num_classes", 5))
    pretrained = bool(model_cfg.get("pretrained", True))
    finetune_mode = model_cfg.get("finetune_mode", "full")
    
    model = create_model(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        finetune_mode=finetune_mode
    )
    model.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    return model, cfg


def ensemble_predict_majority(
    pred1: np.ndarray,
    pred2: np.ndarray
) -> np.ndarray:
    """
    Simple majority vote ensemble (for 2 models, this is just agreement).
    
    Rule:
    - Same prediction → use it
    - Different predictions → use model 1 (baseline)
    
    Args:
        pred1: Predictions from model 1 (shape: N,)
        pred2: Predictions from model 2 (shape: N,)
    
    Returns:
        Ensemble predictions (shape: N,)
    """
    ensemble_pred = np.zeros_like(pred1)
    
    # Same prediction → use it
    same_mask = pred1 == pred2
    ensemble_pred[same_mask] = pred1[same_mask]
    
    # Different predictions → use model 1 (baseline)
    diff_mask = pred1 != pred2
    ensemble_pred[diff_mask] = pred1[diff_mask]
    
    return ensemble_pred


def ensemble_predict_weighted(
    pred1: np.ndarray,
    pred2: np.ndarray,
    class_diff: np.ndarray,
    prefer_model2_for_large_diff: bool = True
) -> np.ndarray:
    """
    Weighted ensemble based on disagreement magnitude.
    
    Rule:
    - Same prediction (diff == 0) → use it
    - Small disagreement (diff == 1) → use model 1
    - Large disagreement (diff > 1) → use model 2 (if prefer_model2_for_large_diff)
    
    Args:
        pred1: Predictions from model 1 (shape: N,)
        pred2: Predictions from model 2 (shape: N,)
        class_diff: Absolute difference between predictions (shape: N,)
        prefer_model2_for_large_diff: If True, use model 2 for large disagreements
    
    Returns:
        Ensemble predictions (shape: N,)
    """
    ensemble_pred = np.zeros_like(pred1)
    
    # Same prediction → use it
    same_mask = class_diff == 0
    ensemble_pred[same_mask] = pred1[same_mask]
    
    # Small disagreement (diff == 1) → use model 1
    diff_1_mask = class_diff == 1
    ensemble_pred[diff_1_mask] = pred1[diff_1_mask]
    
    # Large disagreement (diff > 1) → use model 2 if preferred
    if prefer_model2_for_large_diff:
        diff_greater_mask = class_diff > 1
        ensemble_pred[diff_greater_mask] = pred2[diff_greater_mask]
    else:
        diff_greater_mask = class_diff > 1
        ensemble_pred[diff_greater_mask] = pred1[diff_greater_mask]
    
    return ensemble_pred


def evaluate_ensemble(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    loader: DataLoader,
    device: str,
    class_names: List[str],
    ensemble_method: str = "weighted"
) -> Dict:
    """
    Evaluate ensemble of two classification models.
    
    Args:
        model1: First model
        model2: Second model
        loader: Data loader
        device: Device to use
        class_names: List of class names
        ensemble_method: "majority" or "weighted"
    
    Returns dictionary with:
    - Ensemble metrics
    - Individual model metrics
    - Agreement statistics
    """
    model1.eval()
    model2.eval()
    
    all_pred1 = []
    all_pred2 = []
    all_ensemble_preds = []
    all_labels = []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            
            # Model 1 predictions
            outputs1 = model1(xb)
            pred1 = outputs1.argmax(1).cpu().numpy()
            
            # Model 2 predictions
            outputs2 = model2(xb)
            pred2 = outputs2.argmax(1).cpu().numpy()
            
            # Compute ensemble predictions
            class_diff = np.abs(pred1 - pred2)
            if ensemble_method == "majority":
                ensemble_pred = ensemble_predict_majority(pred1, pred2)
            else:  # weighted
                ensemble_pred = ensemble_predict_weighted(pred1, pred2, class_diff)
            
            all_pred1.extend(pred1)
            all_pred2.extend(pred2)
            all_ensemble_preds.extend(ensemble_pred)
            all_labels.extend(yb.cpu().numpy())
    
    all_pred1 = np.array(all_pred1)
    all_pred2 = np.array(all_pred2)
    all_ensemble_preds = np.array(all_ensemble_preds)
    all_labels = np.array(all_labels)
    
    # Agreement statistics
    same_agreement = (all_pred1 == all_pred2).sum()
    diff_1_count = (np.abs(all_pred1 - all_pred2) == 1).sum()
    diff_greater_count = (np.abs(all_pred1 - all_pred2) > 1).sum()
    total = len(all_labels)
    
    # Calculate metrics for each
    def calc_metrics(preds, labels, name):
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro")
        weighted_f1 = f1_score(labels, preds, average="weighted")
        per_class_recall = recall_score(labels, preds, average=None)
        per_class_precision = precision_score(labels, preds, average=None, zero_division=0)
        cm = confusion_matrix(labels, preds)
        
        # MAE
        mae_class = mean_absolute_error(labels, preds)
        bcs_values = [3.25, 3.5, 3.75, 4.0, 4.25]
        labels_bcs = np.array([bcs_values[int(l)] for l in labels])
        preds_bcs = np.array([bcs_values[int(p)] for p in preds])
        mae_bcs = mean_absolute_error(labels_bcs, preds_bcs)
        
        # Ordinal metrics
        ordinal_acc_1 = (np.abs(labels - preds) <= 1).mean()
        ordinal_acc_025 = (np.abs(labels_bcs - preds_bcs) <= 0.25).mean()
        
        return {
            "name": name,
            "acc": float(acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "underweight_recall": float(per_class_recall[0]),
            "per_class_recall": per_class_recall.tolist(),
            "per_class_precision": per_class_precision.tolist(),
            "confusion_matrix": cm.tolist(),
            "mae_class": float(mae_class),
            "mae_bcs": float(mae_bcs),
            "ordinal_accuracy_1": float(ordinal_acc_1),
            "ordinal_accuracy_025": float(ordinal_acc_025)
        }
    
    model1_metrics = calc_metrics(all_pred1, all_labels, "Model1")
    model2_metrics = calc_metrics(all_pred2, all_labels, "Model2")
    ensemble_metrics = calc_metrics(all_ensemble_preds, all_labels, "Ensemble")
    
    # Agreement breakdown
    agreement_stats = {
        "total_samples": int(total),
        "same_predictions": int(same_agreement),
        "same_predictions_pct": float(same_agreement / total),
        "differ_by_1": int(diff_1_count),
        "differ_by_1_pct": float(diff_1_count / total),
        "differ_by_more_than_1": int(diff_greater_count),
        "differ_by_more_than_1_pct": float(diff_greater_count / total)
    }
    
    return {
        "model1": model1_metrics,
        "model2": model2_metrics,
        "ensemble": ensemble_metrics,
        "agreement_stats": agreement_stats,
        "class_names": class_names
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ensemble of two classification models"
    )
    parser.add_argument("--model1-checkpoint", type=str, required=True,
                        help="Path to first model checkpoint")
    parser.add_argument("--model1-config", type=str, required=True,
                        help="Path to first model config YAML")
    parser.add_argument("--model2-checkpoint", type=str, required=True,
                        help="Path to second model checkpoint")
    parser.add_argument("--model2-config", type=str, required=True,
                        help="Path to second model config YAML")
    parser.add_argument("--ensemble-method", type=str, default="weighted",
                        choices=["majority", "weighted"],
                        help="Ensemble method: majority vote or weighted (default: weighted)")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate on (default: val)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save results (default: auto-generated)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cpu/cuda/mps, default: auto-detect)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("ENSEMBLE EVALUATION (General)")
    print("=" * 70)
    print(f"Model 1:  {args.model1_checkpoint}")
    print(f"Model 2:  {args.model2_checkpoint}")
    print(f"Split:    {args.split}")
    print(f"Method:   {args.ensemble_method}")
    print("=" * 70)
    
    # Get device
    if args.device:
        device = args.device
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    print(f"\nDevice: {device}")
    
    # Load models
    print("\nLoading model 1...")
    model1, cfg1 = load_model(args.model1_checkpoint, args.model1_config, device)
    
    print("Loading model 2...")
    model2, cfg2 = load_model(args.model2_checkpoint, args.model2_config, device)
    
    # Get dataset path (use model1 config)
    data_cfg = cfg1.get("data", {})
    if args.split == "train":
        csv_path = data_cfg.get("train", "data/train.csv")
    elif args.split == "val":
        csv_path = data_cfg.get("val", "data/val.csv")
    else:  # test
        csv_path = "data/test.csv"
    
    # Get image size and class names (use model1 config)
    img_size = int(data_cfg.get("img_size", 224))
    eval_cfg = cfg1.get("eval", {})
    class_names = eval_cfg.get("class_names", ["3.25", "3.5", "3.75", "4.0", "4.25"])
    
    # Load dataset
    print(f"\nLoading {args.split} dataset: {csv_path}")
    dataset = BcsDataset(csv_path, img_size=img_size, train=False, do_aug=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    print(f"Samples: {len(dataset)}")
    
    # Evaluate ensemble
    print("\n" + "=" * 70)
    print("EVALUATING ENSEMBLE")
    print("=" * 70)
    results = evaluate_ensemble(
        model1,
        model2,
        loader,
        device,
        class_names,
        args.ensemble_method
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print("\nAgreement Statistics:")
    stats = results["agreement_stats"]
    print(f"  Total samples:                {stats['total_samples']}")
    print(f"  Same predictions:             {stats['same_predictions']} ({stats['same_predictions_pct']*100:.2f}%)")
    print(f"  Differ by ±1 class:           {stats['differ_by_1']} ({stats['differ_by_1_pct']*100:.2f}%)")
    print(f"  Differ by >1 class:           {stats['differ_by_more_than_1']} ({stats['differ_by_more_than_1_pct']*100:.2f}%)")
    
    print("\n" + "-" * 70)
    print(f"{'Model':<20} {'Accuracy':<12} {'Macro-F1':<12} {'UW Recall':<12} {'MAE (BCS)':<12}")
    print("-" * 70)
    
    for model_name in ["model1", "model2", "ensemble"]:
        m = results[model_name]
        print(f"{model_name:<20} {m['acc']:>11.4f}  {m['macro_f1']:>11.4f}  "
              f"{m['underweight_recall']:>11.4f}  {m['mae_bcs']:>11.4f}")
    
    print("\n" + "-" * 70)
    print("Improvement over best individual model:")
    best_individual = max(results["model1"]["acc"], results["model2"]["acc"])
    ens_acc = results["ensemble"]["acc"]
    best_individual_f1 = max(results["model1"]["macro_f1"], results["model2"]["macro_f1"])
    ens_f1 = results["ensemble"]["macro_f1"]
    best_individual_uw = max(results["model1"]["underweight_recall"], results["model2"]["underweight_recall"])
    ens_uw = results["ensemble"]["underweight_recall"]
    
    print(f"  Accuracy:       {ens_acc - best_individual:+.4f} ({((ens_acc/best_individual - 1)*100):+.2f}%)")
    print(f"  Macro-F1:       {ens_f1 - best_individual_f1:+.4f} ({((ens_f1/best_individual_f1 - 1)*100):+.2f}%)")
    print(f"  Underweight R:  {ens_uw - best_individual_uw:+.4f} ({((ens_uw/best_individual_uw - 1)*100):+.2f}%)")
    
    # Save results
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Create output directory based on experiment names
        model1_name = os.path.basename(os.path.dirname(args.model1_checkpoint))
        model2_name = os.path.basename(os.path.dirname(args.model2_checkpoint))
        output_dir = f"outputs/ensemble_{model1_name}_{model2_name}_{args.ensemble_method}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f"ensemble_metrics_{args.split}.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to: {metrics_path}")
    
    # Save confusion matrices
    for model_name in ["model1", "model2", "ensemble"]:
        cm_path = os.path.join(output_dir, f"confusion_matrix_{model_name}_{args.split}.png")
        plot_confusion_matrix(
            np.array(results[model_name]["confusion_matrix"]),
            class_names,
            save_path=cm_path,
            title=f"{model_name.capitalize()} Model ({args.split})"
        )
        print(f"Saved {model_name} confusion matrix to: {cm_path}")
    
    print(f"\nAll results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

