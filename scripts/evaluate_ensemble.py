#!/usr/bin/env python3
"""Evaluate ensemble of classification baseline + ordinal regression model.

Ensemble rule:
- If both models predict the same class → use it
- If they differ by ±1 class → use baseline (it's better at 0-1 loss)
- If they differ by >1 class → use ordinal prediction (since it's more distance-aware)

This script evaluates the ensemble on the test set and reports metrics.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import os
import json
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    recall_score, precision_score, mean_absolute_error
)

from src.train.dataset import BcsDataset
from src.models import create_model
from src.models.ordinal_utils import decode_ordinal
from src.eval.metrics import plot_confusion_matrix


def load_classification_model(checkpoint_path: str, config_path: str, device: str):
    """Load a classification model from checkpoint."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    
    backbone = model_cfg.get("backbone", "efficientnet_b0")
    num_classes = int(model_cfg.get("num_classes", 5))
    pretrained = bool(model_cfg.get("pretrained", True))
    finetune_mode = model_cfg.get("finetune_mode", "full")
    
    # Create model
    model = create_model(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        finetune_mode=finetune_mode
    )
    model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    return model, cfg


def load_ordinal_model(checkpoint_path: str, config_path: str, device: str, decoding_method: str = "threshold_count"):
    """Load an ordinal regression model from checkpoint.
    
    The model should have been created with an OrdinalHead replacing the classifier.
    This function reconstructs the exact structure to match the saved checkpoint.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    
    backbone = model_cfg.get("backbone", "efficientnet_b0")
    num_classes = int(model_cfg.get("num_classes", 5))
    pretrained = bool(model_cfg.get("pretrained", True))
    img_size = int(data_cfg.get("img_size", 224))
    
    # Load checkpoint first to understand its structure
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_keys = list(checkpoint.keys())
    
    print(f"\n[DEBUG] Loading ordinal model from: {checkpoint_path}")
    print(f"[DEBUG] Checkpoint has {len(checkpoint_keys)} keys")
    
    # Inspect classifier keys to understand structure
    classifier_keys = [k for k in checkpoint_keys if "classifier" in k.lower()]
    print(f"[DEBUG] Classifier-related keys: {len(classifier_keys)}")
    if classifier_keys:
        for key in classifier_keys[:3]:
            shape = checkpoint[key].shape if hasattr(checkpoint[key], 'shape') else 'N/A'
            print(f"[DEBUG]   {key}: {shape}")
    
    import timm
    
    # Create backbone first to get feature dimension
    backbone_model = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
    
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, img_size, img_size)
        dummy_features = backbone_model(dummy_input)
        if isinstance(dummy_features, tuple):
            in_features = dummy_features[0].shape[1]
        else:
            in_features = dummy_features.shape[1]
    
    num_thresholds = num_classes - 1
    print(f"[DEBUG] Feature dimension: {in_features}, Expected thresholds: {num_thresholds}")
    
    # Create the full model structure exactly as it was during training
    # Strategy: Create model structure that matches the checkpoint keys
    
    # First, check what the checkpoint classifier looks like
    classifier_weight_key = None
    classifier_bias_key = None
    for key in checkpoint_keys:
        if "classifier" in key.lower() and "weight" in key:
            classifier_weight_key = key
        if "classifier" in key.lower() and "bias" in key:
            classifier_bias_key = key
    
    if classifier_weight_key and classifier_bias_key:
        # Check the shape of the classifier in checkpoint
        checkpoint_classifier_weight = checkpoint[classifier_weight_key]
        checkpoint_output_size = checkpoint_classifier_weight.shape[0]
        checkpoint_input_size = checkpoint_classifier_weight.shape[1] if len(checkpoint_classifier_weight.shape) > 1 else None
        
        print(f"[DEBUG] Checkpoint classifier: {checkpoint_classifier_weight.shape}")
        print(f"[DEBUG] Expected thresholds: {num_thresholds}")
        
        if checkpoint_output_size != num_thresholds:
            print(f"[DEBUG] ⚠️  WARNING: Checkpoint output size ({checkpoint_output_size}) != expected ({num_thresholds})")
        
        # Check if classifier has nested structure (e.g., classifier.fc)
        has_nested_classifier = "." in classifier_weight_key.replace("classifier.", "", 1)
        classifier_submodule = classifier_weight_key.split(".")[1] if has_nested_classifier else None
        
        print(f"[DEBUG] Classifier structure: nested={has_nested_classifier}, submodule={classifier_submodule}")
        
        # Create model matching the checkpoint structure
        # Use pretrained=False since we'll load all weights from checkpoint
        model = timm.create_model(backbone, pretrained=False, num_classes=0)  # No classifier initially
        
        # Create ordinal head with correct structure
        ordinal_linear = torch.nn.Linear(checkpoint_input_size or in_features, checkpoint_output_size)
        
        # If checkpoint has nested structure (classifier.fc), create wrapper module
        if has_nested_classifier and classifier_submodule:
            class OrdinalHeadWrapper(torch.nn.Module):
                def __init__(self, fc_layer):
                    super().__init__()
                    self.fc = fc_layer
                def forward(self, x):
                    return self.fc(x)
            ordinal_head = OrdinalHeadWrapper(ordinal_linear)
            print(f"[DEBUG] Created nested classifier structure: classifier.{classifier_submodule}")
        else:
            ordinal_head = ordinal_linear
            print(f"[DEBUG] Created direct classifier structure")
        
        # Add the ordinal head to model
        if "efficientnet" in backbone.lower() or hasattr(model, "classifier"):
            model.classifier = ordinal_head
        elif hasattr(model, "head"):
            model.head = ordinal_head
        elif hasattr(model, "fc"):
            model.fc = ordinal_head
        else:
            # Fallback: Sequential wrapper
            model = torch.nn.Sequential(model, ordinal_head)
        
        print(f"[DEBUG] Created model structure matching checkpoint (output size: {checkpoint_output_size})")
    else:
        # Fallback: try standard approach
        print(f"[DEBUG] Could not find classifier keys, using fallback structure")
        model = timm.create_model(backbone, pretrained=False, num_classes=num_classes)
        ordinal_head = torch.nn.Linear(in_features, num_thresholds)
        
        if hasattr(model, "classifier"):
            model.classifier = ordinal_head
        elif hasattr(model, "head"):
            model.head = ordinal_head
        elif hasattr(model, "fc"):
            model.fc = ordinal_head
        else:
            raise ValueError(f"Model doesn't have classifier/head/fc. Available: {[x for x in dir(model) if not x.startswith('_')][:10]}")
    
    model.to(device)
    
    # Load checkpoint - this should load all backbone weights + classifier weights
    try:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        
        if missing_keys:
            print(f"[DEBUG] ⚠️  {len(missing_keys)} keys missing (will use random/default values)")
            print(f"[DEBUG] First 3 missing: {missing_keys[:3]}")
        
        if unexpected_keys:
            print(f"[DEBUG] ⚠️  {len(unexpected_keys)} unexpected keys (ignored)")
            print(f"[DEBUG] First 3 unexpected: {unexpected_keys[:3]}")
        
        if not missing_keys and not unexpected_keys:
            print(f"[DEBUG] ✓ Perfect match! All keys loaded successfully")
        elif len(missing_keys) > len(checkpoint_keys) * 0.1:  # More than 10% missing
            print(f"[DEBUG] ⚠️  WARNING: Many keys missing ({len(missing_keys)}/{len(checkpoint_keys)})")
            print(f"[DEBUG] This suggests the model structure doesn't match!")
        
        # Verify output shape
        model.eval()
        with torch.no_grad():
            test_output = model(dummy_input.to(device))
            actual_shape = test_output.shape
            expected_shape = (1, num_thresholds)
            print(f"[DEBUG] Model output shape: {actual_shape}, Expected: {expected_shape}")
            
            if actual_shape[1] != num_thresholds:
                raise ValueError(
                    f"Model output shape {actual_shape} doesn't match expected {expected_shape}. "
                    f"Model structure may not match checkpoint!"
                )
            print(f"[DEBUG] ✓ Output shape verified")
        
    except RuntimeError as e:
        print(f"[DEBUG] ❌ Error loading checkpoint: {e}")
        raise
    
    return model, cfg, decoding_method


def ensemble_predict(
    classification_pred: np.ndarray,
    ordinal_pred: np.ndarray,
    class_diff: np.ndarray
) -> np.ndarray:
    """
    Apply ensemble rule to combine predictions.
    
    Rule:
    - Same class (diff == 0) → use it
    - Differ by ±1 class (diff == 1) → use baseline
    - Differ by >1 class (diff > 1) → use ordinal
    
    Args:
        classification_pred: Predictions from classification model (shape: N,)
        ordinal_pred: Predictions from ordinal model (shape: N,)
        class_diff: Absolute difference between predictions (shape: N,)
    
    Returns:
        Ensemble predictions (shape: N,)
    """
    ensemble_pred = np.zeros_like(classification_pred)
    
    # Same class → use it
    same_mask = class_diff == 0
    ensemble_pred[same_mask] = classification_pred[same_mask]
    
    # Differ by ±1 → use baseline
    diff_1_mask = class_diff == 1
    ensemble_pred[diff_1_mask] = classification_pred[diff_1_mask]
    
    # Differ by >1 → use ordinal
    diff_greater_mask = class_diff > 1
    ensemble_pred[diff_greater_mask] = ordinal_pred[diff_greater_mask]
    
    return ensemble_pred


def evaluate_ensemble(
    classification_model: torch.nn.Module,
    ordinal_model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    ordinal_decoding_method: str,
    class_names: List[str]
) -> Dict:
    """
    Evaluate ensemble of classification and ordinal models.
    
    Returns dictionary with:
    - Ensemble metrics
    - Individual model metrics
    - Agreement statistics
    """
    classification_model.eval()
    ordinal_model.eval()
    
    all_cls_preds = []
    all_ord_preds = []
    all_ensemble_preds = []
    all_labels = []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            
            # Classification predictions
            cls_outputs = classification_model(xb)
            cls_pred = cls_outputs.argmax(1).cpu().numpy()
            
            # Ordinal predictions
            ord_outputs = ordinal_model(xb)
            ord_pred = decode_ordinal(ord_outputs, method=ordinal_decoding_method).cpu().numpy()
            
            # Compute ensemble predictions
            class_diff = np.abs(cls_pred - ord_pred)
            ensemble_pred = ensemble_predict(cls_pred, ord_pred, class_diff)
            
            all_cls_preds.extend(cls_pred)
            all_ord_preds.extend(ord_pred)
            all_ensemble_preds.extend(ensemble_pred)
            all_labels.extend(yb.cpu().numpy())
    
    all_cls_preds = np.array(all_cls_preds)
    all_ord_preds = np.array(all_ord_preds)
    all_ensemble_preds = np.array(all_ensemble_preds)
    all_labels = np.array(all_labels)
    
    # Agreement statistics
    same_agreement = (all_cls_preds == all_ord_preds).sum()
    diff_1_count = (np.abs(all_cls_preds - all_ord_preds) == 1).sum()
    diff_greater_count = (np.abs(all_cls_preds - all_ord_preds) > 1).sum()
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
    
    cls_metrics = calc_metrics(all_cls_preds, all_labels, "Classification")
    ord_metrics = calc_metrics(all_ord_preds, all_labels, "Ordinal")
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
        "classification": cls_metrics,
        "ordinal": ord_metrics,
        "ensemble": ensemble_metrics,
        "agreement_stats": agreement_stats,
        "class_names": class_names
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ensemble of classification + ordinal models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate ensemble on test set
  python scripts/evaluate_ensemble.py \\
      --baseline-checkpoint outputs/ablation_full_enhanced_b0_224_jitter/best_model.pt \\
      --baseline-config outputs/ablation_full_enhanced_b0_224_jitter/config.yaml \\
      --ordinal-checkpoint outputs/ordinal_b0_224_jitter_weighted/best_model.pt \\
      --ordinal-config outputs/ordinal_b0_224_jitter_weighted/config.yaml \\
      --split test
  
  # Evaluate with specific ordinal decoding method
  python scripts/evaluate_ensemble.py \\
      --baseline-checkpoint outputs/ablation_full_enhanced_b0_224_jitter/best_model.pt \\
      --baseline-config outputs/ablation_full_enhanced_b0_224_jitter/config.yaml \\
      --ordinal-checkpoint outputs/ordinal_b0_224_jitter_weighted/best_model.pt \\
      --ordinal-config outputs/ordinal_b0_224_jitter_weighted/config.yaml \\
      --ordinal-decoding expected_value \\
      --split val
        """
    )
    parser.add_argument("--baseline-checkpoint", type=str, required=True,
                        help="Path to classification baseline model checkpoint")
    parser.add_argument("--baseline-config", type=str, required=True,
                        help="Path to baseline model config YAML")
    parser.add_argument("--ordinal-checkpoint", type=str, required=True,
                        help="Path to ordinal regression model checkpoint")
    parser.add_argument("--ordinal-config", type=str, required=True,
                        help="Path to ordinal model config YAML")
    parser.add_argument("--ordinal-decoding", type=str, default="threshold_count",
                        choices=["threshold_count", "expected_value", "max_prob"],
                        help="Decoding method for ordinal model (default: threshold_count)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate on (default: test)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save results (default: creates ensemble_results/)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cpu/cuda/mps, default: auto-detect)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("ENSEMBLE EVALUATION")
    print("=" * 70)
    print(f"Baseline:  {args.baseline_checkpoint}")
    print(f"Ordinal:   {args.ordinal_checkpoint}")
    print(f"Split:     {args.split}")
    print(f"Decoding:  {args.ordinal_decoding}")
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
    print("\nLoading classification baseline model...")
    cls_model, baseline_cfg = load_classification_model(
        args.baseline_checkpoint,
        args.baseline_config,
        device
    )
    
    print("Loading ordinal regression model...")
    ord_model, ordinal_cfg, decoding_method = load_ordinal_model(
        args.ordinal_checkpoint,
        args.ordinal_config,
        device,
        args.ordinal_decoding
    )
    
    # Get dataset path
    data_cfg = baseline_cfg.get("data", {})
    if args.split == "train":
        csv_path = data_cfg.get("train", "data/train.csv")
    elif args.split == "val":
        csv_path = data_cfg.get("val", "data/val.csv")
    else:  # test
        csv_path = "data/test.csv"
    
    # Get image size and class names
    img_size = int(data_cfg.get("img_size", 224))
    eval_cfg = baseline_cfg.get("eval", {})
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
        cls_model,
        ord_model,
        loader,
        device,
        decoding_method,
        class_names
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
    
    for model_name in ["classification", "ordinal", "ensemble"]:
        m = results[model_name]
        print(f"{model_name:<20} {m['acc']:>11.4f}  {m['macro_f1']:>11.4f}  "
              f"{m['underweight_recall']:>11.4f}  {m['mae_bcs']:>11.4f}")
    
    print("\n" + "-" * 70)
    print("Improvement over baseline:")
    base_acc = results["classification"]["acc"]
    ens_acc = results["ensemble"]["acc"]
    base_f1 = results["classification"]["macro_f1"]
    ens_f1 = results["ensemble"]["macro_f1"]
    base_uw = results["classification"]["underweight_recall"]
    ens_uw = results["ensemble"]["underweight_recall"]
    
    print(f"  Accuracy:       {ens_acc - base_acc:+.4f} ({((ens_acc/base_acc - 1)*100):+.2f}%)")
    print(f"  Macro-F1:       {ens_f1 - base_f1:+.4f} ({((ens_f1/base_f1 - 1)*100):+.2f}%)")
    print(f"  Underweight R:  {ens_uw - base_uw:+.4f} ({((ens_uw/base_uw - 1)*100):+.2f}%)")
    
    # Save results
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Create output directory based on experiment names
        base_name = os.path.basename(os.path.dirname(args.baseline_checkpoint))
        ord_name = os.path.basename(os.path.dirname(args.ordinal_checkpoint))
        output_dir = f"outputs/ensemble_{base_name}_{ord_name}_{args.ordinal_decoding}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f"ensemble_metrics_{args.split}.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to: {metrics_path}")
    
    # Save confusion matrices
    for model_name in ["classification", "ordinal", "ensemble"]:
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

