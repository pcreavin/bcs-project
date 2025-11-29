#!/usr/bin/env python3
"""Compare different decoding methods for ordinal regression models."""
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

from src.train.dataset import BcsDataset
from src.models import create_model
from src.models.ordinal_utils import decode_ordinal
from src.eval.metrics import evaluate as evaluate_standard, plot_confusion_matrix


def evaluate_ordinal(model, loader, device, decoding_method: str, class_names: List[str]) -> Dict:
    """
    Evaluate ordinal model with specified decoding method.
    
    Args:
        model: Ordinal regression model (outputs threshold logits)
        loader: DataLoader for evaluation
        device: Device to run evaluation on
        decoding_method: Method to decode ordinal logits
        class_names: List of class names
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)  # Shape: (batch_size, num_thresholds)
            
            # Decode ordinal logits to class predictions
            pred = decode_ordinal(outputs, method=decoding_method)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate standard metrics
    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix,
        recall_score, precision_score, mean_absolute_error
    )
    
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    per_class_recall = recall_score(all_labels, all_preds, average=None)
    per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Ordinal-specific metrics
    mae_class = mean_absolute_error(all_labels, all_preds)
    
    # Convert class indices to BCS values for MAE
    bcs_values = [3.25, 3.5, 3.75, 4.0, 4.25]
    all_labels_bcs = np.array([bcs_values[int(l)] for l in all_labels])
    all_preds_bcs = np.array([bcs_values[int(p)] for p in all_preds])
    mae_bcs = mean_absolute_error(all_labels_bcs, all_preds_bcs)
    
    # Ordinal accuracy (±1 class)
    ordinal_acc_1 = (np.abs(all_labels - all_preds) <= 1).mean()
    
    # Ordinal accuracy (±0.25 BCS)
    ordinal_acc_025 = (np.abs(all_labels_bcs - all_preds_bcs) <= 0.25).mean()
    
    underweight_recall = per_class_recall[0] if len(per_class_recall) > 0 else 0.0
    
    results = {
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "underweight_recall": float(underweight_recall),
        "per_class_recall": per_class_recall.tolist(),
        "per_class_precision": per_class_precision.tolist(),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "mae_class": float(mae_class),
        "mae_bcs": float(mae_bcs),
        "ordinal_accuracy_1": float(ordinal_acc_1),
        "ordinal_accuracy_025": float(ordinal_acc_025),
        "decoding_method": decoding_method
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare different decoding methods for ordinal regression")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to ordinal model checkpoint (.pt file)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file")
    parser.add_argument("--method", type=str, default=None,
                        choices=["threshold_count", "expected_value", "max_prob", "all"],
                        help="Decoding method to test (default: all)")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate on (default: val)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save results (default: checkpoint directory)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cpu/cuda/mps, default: auto-detect)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("ORDINAL DECODING METHOD COMPARISON")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Split: {args.split}")
    print("=" * 70)
    
    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    eval_cfg = cfg.get("eval", {})
    
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
    
    # Load model config
    backbone = model_cfg.get("backbone", "efficientnet_b0")
    num_classes = int(model_cfg.get("num_classes", 5))
    pretrained = bool(model_cfg.get("pretrained", True))
    finetune_mode = model_cfg.get("finetune_mode", "full")
    head_type = model_cfg.get("head_type", "classification")
    
    if head_type != "ordinal":
        print(f"\nWARNING: Model head_type is '{head_type}', not 'ordinal'.")
        print("This script is designed for ordinal regression models.")
    
    img_size = int(data_cfg.get("img_size", 224))
    class_names = eval_cfg.get("class_names", ["3.25", "3.5", "3.75", "4.0", "4.25"])
    
    # Get dataset path
    if args.split == "train":
        csv_path = data_cfg.get("train", "data/train.csv")
    elif args.split == "val":
        csv_path = data_cfg.get("val", "data/val.csv")
    else:  # test
        csv_path = "data/test.csv"
    
    # Create model - need to handle ordinal head creation
    # Use robust loading logic that matches checkpoint structure exactly
    print(f"\nCreating model: {backbone}, head_type={head_type}")
    
    # Load checkpoint first to understand its structure
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_keys = list(checkpoint.keys())
    
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
    
    # Create the full model structure exactly as it was during training
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
        
        if checkpoint_output_size != num_thresholds:
            print(f"⚠️  WARNING: Checkpoint output size ({checkpoint_output_size}) != expected ({num_thresholds})")
        
        # Check if classifier has nested structure (e.g., classifier.fc)
        has_nested_classifier = "." in classifier_weight_key.replace("classifier.", "", 1)
        classifier_submodule = classifier_weight_key.split(".")[1] if has_nested_classifier else None
        
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
        else:
            ordinal_head = ordinal_linear
        
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
    else:
        # Fallback: try standard approach
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
        
        if missing_keys and len(missing_keys) > len(checkpoint_keys) * 0.1:
            print(f"⚠️  WARNING: {len(missing_keys)} keys missing (>{len(checkpoint_keys)*0.1:.0f}%) - model structure may not match checkpoint")
        
        # Verify output shape
        model.eval()
        with torch.no_grad():
            test_output = model(dummy_input.to(device))
            actual_shape = test_output.shape
            expected_shape = (1, num_thresholds)
            
            if actual_shape[1] != num_thresholds:
                raise ValueError(
                    f"Model output shape {actual_shape} doesn't match expected {expected_shape}. "
                    f"Model structure may not match checkpoint!"
                )
        
    except RuntimeError as e:
        print(f"❌ Error loading checkpoint: {e}")
        raise
    
    model.eval()
    
    # Load dataset
    print(f"\nLoading {args.split} dataset: {csv_path}")
    dataset = BcsDataset(csv_path, img_size=img_size, train=False, do_aug=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    print(f"Samples: {len(dataset)}")
    
    # Determine methods to test
    methods_to_test = ["threshold_count", "expected_value", "max_prob"] if args.method in [None, "all"] else [args.method]
    
    # Evaluate each method
    all_results = {}
    
    print("\n" + "=" * 70)
    print(f"EVALUATING {len(methods_to_test)} DECODING METHOD(S)")
    print("=" * 70)
    
    for method in methods_to_test:
        print(f"\n[{method}] Evaluating...")
        results = evaluate_ordinal(model, loader, device, method, class_names)
        all_results[method] = results
        
        # Print results
        print(f"\n  Accuracy:           {results['acc']:.4f}")
        print(f"  Macro-F1:           {results['macro_f1']:.4f}")
        print(f"  Underweight Recall: {results['underweight_recall']:.4f}")
        print(f"  MAE (class):        {results['mae_class']:.4f}")
        print(f"  MAE (BCS):          {results['mae_bcs']:.4f}")
        print(f"  Ordinal Acc (±1):   {results['ordinal_accuracy_1']:.4f}")
        print(f"  Ordinal Acc (±0.25): {results['ordinal_accuracy_025']:.4f}")
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Method':<20} {'Accuracy':<12} {'Macro-F1':<12} {'UW Recall':<12} {'MAE (BCS)':<12}")
    print("-" * 70)
    for method, results in all_results.items():
        print(f"{method:<20} {results['acc']:>11.4f}  {results['macro_f1']:>11.4f}  "
              f"{results['underweight_recall']:>11.4f}  {results['mae_bcs']:>11.4f}")
    
    # Save results
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(args.checkpoint)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save comparison results
    comparison_path = os.path.join(output_dir, f"decoding_comparison_{args.split}.json")
    with open(comparison_path, "w") as f:
        json.dump({
            "split": args.split,
            "checkpoint": args.checkpoint,
            "results": all_results
        }, f, indent=2)
    print(f"\nSaved comparison results to: {comparison_path}")
    
    # Save confusion matrices for each method
    for method, results in all_results.items():
        cm_path = os.path.join(output_dir, f"confusion_matrix_{method}_{args.split}.png")
        plot_confusion_matrix(
            results["confusion_matrix"],
            class_names,
            save_path=cm_path,
            title=f"Decoding: {method}"
        )
        print(f"Saved confusion matrix to: {cm_path}")
    
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()

