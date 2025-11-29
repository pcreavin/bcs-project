#!/usr/bin/env python3
"""Evaluate ensemble of 3, 4, 5, or more classification models."""
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
from collections import Counter
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


def load_ordinal_model(checkpoint_path: str, config_path: str, device: str, decoding_method: str = "threshold_count"):
    """Load ordinal regression model from checkpoint."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    
    backbone = model_cfg.get("backbone", "efficientnet_b0")
    num_classes = int(model_cfg.get("num_classes", 5))
    pretrained = bool(model_cfg.get("pretrained", True))
    img_size = int(data_cfg.get("img_size", 224))
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_keys = list(checkpoint.keys())
    
    import timm
    
    backbone_model = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
    
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, img_size, img_size)
        dummy_features = backbone_model(dummy_input)
        if isinstance(dummy_features, tuple):
            in_features = dummy_features[0].shape[1]
        else:
            in_features = dummy_features.shape[1]
    
    num_thresholds = num_classes - 1
    classifier_weight_key = None
    classifier_bias_key = None
    for key in checkpoint_keys:
        if "classifier" in key.lower() and "weight" in key:
            classifier_weight_key = key
        if "classifier" in key.lower() and "bias" in key:
            classifier_bias_key = key
    
    if classifier_weight_key and classifier_bias_key:
        checkpoint_classifier_weight = checkpoint[classifier_weight_key]
        checkpoint_output_size = checkpoint_classifier_weight.shape[0]
        checkpoint_input_size = checkpoint_classifier_weight.shape[1] if len(checkpoint_classifier_weight.shape) > 1 else None
        
        model = timm.create_model(backbone, pretrained=False, num_classes=0)
        ordinal_linear = torch.nn.Linear(checkpoint_input_size or in_features, checkpoint_output_size)
        
        if "efficientnet" in backbone.lower() or hasattr(model, "classifier"):
            model.classifier = ordinal_linear
        elif hasattr(model, "head"):
            model.head = ordinal_linear
        elif hasattr(model, "fc"):
            model.fc = ordinal_linear
        else:
            model = torch.nn.Sequential(model, ordinal_linear)
    else:
        model = timm.create_model(backbone, pretrained=False, num_classes=num_classes)
        ordinal_head = torch.nn.Linear(in_features, num_thresholds)
        
        if hasattr(model, "classifier"):
            model.classifier = ordinal_head
        elif hasattr(model, "head"):
            model.head = ordinal_head
        elif hasattr(model, "fc"):
            model.fc = ordinal_head
        else:
            raise ValueError(f"Model doesn't have classifier/head/fc")
    
    model.to(device)
    
    try:
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
    except RuntimeError as e:
        print(f"❌ Error loading checkpoint: {e}")
        raise
    
    return model, cfg, decoding_method


def ensemble_predict_majority(predictions: List[np.ndarray]) -> np.ndarray:
    """
    Majority vote ensemble.
    
    For each sample, use the class that most models predict.
    In case of tie, use the first model's prediction.
    
    Args:
        predictions: List of prediction arrays, each shape (N,)
    
    Returns:
        Ensemble predictions (shape: N,)
    """
    num_samples = len(predictions[0])
    ensemble_pred = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        votes = [pred[i] for pred in predictions]
        # Count votes
        vote_counts = Counter(votes)
        # Get most common (majority)
        most_common = vote_counts.most_common(1)[0][0]
        ensemble_pred[i] = most_common
    
    return ensemble_pred


def ensemble_predict_weighted(
    predictions: List[np.ndarray],
    weights: List[float]
) -> np.ndarray:
    """
    Weighted vote ensemble.
    
    Each model's vote is weighted by its performance.
    For each sample, sum weighted votes and pick class with highest score.
    
    Args:
        predictions: List of prediction arrays, each shape (N,)
        weights: List of weights for each model (should sum to 1.0)
    
    Returns:
        Ensemble predictions (shape: N,)
    """
    num_samples = len(predictions[0])
    num_classes = 5  # BCS has 5 classes
    ensemble_pred = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        # Weighted vote count per class
        class_scores = np.zeros(num_classes)
        for pred, weight in zip(predictions, weights):
            class_scores[pred[i]] += weight
        # Pick class with highest weighted score
        ensemble_pred[i] = np.argmax(class_scores)
    
    return ensemble_pred


def ensemble_predict_consensus(
    predictions: List[np.ndarray],
    consensus_threshold: float = 0.6
) -> np.ndarray:
    """
    Consensus ensemble - only use prediction if enough models agree.
    
    For each sample:
    - If >= consensus_threshold models agree → use that prediction
    - Otherwise → use first model's prediction (fallback)
    
    Args:
        predictions: List of prediction arrays, each shape (N,)
        consensus_threshold: Fraction of models that must agree (0.0-1.0)
    
    Returns:
        Ensemble predictions (shape: N,)
    """
    num_samples = len(predictions[0])
    num_models = len(predictions)
    min_consensus = int(np.ceil(consensus_threshold * num_models))
    ensemble_pred = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        votes = [pred[i] for pred in predictions]
        vote_counts = Counter(votes)
        most_common, count = vote_counts.most_common(1)[0]
        
        if count >= min_consensus:
            # Consensus reached
            ensemble_pred[i] = most_common
        else:
            # No consensus, use first model
            ensemble_pred[i] = predictions[0][i]
    
    return ensemble_pred


def evaluate_ensemble(
    models: List[torch.nn.Module],
    model_types: List[str],  # "classification" or "ordinal"
    ordinal_decoding_methods: List[str],
    loader: DataLoader,
    device: str,
    class_names: List[str],
    ensemble_method: str = "majority",
    weights: List[float] = None,
    consensus_threshold: float = 0.6
) -> Dict:
    """
    Evaluate ensemble of multiple models.
    
    Args:
        models: List of models
        model_types: List of model types ("classification" or "ordinal")
        ordinal_decoding_methods: List of decoding methods for ordinal models
        loader: Data loader
        device: Device to use
        class_names: List of class names
        ensemble_method: "majority", "weighted", or "consensus"
        weights: Weights for weighted voting (must match number of models)
        consensus_threshold: Threshold for consensus method (0.0-1.0)
    
    Returns dictionary with metrics for each model and ensemble.
    """
    for model in models:
        model.eval()
    
    all_predictions = [[] for _ in models]
    all_ensemble_preds = []
    all_labels = []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            
            batch_predictions = []
            for model, model_type, ord_method in zip(models, model_types, ordinal_decoding_methods):
                if model_type == "ordinal":
                    outputs = model(xb)
                    pred = decode_ordinal(outputs, method=ord_method).cpu().numpy()
                else:
                    outputs = model(xb)
                    pred = outputs.argmax(1).cpu().numpy()
                
                batch_predictions.append(pred)
            
            # Convert to numpy arrays for easier manipulation
            batch_predictions = [np.array(p) for p in batch_predictions]
            
            # Compute ensemble predictions
            if ensemble_method == "majority":
                ensemble_pred = ensemble_predict_majority(batch_predictions)
            elif ensemble_method == "weighted":
                if weights is None:
                    weights = [1.0 / len(models)] * len(models)  # Equal weights
                ensemble_pred = ensemble_predict_weighted(batch_predictions, weights)
            elif ensemble_method == "consensus":
                ensemble_pred = ensemble_predict_consensus(batch_predictions, consensus_threshold)
            else:
                raise ValueError(f"Unknown ensemble method: {ensemble_method}")
            
            for i, pred in enumerate(batch_predictions):
                all_predictions[i].extend(pred)
            all_ensemble_preds.extend(ensemble_pred)
            all_labels.extend(yb.cpu().numpy())
    
    all_predictions = [np.array(p) for p in all_predictions]
    all_ensemble_preds = np.array(all_ensemble_preds)
    all_labels = np.array(all_labels)
    
    # Agreement statistics
    num_models = len(models)
    agreement_stats = {
        "num_models": num_models,
        "total_samples": len(all_labels),
    }
    
    # Calculate agreement between models
    same_all = 0
    for i in range(len(all_labels)):
        votes = [pred[i] for pred in all_predictions]
        if len(set(votes)) == 1:  # All agree
            same_all += 1
    
    agreement_stats["all_agree"] = int(same_all)
    agreement_stats["all_agree_pct"] = float(same_all / len(all_labels))
    
    # Calculate metrics for each model and ensemble
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
    
    # Individual model metrics
    individual_metrics = {}
    for i, (pred, model_type) in enumerate(zip(all_predictions, model_types)):
        model_name = f"model{i+1}_{model_type}"
        individual_metrics[model_name] = calc_metrics(pred, all_labels, model_name)
    
    # Ensemble metrics
    ensemble_metrics = calc_metrics(all_ensemble_preds, all_labels, "ensemble")
    
    return {
        **individual_metrics,
        "ensemble": ensemble_metrics,
        "agreement_stats": agreement_stats,
        "class_names": class_names
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ensemble of 3, 4, 5, or more models"
    )
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="Paths to model checkpoints (space-separated)")
    parser.add_argument("--configs", type=str, nargs="+", required=True,
                        help="Paths to model configs (space-separated, must match checkpoints)")
    parser.add_argument("--model-types", type=str, nargs="+", required=True,
                        choices=["classification", "ordinal"],
                        help="Model types: 'classification' or 'ordinal' (must match checkpoints)")
    parser.add_argument("--ordinal-decoding", type=str, nargs="+", default=None,
                        help="Decoding methods for ordinal models (default: threshold_count for all)")
    parser.add_argument("--ensemble-method", type=str, default="majority",
                        choices=["majority", "weighted", "consensus"],
                        help="Ensemble method (default: majority)")
    parser.add_argument("--weights", type=float, nargs="+", default=None,
                        help="Weights for weighted voting (must sum to 1.0, must match number of models)")
    parser.add_argument("--consensus-threshold", type=float, default=0.6,
                        help="Consensus threshold (0.0-1.0) for consensus method (default: 0.6)")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate on (default: val)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save results (default: auto-generated)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cpu/cuda/mps, default: auto-detect)")
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.checkpoints) != len(args.configs):
        raise ValueError(f"Number of checkpoints ({len(args.checkpoints)}) must match number of configs ({len(args.configs)})")
    if len(args.checkpoints) != len(args.model_types):
        raise ValueError(f"Number of checkpoints ({len(args.checkpoints)}) must match number of model types ({len(args.model_types)})")
    if len(args.checkpoints) < 2:
        raise ValueError("Must provide at least 2 models for ensemble")
    
    # Set default ordinal decoding methods
    if args.ordinal_decoding is None:
        args.ordinal_decoding = ["threshold_count"] * len(args.checkpoints)
    elif len(args.ordinal_decoding) != len(args.checkpoints):
        # Extend to match number of models
        args.ordinal_decoding = args.ordinal_decoding + ["threshold_count"] * (len(args.checkpoints) - len(args.ordinal_decoding))
    
    print("=" * 70)
    print("MULTI-WAY ENSEMBLE EVALUATION")
    print("=" * 70)
    print(f"Number of models: {len(args.checkpoints)}")
    for i, (ckpt, cfg, mtype) in enumerate(zip(args.checkpoints, args.configs, args.model_types)):
        print(f"  Model {i+1}: {os.path.basename(os.path.dirname(ckpt))} ({mtype})")
    print(f"Split:    {args.split}")
    print(f"Method:   {args.ensemble_method}")
    if args.weights:
        print(f"Weights:  {args.weights}")
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
    models = []
    configs = []
    model_types = []
    ordinal_decoding_methods = []
    
    for i, (checkpoint, config, model_type, ord_method) in enumerate(
        zip(args.checkpoints, args.configs, args.model_types, args.ordinal_decoding)
    ):
        print(f"\nLoading model {i+1} ({model_type})...")
        if model_type == "ordinal":
            model, cfg, decoding_method = load_ordinal_model(checkpoint, config, device, ord_method)
            models.append(model)
            configs.append(cfg)
            model_types.append("ordinal")
            ordinal_decoding_methods.append(decoding_method)
        else:
            model, cfg = load_classification_model(checkpoint, config, device)
            models.append(model)
            configs.append(cfg)
            model_types.append("classification")
            ordinal_decoding_methods.append("threshold_count")  # Not used for classification
    
    # Get dataset path (use first model's config)
    data_cfg = configs[0].get("data", {})
    if args.split == "train":
        csv_path = data_cfg.get("train", "data/train.csv")
    elif args.split == "val":
        csv_path = data_cfg.get("val", "data/val.csv")
    else:  # test
        csv_path = "data/test.csv"
    
    # Get image size and class names (use first model's config)
    img_size = int(data_cfg.get("img_size", 224))
    eval_cfg = configs[0].get("eval", {})
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
        models,
        model_types,
        ordinal_decoding_methods,
        loader,
        device,
        class_names,
        args.ensemble_method,
        args.weights,
        args.consensus_threshold
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print("\nAgreement Statistics:")
    stats = results["agreement_stats"]
    print(f"  Total samples:     {stats['total_samples']}")
    print(f"  All models agree:  {stats['all_agree']} ({stats['all_agree_pct']*100:.2f}%)")
    
    print("\n" + "-" * 70)
    print(f"{'Model':<30} {'Accuracy':<12} {'Macro-F1':<12} {'UW Recall':<12}")
    print("-" * 70)
    
    # Print individual models
    for key in sorted(results.keys()):
        if key not in ["ensemble", "agreement_stats", "class_names"]:
            m = results[key]
            print(f"{key:<30} {m['acc']:>11.4f}  {m['macro_f1']:>11.4f}  {m['underweight_recall']:>11.4f}")
    
    # Print ensemble
    m = results["ensemble"]
    print("-" * 70)
    print(f"{'ENSEMBLE':<30} {m['acc']:>11.4f}  {m['macro_f1']:>11.4f}  {m['underweight_recall']:>11.4f}")
    print("-" * 70)
    
    # Find best individual model
    best_individual_acc = max([results[k]["acc"] for k in results.keys() if k not in ["ensemble", "agreement_stats", "class_names"]])
    best_individual_f1 = max([results[k]["macro_f1"] for k in results.keys() if k not in ["ensemble", "agreement_stats", "class_names"]])
    best_individual_uw = max([results[k]["underweight_recall"] for k in results.keys() if k not in ["ensemble", "agreement_stats", "class_names"]])
    
    print("\nImprovement over best individual model:")
    print(f"  Accuracy:       {m['acc'] - best_individual_acc:+.4f} ({((m['acc']/best_individual_acc - 1)*100):+.2f}%)")
    print(f"  Macro-F1:       {m['macro_f1'] - best_individual_f1:+.4f} ({((m['macro_f1']/best_individual_f1 - 1)*100):+.2f}%)")
    print(f"  Underweight R:  {m['underweight_recall'] - best_individual_uw:+.4f} ({((m['underweight_recall']/best_individual_uw - 1)*100):+.2f}%)")
    
    # Save results
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Create output directory based on model names
        model_names = [os.path.basename(os.path.dirname(ckpt)) for ckpt in args.checkpoints]
        output_dir = f"outputs/ensemble_{len(args.checkpoints)}way_{'_'.join(model_names[:3])}"  # Limit name length
        if len(model_names) > 3:
            output_dir += f"_and{len(model_names)-3}more"
        output_dir += f"_{args.ensemble_method}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f"ensemble_metrics_{args.split}.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to: {metrics_path}")
    
    # Save confusion matrices
    for key in results.keys():
        if key not in ["agreement_stats", "class_names"]:
            cm_path = os.path.join(output_dir, f"confusion_matrix_{key}_{args.split}.png")
            plot_confusion_matrix(
                np.array(results[key]["confusion_matrix"]),
                class_names,
                save_path=cm_path,
                title=f"{key.capitalize()} ({args.split})"
            )
    
    print(f"\nAll results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

