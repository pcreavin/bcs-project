#!/usr/bin/env python3
"""Evaluate a trained model on the held-out test set.

This script should only be run after model selection is complete.
The test set is kept separate to ensure unbiased final evaluation.
"""
import argparse
import os
import json
import yaml
import torch
from torch.utils.data import DataLoader

from src.train.dataset import BcsDataset
from src.models import create_model
from src.eval import evaluate, plot_confusion_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a specific model checkpoint
  python scripts/evaluate_test.py --checkpoint outputs/ablation_full/best_model.pt \\
                                   --config outputs/ablation_full/config.yaml
  
  # Evaluate and save results
  python scripts/evaluate_test.py --checkpoint outputs/ablation_full/best_model.pt \\
                                   --config outputs/ablation_full/config.yaml \\
                                   --output-dir outputs/ablation_full/test_results
        """
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file used for training")
    parser.add_argument("--test-csv", type=str, default="data/test.csv",
                        help="Path to test CSV file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save test results (default: same as checkpoint dir)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cpu/cuda/mps, default: auto-detect)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Test CSV: {args.test_csv}")
    print("=" * 60)
    
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
    
    img_size = int(data_cfg.get("img_size", 224))
    class_names = eval_cfg.get("class_names", ["3.25", "3.5", "3.75", "4.0", "4.25"])
    
    # Create model
    print(f"\nCreating model: {backbone}, finetune_mode={finetune_mode}")
    model = create_model(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        finetune_mode=finetune_mode
    )
    model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    # Load test dataset
    print(f"\nLoading test dataset: {args.test_csv}")
    test_ds = BcsDataset(args.test_csv, img_size=img_size, train=False, do_aug=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )
    print(f"Test samples: {len(test_ds)}")
    
    # Evaluate
    print("\nEvaluating on test set...")
    test_results = evaluate(model, test_loader, device, class_names=class_names)
    
    # Print results
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Accuracy:           {test_results['acc']:.4f}")
    print(f"Macro-F1:           {test_results['macro_f1']:.4f}")
    print(f"Weighted-F1:        {test_results['weighted_f1']:.4f}")
    print(f"Underweight Recall: {test_results['underweight_recall']:.4f}")
    print("\nPer-class Recall:")
    for name, recall in zip(class_names, test_results['per_class_recall']):
        print(f"  {name}: {recall:.4f}")
    print("\nPer-class Precision:")
    for name, precision in zip(class_names, test_results['per_class_precision']):
        print(f"  {name}: {precision:.4f}")
    print("=" * 60)
    
    # Save results
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Default: save in same directory as checkpoint
        output_dir = os.path.dirname(args.checkpoint)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\nSaved test metrics to: {metrics_path}")
    
    # Save confusion matrix
    cm_path = os.path.join(output_dir, "test_confusion_matrix.png")
    plot_confusion_matrix(
        test_results["confusion_matrix"],
        class_names,
        save_path=cm_path
    )
    
    print(f"Saved confusion matrix to: {cm_path}")
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()

