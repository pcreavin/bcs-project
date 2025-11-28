"""Compare different decoding methods on an ordinal model."""
import argparse
import os
import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader

from src.models.factory import create_model
from src.models.heads import OrdinalHead
from src.train.dataset import BcsDataset
from src.eval import evaluate


def main():
    parser = argparse.ArgumentParser(description="Compare ordinal decoding methods")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--val-csv", type=str, default="data/val.csv", help="Validation CSV")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file for results")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Setup device
    device_str = cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device_str == "null" or device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    
    # Create model
    model_cfg = cfg["model"]
    model = create_model(
        backbone=model_cfg["backbone"],
        num_classes=model_cfg["num_classes"],
        pretrained=False,
        finetune_mode=model_cfg.get("finetune_mode", "full"),
        head_type=model_cfg.get("head_type", "classification")
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # Create dataset
    img_size = cfg["data"]["img_size"]
    dataset = BcsDataset(args.val_csv, img_size=img_size, train=False, do_aug=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    class_names = cfg.get("eval", {}).get("class_names", ["3.25", "3.5", "3.75", "4.0", "4.25"])
    
    # Test different decoding methods
    decoding_methods = ["threshold_count", "expected_value", "max_prob"]
    results = []
    
    print("=" * 80)
    print("Comparing Ordinal Decoding Methods")
    print("=" * 80)
    print(f"Model: {args.checkpoint}")
    print(f"Validation set: {args.val_csv} ({len(dataset)} samples)")
    print()
    
    for method in decoding_methods:
        print(f"Testing decoding method: {method}")
        print("-" * 80)
        
        eval_results = evaluate(
            model=model,
            loader=loader,
            device=device,
            class_names=class_names,
            head_type="ordinal",
            ordinal_decoding_method=method
        )
        
        results.append({
            "decoding_method": method,
            "accuracy": eval_results["acc"],
            "macro_f1": eval_results["macro_f1"],
            "weighted_f1": eval_results["weighted_f1"],
            "underweight_recall": eval_results["underweight_recall"],
            "mae_class": eval_results.get("mae_class", None),
            "mae_bcs": eval_results.get("mae_bcs", None),
            "ordinal_accuracy_1": eval_results.get("ordinal_accuracy_1", None),
            "ordinal_accuracy_025": eval_results.get("ordinal_accuracy_025", None),
        })
        
        print(f"  Accuracy: {eval_results['acc']:.4f}")
        print(f"  Macro-F1: {eval_results['macro_f1']:.4f}")
        print(f"  Underweight Recall: {eval_results['underweight_recall']:.4f}")
        if "mae_class" in eval_results:
            print(f"  MAE (class): {eval_results['mae_class']:.4f}")
            print(f"  MAE (BCS): {eval_results['mae_bcs']:.4f}")
            print(f"  Ordinal Acc (±1 class): {eval_results['ordinal_accuracy_1']:.4f}")
        print()
    
    # Create comparison DataFrame
    df = pd.DataFrame(results)
    
    # Print comparison table
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))
    print()
    
    # Find best methods
    print("Best Results:")
    print(f"  Accuracy: {df.loc[df['accuracy'].idxmax(), 'decoding_method']} ({df['accuracy'].max():.4f})")
    print(f"  Macro-F1: {df.loc[df['macro_f1'].idxmax(), 'decoding_method']} ({df['macro_f1'].max():.4f})")
    print(f"  Underweight Recall: {df.loc[df['underweight_recall'].idxmax(), 'decoding_method']} ({df['underweight_recall'].max():.4f})")
    if df['mae_class'].notna().any():
        print(f"  MAE (class): {df.loc[df['mae_class'].idxmin(), 'decoding_method']} ({df['mae_class'].min():.4f})")
    
    # Save results
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")
    else:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        output_path = os.path.join(checkpoint_dir, "decoding_comparison.csv")
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

