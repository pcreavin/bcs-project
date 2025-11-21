#!/usr/bin/env python3
"""ROI Robustness Check: Evaluate model on ROI crops vs full images.

This script tests whether the model relies on background context by comparing
performance on tight ROI crops vs full images with context.
"""
import argparse
import os
import json
import yaml
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from src.models import create_model
from src.eval import evaluate, plot_confusion_matrix


class BcsDatasetRobustness:
    """Dataset that can load either ROI crops or full images."""
    def __init__(self, csv_path, img_size=224, use_full_image=False, do_aug=False):
        """
        Args:
            csv_path: Path to CSV with image paths and labels
            img_size: Target image size
            use_full_image: If True, use full image; if False, crop ROI
            do_aug: Whether to apply augmentation
        """
        self.df = pd.read_csv(csv_path)
        self.img_size = img_size
        self.use_full_image = use_full_image
        self.do_aug = do_aug
        
        tf = [A.Resize(img_size, img_size)]
        if do_aug:
            tf += [A.HorizontalFlip(p=0.5), A.Rotate(limit=10, p=0.5)]
        tf += [A.Normalize(), ToTensorV2()]
        self.tf = A.Compose(tf)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(r.image_path)[:, :, ::-1]  # BGR->RGB
        
        if img is None:
            raise ValueError(f"Could not load image: {r.image_path}")
        
        # If use_full_image is False, crop ROI (if bbox exists)
        if not self.use_full_image and pd.notna(r.get("xmin", np.nan)):
            try:
                h, w = img.shape[:2]
                xmin, ymin, xmax, ymax = map(int, [r.xmin, r.ymin, r.xmax, r.ymax])
                xmin = max(0, min(xmin, w - 1))
                xmax = max(1, min(xmax, w))
                ymin = max(0, min(ymin, h - 1))
                ymax = max(1, min(ymax, h))
                if xmax > xmin and ymax > ymin:
                    img = img[ymin:ymax, xmin:xmax]
            except Exception:
                pass  # fall back to full image if anything goes wrong
        
        x = self.tf(image=img)["image"]
        y = int(r.bcs_5class)  # 0..4
        return x, y


def main():
    parser = argparse.ArgumentParser(
        description="ROI Robustness Check: Compare ROI crops vs full images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check ROI robustness for the enhanced B0 model
  python scripts/check_roi_robustness.py \\
      --checkpoint outputs/ablation_full_enhanced/best_model.pt \\
      --config outputs/ablation_full_enhanced/config.yaml \\
      --val-csv data/val.csv
        """
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file used for training")
    parser.add_argument("--val-csv", type=str, default="data/val.csv",
                        help="Path to validation CSV file (must have original images with bboxes)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save results (default: same as checkpoint dir)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cpu/cuda/mps, default: auto-detect)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation")
    args = parser.parse_args()
    
    print("=" * 60)
    print("ROI ROBUSTNESS CHECK")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Validation CSV: {args.val_csv}")
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
    
    # Check if val CSV has bbox columns
    val_df = pd.read_csv(args.val_csv)
    has_bbox = all(col in val_df.columns for col in ["xmin", "ymin", "xmax", "ymax"])
    
    if not has_bbox:
        print("\n⚠️  WARNING: Validation CSV does not have bbox columns (xmin, ymin, xmax, ymax)")
        print("   ROI cropping will be skipped. Using full images for both tests.")
        print("   Make sure you're using the original val.csv, not the preprocessed one.")
    
    # Evaluate on ROI crops (current training setup)
    print("\n" + "=" * 60)
    print("EVALUATION 1: ROI Crops (Current Training Setup)")
    print("=" * 60)
    roi_ds = BcsDatasetRobustness(
        args.val_csv,
        img_size=img_size,
        use_full_image=False,
        do_aug=False
    )
    roi_loader = DataLoader(
        roi_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0  # Use 0 to avoid MPS issues
    )
    print(f"Validation samples: {len(roi_ds)}")
    
    print("Evaluating on ROI crops...")
    roi_results = evaluate(model, roi_loader, device, class_names=class_names)
    
    print(f"\nROI Crop Results:")
    print(f"  Accuracy:           {roi_results['acc']:.4f}")
    print(f"  Macro-F1:           {roi_results['macro_f1']:.4f}")
    print(f"  Weighted-F1:        {roi_results['weighted_f1']:.4f}")
    print(f"  Underweight Recall: {roi_results['underweight_recall']:.4f}")
    
    # Evaluate on full images (with context)
    print("\n" + "=" * 60)
    print("EVALUATION 2: Full Images (With Context)")
    print("=" * 60)
    full_ds = BcsDatasetRobustness(
        args.val_csv,
        img_size=img_size,
        use_full_image=True,
        do_aug=False
    )
    full_loader = DataLoader(
        full_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0  # Use 0 to avoid MPS issues
    )
    print(f"Validation samples: {len(full_ds)}")
    
    print("Evaluating on full images...")
    full_results = evaluate(model, full_loader, device, class_names=class_names)
    
    print(f"\nFull Image Results:")
    print(f"  Accuracy:           {full_results['acc']:.4f}")
    print(f"  Macro-F1:           {full_results['macro_f1']:.4f}")
    print(f"  Weighted-F1:        {full_results['weighted_f1']:.4f}")
    print(f"  Underweight Recall: {full_results['underweight_recall']:.4f}")
    
    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    acc_diff = full_results['acc'] - roi_results['acc']
    f1_diff = full_results['macro_f1'] - roi_results['macro_f1']
    recall_diff = full_results['underweight_recall'] - roi_results['underweight_recall']
    
    print(f"Accuracy difference (Full - ROI):     {acc_diff:+.4f}")
    print(f"Macro-F1 difference (Full - ROI):     {f1_diff:+.4f}")
    print(f"Underweight Recall diff (Full - ROI): {recall_diff:+.4f}")
    
    print("\n" + "-" * 60)
    if abs(acc_diff) < 0.01 and abs(f1_diff) < 0.01:
        print("✅ Model is ROBUST: Performance is similar on ROI vs full images")
        print("   The model is learning from the animal body, not background context.")
    elif acc_diff < -0.05 or f1_diff < -0.05:
        print("⚠️  Model may rely on BACKGROUND: Performance drops on full images")
        print("   Consider training with full images or adding crop jitter augmentation.")
    elif acc_diff > 0.05 or f1_diff > 0.05:
        print("✅ Model benefits from CONTEXT: Performance improves on full images")
        print("   Consider training with full images to leverage context cues.")
    else:
        print("📊 Model shows moderate difference between ROI and full images")
        print("   Results are within acceptable range.")
    print("=" * 60)
    
    # Save results
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Default: save in same directory as checkpoint
        output_dir = os.path.dirname(args.checkpoint)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save comparison results
    comparison = {
        "roi_crops": roi_results,
        "full_images": full_results,
        "differences": {
            "accuracy": float(acc_diff),
            "macro_f1": float(f1_diff),
            "underweight_recall": float(recall_diff)
        }
    }
    
    results_path = os.path.join(output_dir, "roi_robustness_check.json")
    with open(results_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved results to: {results_path}")
    
    # Save confusion matrices
    roi_cm_path = os.path.join(output_dir, "roi_crops_confusion_matrix.png")
    plot_confusion_matrix(
        roi_results["confusion_matrix"],
        class_names,
        save_path=roi_cm_path
    )
    print(f"Saved ROI crops confusion matrix to: {roi_cm_path}")
    
    full_cm_path = os.path.join(output_dir, "full_images_confusion_matrix.png")
    plot_confusion_matrix(
        full_results["confusion_matrix"],
        class_names,
        save_path=full_cm_path
    )
    print(f"Saved full images confusion matrix to: {full_cm_path}")
    
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()

