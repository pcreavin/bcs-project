#!/usr/bin/env python3
"""Generate Grad-CAM visualizations for model predictions on test data."""
import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import random

from src.train.dataset import BcsDataset
from src.models import create_model
from src.eval.gradcam import visualize_gradcam


def main():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualizations for test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate Grad-CAM for a few test samples
  python scripts/generate_gradcam.py --checkpoint outputs/roi_robustness_padding_05/best_model.pt \\
                                      --config outputs/roi_robustness_padding_05/config.yaml \\
                                      --test-csv data/test.csv \\
                                      --num-samples 10
  
  # Generate for specific classes
  python scripts/generate_gradcam.py --checkpoint outputs/roi_robustness_padding_05/best_model.pt \\
                                      --config outputs/roi_robustness_padding_05/config.yaml \\
                                      --test-csv data/test.csv \\
                                      --num-samples 5 \\
                                      --class-filter 0 2 4
        """
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file used for training")
    parser.add_argument("--test-csv", type=str, default="data/test.csv",
                        help="Path to test CSV file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save Grad-CAM visualizations (default: checkpoint_dir/gradcam)")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of samples to visualize (default: 10)")
    parser.add_argument("--class-filter", type=int, nargs="+", default=None,
                        help="Only visualize samples from these classes (default: all classes)")
    parser.add_argument("--incorrect-only", action="store_true",
                        help="Only visualize incorrectly predicted samples")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cpu/cuda/mps, default: auto-detect)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("GRAD-CAM VISUALIZATION")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Test CSV: {args.test_csv}")
    print(f"Number of samples: {args.num_samples}")
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
    
    # Store original dataset reference
    original_ds = test_ds
    indices = None
    
    # If incorrect-only, first find all incorrect predictions
    if args.incorrect_only:
        print("Finding incorrectly predicted samples...")
        model.eval()
        incorrect_indices = []
        test_loader_full = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader_full):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                
                # Find incorrect predictions in this batch
                incorrect_mask = (preds != labels)
                incorrect_batch_indices = torch.where(incorrect_mask)[0]
                
                for local_idx in incorrect_batch_indices:
                    global_idx = batch_idx * 32 + local_idx.item()
                    incorrect_indices.append(global_idx)
        
        print(f"Found {len(incorrect_indices)} incorrectly predicted samples")
        if len(incorrect_indices) == 0:
            print("No incorrect predictions found! Model is perfect on test set.")
            return
        
        # Sample from incorrect predictions
        random.seed(42)
        num_samples = min(args.num_samples, len(incorrect_indices))
        indices = random.sample(incorrect_indices, num_samples)
        test_ds = Subset(test_ds, indices)
        print(f"Selected {num_samples} incorrect predictions for visualization")
    
    # Filter by class if specified
    elif args.class_filter is not None:
        filtered_indices = []
        for idx in range(len(test_ds)):
            _, label = test_ds[idx]
            if label in args.class_filter:
                filtered_indices.append(idx)
        print(f"Filtered to {len(filtered_indices)} samples from classes {args.class_filter}")
        if len(filtered_indices) == 0:
            print("No samples found for specified classes!")
            return
        indices = filtered_indices[:args.num_samples]
        test_ds = Subset(test_ds, indices)
    else:
        # Randomly sample
        random.seed(42)
        indices = random.sample(range(len(test_ds)), min(args.num_samples, len(test_ds)))
        test_ds = Subset(test_ds, indices)
    
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    
    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        if args.incorrect_only:
            output_dir = os.path.join(checkpoint_dir, "gradcam_incorrect")
        else:
            output_dir = os.path.join(checkpoint_dir, "gradcam")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving visualizations to: {output_dir}")
    
    # Generate Grad-CAM for each sample
    print("\nGenerating Grad-CAM visualizations...")
    correct_count = 0
    total_count = 0
    
    for idx, (image_tensor, label) in enumerate(test_loader):
        # Convert tensor to numpy image
        # Image is already normalized, so we need to denormalize for visualization
        img_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Denormalize (ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)
        
        # Get original image path for better visualization
        actual_idx = test_ds.indices[idx]
        original_img_path = original_ds.df.iloc[actual_idx].image_path
        
        if original_img_path and os.path.exists(original_img_path):
            # Load original image
            original_img = cv2.imread(original_img_path)
            if original_img is not None:
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                # Apply ROI crop if needed
                row = original_ds.df.iloc[actual_idx]
                if pd.notna(row.get("xmin", np.nan)):
                    try:
                        h, w = original_img.shape[:2]
                        xmin, ymin, xmax, ymax = map(int, [row.xmin, row.ymin, row.xmax, row.ymax])
                        xmin = max(0, min(xmin, w - 1))
                        xmax = max(1, min(xmax, w))
                        ymin = max(0, min(ymin, h - 1))
                        ymax = max(1, min(ymax, h))
                        if xmax > xmin and ymax > ymin:
                            original_img = original_img[ymin:ymax, xmin:xmax]
                    except Exception:
                        pass
                img_np = original_img
    
        # Generate Grad-CAM
        save_path = os.path.join(output_dir, f"gradcam_sample_{idx:03d}_class_{label.item()}.png")
        
        try:
            overlaid, pred_class, confidence = visualize_gradcam(
                model=model,
                image=img_np,
                class_names=class_names,
                device=device,
                save_path=save_path,
                target_class=None  # Use predicted class
            )
            
            is_correct = (pred_class == label.item())
            if is_correct:
                correct_count += 1
            total_count += 1
            
            true_class_name = class_names[label.item()] if label.item() < len(class_names) else str(label.item())
            pred_class_name = class_names[pred_class] if pred_class < len(class_names) else str(pred_class)
            
            status = "✓" if is_correct else "✗"
            print(f"  [{idx+1}/{len(test_loader)}] {status} True: {true_class_name}, Pred: {pred_class_name} ({confidence:.2%})")
            
        except Exception as e:
            print(f"  [{idx+1}/{len(test_loader)}] Error: {e}")
            continue
    
    print("\n" + "=" * 60)
    print(f"Generated {total_count} Grad-CAM visualizations")
    print(f"Accuracy on sampled images: {correct_count}/{total_count} ({100*correct_count/total_count:.1f}%)")
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

