"""Script to generate Grad-CAM visualizations for BCS classification."""
import argparse
import os
import yaml
import torch
import numpy as np
import cv2
import pandas as pd
import random

from src.models.factory import create_model
from src.train.dataset import BcsDataset
from src.eval.gradcam import plot_gradcam


def load_original_image(row, img_size: int) -> np.ndarray:
    """
    Load and preprocess original image for visualization.
    Handles ROI cropping if bounding box is present.
    
    Args:
        row: DataFrame row with image_path and optional bbox coordinates
        img_size: Target size for resizing
    
    Returns:
        Original image as numpy array (H, W, 3) in RGB
    """
    img = cv2.imread(row.image_path)[:, :, ::-1]  # BGR->RGB
    
    # ROI crop (if bbox present) - same logic as dataset
    if pd.notna(row.get("xmin", np.nan)):
        try:
            h, w = img.shape[:2]
            xmin, ymin, xmax, ymax = map(int, [row.xmin, row.ymin, row.xmax, row.ymax])
            xmin = max(0, min(xmin, w - 1))
            xmax = max(1, min(xmax, w))
            ymin = max(0, min(ymin, h - 1))
            ymax = max(1, min(ymax, h))
            if xmax > xmin and ymax > ymin:
                img = img[ymin:ymax, xmin:xmax]
        except Exception:
            pass  # fall back to full image if anything goes wrong
    
    # Resize to target size
    img = cv2.resize(img, (img_size, img_size))
    
    return img


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, help="Path to config file (optional)")
    parser.add_argument("--data-csv", type=str, required=True, help="Path to CSV with image paths")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for visualizations")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to visualize")
    parser.add_argument("--target-class", type=int, default=None, help="Specific class to visualize (None = use predicted)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset before sampling")
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load config
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        # Try to load from checkpoint directory
        checkpoint_dir = os.path.dirname(args.checkpoint)
        config_path = os.path.join(checkpoint_dir, "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
        else:
            raise ValueError("No config file provided and not found in checkpoint directory")
    
    # Setup device
    device_str = cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device_str == "null" or device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")
    
    # Create model
    model_cfg = cfg["model"]
    model = create_model(
        backbone=model_cfg["backbone"],
        num_classes=model_cfg["num_classes"],
        pretrained=False  # We're loading from checkpoint
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    model.to(device)
    model.eval()
    
    # Get class names
    class_names = cfg.get("eval", {}).get("class_names", ["3.25", "3.5", "3.75", "4.0", "4.25"])
    
    # Create dataset
    img_size = cfg["data"]["img_size"]
    dataset = BcsDataset(
        csv_path=args.data_csv,
        img_size=img_size,
        train=False,
        do_aug=False
    )
    
    # Load CSV for accessing original images
    df = pd.read_csv(args.data_csv)
    
    # Setup output directory
    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(args.checkpoint), "gradcam")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Select samples to visualize
    num_samples = min(args.num_samples, len(dataset))
    if args.shuffle:
        indices = random.sample(range(len(dataset)), num_samples)
    else:
        indices = list(range(num_samples))
    
    # Generate visualizations
    print(f"\nGenerating Grad-CAM visualizations...")
    print(f"Model backbone: {model_cfg['backbone']}")
    print(f"Number of samples: {num_samples}")
    print(f"Output directory: {output_dir}")
    if args.target_class is not None:
        print(f"Target class: {class_names[args.target_class]}")
    print()
    
    for i, idx in enumerate(indices):
        # Get data
        input_tensor, label = dataset[idx]
        input_tensor = input_tensor.unsqueeze(0).to(device)  # Add batch dimension
        
        # Load original image for visualization
        row = df.iloc[idx]
        original_img = load_original_image(row, img_size)
        
        # Generate visualization
        class_str = class_names[label].replace(".", "_")
        save_path = os.path.join(output_dir, f"gradcam_sample_{i:03d}_true_{class_str}_idx_{idx}.png")
        
        try:
            plot_gradcam(
                model=model,
                input_tensor=input_tensor,
                original_image=original_img,
                backbone=model_cfg["backbone"],
                class_names=class_names,
                true_label=label,
                save_path=save_path,
                target_class=args.target_class
            )
            print(f"  [{i+1}/{num_samples}] Saved: {os.path.basename(save_path)}")
        except Exception as e:
            print(f"  [{i+1}/{num_samples}] ERROR processing sample {idx}: {e}")
            continue
    
    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()

