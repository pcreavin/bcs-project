#!/usr/bin/env python3
"""Visualize different ROI cropping methods: ROI-only, ROI-jitter, ROI-5% padding, ROI-10% padding."""
import argparse
import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def apply_roi_crop(img, xmin, ymin, xmax, ymax, h, w, crop_jitter=None, crop_padding=None):
    """
    Apply ROI cropping with optional jitter or padding.
    
    Args:
        img: Input image (H, W, 3)
        xmin, ymin, xmax, ymax: Bounding box coordinates
        h, w: Image height and width
        crop_jitter: If not None, apply random jitter (e.g., 0.1 = 10% random padding)
        crop_padding: If not None, apply fixed padding (e.g., 0.05 = 5% padding on each side)
    
    Returns:
        Cropped image
    """
    # Clamp bbox to image bounds
    xmin = max(0, min(int(xmin), w - 1))
    xmax = max(1, min(int(xmax), w))
    ymin = max(0, min(int(ymin), h - 1))
    ymax = max(1, min(int(ymax), h))
    
    if xmax <= xmin or ymax <= ymin:
        return img  # Invalid bbox, return full image
    
    # Calculate bbox dimensions
    bbox_w = xmax - xmin
    bbox_h = ymax - ymin
    
    if crop_padding is not None:
        # Fixed padding: expand bbox by padding% on each side
        pad_w = int(bbox_w * crop_padding)
        pad_h = int(bbox_h * crop_padding)
        xmin = max(0, xmin - pad_w)
        ymin = max(0, ymin - pad_h)
        xmax = min(w, xmax + pad_w)
        ymax = min(h, ymax + pad_h)
    elif crop_jitter is not None:
        # Random jitter: add random padding up to jitter% on each side
        np.random.seed(42)  # For reproducibility in visualization
        pad_w = int(bbox_w * crop_jitter * np.random.random())
        pad_h = int(bbox_h * crop_jitter * np.random.random())
        xmin = max(0, xmin - pad_w)
        ymin = max(0, ymin - pad_h)
        xmax = min(w, xmax + pad_w)
        ymax = min(h, ymax + pad_h)
    
    # Crop image
    cropped = img[ymin:ymax, xmin:xmax]
    return cropped


def create_roi_comparison_grid(csv_path, output_path, img_size=224):
    """
    Create a 2x2 grid showing ROI-only, ROI-jitter, ROI-5% padding, ROI-10% padding.
    """
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Find a sample with a valid bounding box
    sample = None
    for idx, row in df.iterrows():
        if pd.notna(row.get("xmin", np.nan)):
            try:
                img_path = row.image_path
                if os.path.exists(img_path):
                    sample = row
                    break
            except Exception:
                continue
    
    if sample is None:
        print("Error: No sample with valid bounding box found!")
        return
    
    # Load image
    img = cv2.imread(sample.image_path)
    if img is None:
        print(f"Error: Could not load image {sample.image_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Get bbox coordinates
    xmin, ymin, xmax, ymax = map(float, [sample.xmin, sample.ymin, sample.xmax, sample.ymax])
    
    # Apply each cropping method
    crops = {}
    
    # 1. ROI-only (no jitter, no padding)
    crops["ROI-only"] = apply_roi_crop(img.copy(), xmin, ymin, xmax, ymax, h, w)
    
    # 2. ROI-jitter (10% random padding)
    np.random.seed(42)  # For reproducibility
    crops["ROI-jitter (10%)"] = apply_roi_crop(img.copy(), xmin, ymin, xmax, ymax, h, w, crop_jitter=0.1)
    
    # 3. ROI-5% padding
    crops["ROI-5% padding"] = apply_roi_crop(img.copy(), xmin, ymin, xmax, ymax, h, w, crop_padding=0.05)
    
    # 4. ROI-10% padding
    crops["ROI-10% padding"] = apply_roi_crop(img.copy(), xmin, ymin, xmax, ymax, h, w, crop_padding=0.10)
    
    # Resize all crops to img_size for consistent display
    for key in crops:
        crops[key] = cv2.resize(crops[key], (img_size, img_size), interpolation=cv2.INTER_AREA)
    
    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    titles = ["ROI-only", "ROI-jitter (10%)", "ROI-5% padding", "ROI-10% padding"]
    
    for idx, title in enumerate(titles):
        axes[idx].imshow(crops[title])
        axes[idx].set_title(title, fontsize=14, fontweight='bold', pad=10)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved ROI comparison grid to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create a 2x2 grid comparing ROI-only, ROI-jitter, ROI-5% padding, and ROI-10% padding",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", type=str, default="data/train.csv",
                        help="Path to dataset CSV file")
    parser.add_argument("--output", type=str,
                        default="outputs/roi_cropping_comparison.png",
                        help="Output path for comparison grid")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Size to resize cropped images for display (default: 224)")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    create_roi_comparison_grid(args.csv, args.output, args.img_size)


if __name__ == "__main__":
    main()

