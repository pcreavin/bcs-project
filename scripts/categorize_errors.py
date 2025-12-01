#!/usr/bin/env python3
"""Helper script to categorize incorrect predictions for analysis."""
import argparse
import os
import json
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def load_error_info(gradcam_dir):
    """Load information about incorrect predictions."""
    errors = []
    
    # Find all gradcam images
    pattern = os.path.join(gradcam_dir, "gradcam_sample_*_class_*.png")
    files = sorted(glob.glob(pattern))
    
    for filepath in files:
        filename = os.path.basename(filepath)
        # Extract sample number and class from filename
        # gradcam_sample_XXX_class_Y.png
        parts = filename.replace(".png", "").split("_")
        sample_num = int(parts[2])
        true_class = int(parts[4])
        
        errors.append({
            'filename': filename,
            'filepath': filepath,
            'sample_num': sample_num,
            'true_class': true_class
        })
    
    return errors


def display_image_with_info(img_path, true_class, pred_class, class_names, save_path=None):
    """Display image with classification info."""
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img)
    ax.axis('off')
    
    true_name = class_names[true_class] if true_class < len(class_names) else str(true_class)
    pred_name = class_names[pred_class] if pred_class < len(class_names) else str(pred_class)
    
    title = f"True: BCS {true_name} | Pred: BCS {pred_name}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()


def create_error_catalog(gradcam_dir, test_csv, output_dir, class_names):
    """Create a catalog of error images for manual review."""
    import glob
    
    # Load test data to get image paths and predictions
    test_df = pd.read_csv(test_csv)
    
    # Find all error images
    pattern = os.path.join(gradcam_dir, "gradcam_sample_*_class_*.png")
    error_files = sorted(glob.glob(pattern))
    
    os.makedirs(output_dir, exist_ok=True)
    
    catalog = []
    
    for filepath in error_files:
        filename = os.path.basename(filepath)
        parts = filename.replace(".png", "").split("_")
        sample_num = int(parts[2])
        true_class = int(parts[4])
        
        # Get corresponding row from test data
        if sample_num < len(test_df):
            row = test_df.iloc[sample_num]
            img_path = row['image_path']
            
            # We need to get the prediction - this would require running inference
            # For now, we'll just note the true class
            
            catalog.append({
                'sample_num': sample_num,
                'filename': filename,
                'true_class': true_class,
                'true_bcs': class_names[true_class],
                'image_path': img_path,
                'gradcam_path': filepath
            })
    
    # Save catalog as JSON
    catalog_path = os.path.join(output_dir, "error_catalog.json")
    with open(catalog_path, 'w') as f:
        json.dump(catalog, f, indent=2)
    
    print(f"Created error catalog with {len(catalog)} errors")
    print(f"Saved to: {catalog_path}")
    
    return catalog


def main():
    parser = argparse.ArgumentParser(
        description="Categorize incorrect predictions for analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gradcam-dir", type=str, required=True,
                        help="Directory containing Grad-CAM error images")
    parser.add_argument("--test-csv", type=str, default="data/test.csv",
                        help="Path to test CSV file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for catalog (default: gradcam_dir/../error_analysis)")
    parser.add_argument("--class-names", type=str, nargs="+", 
                        default=["3.25", "3.5", "3.75", "4.0", "4.25"],
                        help="Class names in order")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        base_dir = os.path.dirname(args.gradcam_dir)
        args.output_dir = os.path.join(base_dir, "error_analysis")
    
    catalog = create_error_catalog(
        args.gradcam_dir,
        args.test_csv,
        args.output_dir,
        args.class_names
    )
    
    print(f"\nError distribution by true class:")
    from collections import Counter
    class_counts = Counter([e['true_class'] for e in catalog])
    for class_idx, count in sorted(class_counts.items()):
        print(f"  BCS {args.class_names[class_idx]}: {count} errors")


if __name__ == "__main__":
    import glob
    main()

