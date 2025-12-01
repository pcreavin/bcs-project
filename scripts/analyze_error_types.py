#!/usr/bin/env python3
"""Analyze error types in incorrect predictions and create categorized visualizations."""
import argparse
import os
import json
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import glob


def load_error_catalog(catalog_path):
    """Load error catalog."""
    with open(catalog_path, 'r') as f:
        return json.load(f)


def create_error_grid(gradcam_dir, catalog, output_path, title="Error Analysis"):
    """Create a grid of error images for manual review."""
    n_errors = len(catalog)
    cols = 5
    rows = (n_errors + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    class_names = ["3.25", "3.5", "3.75", "4.0", "4.25"]
    
    for idx, error in enumerate(catalog):
        if idx >= len(axes):
            break
            
        gradcam_path = error['gradcam_path']
        
        # Load the overlay panel from the gradcam image
        img = mpimg.imread(gradcam_path)
        h, w = img.shape[:2]
        panel_width = w // 3
        overlay = img[:, 2*panel_width:]  # Last panel is overlay
        
        axes[idx].imshow(overlay)
        axes[idx].axis('off')
        
        # Add label with true and predicted class
        true_bcs = error['true_bcs']
        sample_num = error['sample_num']
        axes[idx].set_title(f"#{sample_num}\nTrue: {true_bcs}", 
                           fontsize=10, fontweight='bold')
    
    # Hide unused axes
    for idx in range(len(catalog), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved error grid to {output_path}")
    plt.close()


def create_categorized_summary(gradcam_dir, catalog, output_dir, class_names):
    """Create summary images for different error categories."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original images to check for specific characteristics
    categories = {
        'extreme_pose': [],
        'mud_dirt': [],
        'borderline': [],
        'non_tailhead': []
    }
    
    # For now, we'll create a script output that helps identify these
    # The user can manually review and categorize
    
    print("\n" + "="*60)
    print("ERROR ANALYSIS GUIDE")
    print("="*60)
    print("\nReview the error grid image to identify:")
    print("\n1. EXTREME POSE:")
    print("   - Look for images where the cow is at unusual angles")
    print("   - Side views, extreme tilts, or non-standard positions")
    print("   - Check sample numbers in the grid")
    
    print("\n2. MUD/DIRT:")
    print("   - Look for images with visible mud, dirt, or debris")
    print("   - Obscured tailhead region")
    print("   - Dirty or stained fur")
    
    print("\n3. BORDERLINE CASE:")
    print("   - Cases where true and predicted are adjacent classes")
    print("   - Low confidence predictions (check original output)")
    print("   - Ambiguous body condition")
    
    print("\n4. NON-TAILHEAD FEATURES:")
    print("   - Check Grad-CAM heatmaps focusing on wrong regions")
    print("   - Attention on background, legs, or other body parts")
    print("   - Not focusing on tailhead area")
    
    print("\n" + "="*60)
    print(f"\nTotal errors to review: {len(catalog)}")
    print(f"Error grid saved for visual review")
    print("="*60)
    
    # Create a CSV for manual categorization
    csv_data = []
    for error in catalog:
        csv_data.append({
            'sample_num': error['sample_num'],
            'filename': error['filename'],
            'true_class': error['true_class'],
            'true_bcs': error['true_bcs'],
            'image_path': error['image_path'],
            'gradcam_path': error['gradcam_path'],
            'category': '',  # To be filled manually
            'notes': ''
        })
    
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(output_dir, "error_categorization.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCreated categorization CSV: {csv_path}")
    print("Fill in the 'category' column with: extreme_pose, mud_dirt, borderline, or non_tailhead")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze error types in incorrect predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--catalog", type=str, 
                        default="outputs/roi_robustness_padding_05/error_analysis/error_catalog.json",
                        help="Path to error catalog JSON")
    parser.add_argument("--gradcam-dir", type=str,
                        default="outputs/roi_robustness_padding_05/gradcam_incorrect",
                        help="Directory containing Grad-CAM images")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/roi_robustness_padding_05/error_analysis",
                        help="Output directory for analysis")
    parser.add_argument("--class-names", type=str, nargs="+", 
                        default=["3.25", "3.5", "3.75", "4.0", "4.25"],
                        help="Class names in order")
    
    args = parser.parse_args()
    
    # Load catalog
    catalog = load_error_catalog(args.catalog)
    
    # Update paths in catalog
    for error in catalog:
        error['gradcam_path'] = os.path.join(args.gradcam_dir, error['filename'])
    
    # Create error grid for visual review
    grid_path = os.path.join(args.output_dir, "error_grid_all.png")
    create_error_grid(args.gradcam_dir, catalog, grid_path, 
                     title="All Error Cases - Review for Categorization")
    
    # Create categorization guide
    create_categorized_summary(args.gradcam_dir, catalog, args.output_dir, args.class_names)
    
    print(f"\nNext steps:")
    print(f"1. Review the error grid: {grid_path}")
    print(f"2. Fill in the CSV: {os.path.join(args.output_dir, 'error_categorization.csv')}")
    print(f"3. Use the categorized errors for your report")


if __name__ == "__main__":
    main()

