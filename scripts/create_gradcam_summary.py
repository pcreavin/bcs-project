#!/usr/bin/env python3
"""Create a clean summary figure with one Grad-CAM example per class."""
import argparse
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np


def find_best_example_per_class(gradcam_dir, class_names):
    """
    Find one example for each class.
    Returns dict mapping class_idx to filename
    """
    class_to_file = {}
    
    # Find all gradcam images
    pattern = os.path.join(gradcam_dir, "gradcam_sample_*_class_*.png")
    files = sorted(glob.glob(pattern))
    
    for filepath in files:
        filename = os.path.basename(filepath)
        # Extract class from filename: gradcam_sample_XXX_class_Y.png
        parts = filename.split("_class_")
        if len(parts) == 2:
            class_idx = int(parts[1].replace(".png", ""))
            
            # Pick the first one we find per class (they're already sorted)
            if class_idx not in class_to_file:
                class_to_file[class_idx] = filepath
    
    return class_to_file


def create_summary_figure(gradcam_dir, output_path, class_names, title="Grad-CAM Visualizations by BCS Class"):
    """
    Create a clean summary figure with one example per class.
    Each example shows: Original | Heatmap | Overlay
    """
    class_to_file = find_best_example_per_class(gradcam_dir, class_names)
    
    if len(class_to_file) == 0:
        print(f"No Grad-CAM images found in {gradcam_dir}")
        return
    
    # Sort by class index
    sorted_classes = sorted(class_to_file.keys())
    
    # Create figure: rows = classes, cols = 3 (Original, Heatmap, Overlay)
    n_classes = len(sorted_classes)
    fig, axes = plt.subplots(n_classes, 3, figsize=(12, 4 * n_classes))
    
    if n_classes == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, class_idx in enumerate(sorted_classes):
        filepath = class_to_file[class_idx]
        
        # Load the image (it's a 3-panel image)
        img = mpimg.imread(filepath)
        
        # The saved image is already a 3-panel figure, so we need to split it
        # Or we can just display it in the middle column and add labels
        
        # For now, let's display the full 3-panel image in the middle
        # and add class labels
        axes[row_idx, 1].imshow(img)
        axes[row_idx, 1].axis('off')
        axes[row_idx, 1].set_title(f'BCS {class_names[class_idx]}', fontsize=14, fontweight='bold')
        
        # Hide the other columns for now (we'll extract individual panels)
        axes[row_idx, 0].axis('off')
        axes[row_idx, 2].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=300, bpi='tight')
    print(f"Saved summary figure to {output_path}")
    plt.close()


def extract_panels_from_gradcam_image(img_path):
    """
    Extract the three panels (Original, Heatmap, Overlay) from a Grad-CAM image.
    The saved image is 15x5 inches with 3 subplots side by side.
    """
    img = mpimg.imread(img_path)
    h, w = img.shape[:2]
    
    # The image is divided into 3 equal parts horizontally
    # Account for any spacing between panels
    panel_width = w // 3
    
    original = img[:, :panel_width]
    heatmap = img[:, panel_width:2*panel_width]
    overlay = img[:, 2*panel_width:]
    
    return original, heatmap, overlay


def create_clean_summary_figure(gradcam_dir, output_path, class_names, title="Grad-CAM Visualizations by BCS Class"):
    """
    Create a clean summary figure with one example per class.
    Each row shows: Original | Heatmap | Overlay for one class.
    """
    class_to_file = find_best_example_per_class(gradcam_dir, class_names)
    
    if len(class_to_file) == 0:
        print(f"No Grad-CAM images found in {gradcam_dir}")
        return
    
    # Sort by class index
    sorted_classes = sorted(class_to_file.keys())
    n_classes = len(sorted_classes)
    
    # Create figure: rows = classes, cols = 3 (Original, Heatmap, Overlay)
    fig, axes = plt.subplots(n_classes, 3, figsize=(15, 5 * n_classes))
    
    if n_classes == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, class_idx in enumerate(sorted_classes):
        filepath = class_to_file[class_idx]
        
        # Extract the three panels from the saved image
        original, heatmap, overlay = extract_panels_from_gradcam_image(filepath)
        
        # Display each panel
        axes[row_idx, 0].imshow(original)
        axes[row_idx, 0].axis('off')
        if row_idx == 0:
            axes[row_idx, 0].set_title('Original Image', fontsize=12, fontweight='bold', pad=10)
        
        axes[row_idx, 1].imshow(heatmap)
        axes[row_idx, 1].axis('off')
        if row_idx == 0:
            axes[row_idx, 1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold', pad=10)
        
        axes[row_idx, 2].imshow(overlay)
        axes[row_idx, 2].axis('off')
        if row_idx == 0:
            axes[row_idx, 2].set_title('Overlay', fontsize=12, fontweight='bold', pad=10)
        
        # Add class label on the left side of the row
        axes[row_idx, 0].text(-0.15, 0.5, f'BCS {class_names[class_idx]}', 
                fontsize=14, fontweight='bold', 
                rotation=0, ha='right', va='center',
                transform=axes[row_idx, 0].transAxes)
    
    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0.08, 0, 1, 0.99])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved summary figure to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create a clean summary figure with one Grad-CAM example per class",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gradcam-dir", type=str, required=True,
                        help="Directory containing Grad-CAM images")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for summary figure (default: gradcam_dir/summary.png)")
    parser.add_argument("--class-names", type=str, nargs="+", 
                        default=["3.25", "3.5", "3.75", "4.0", "4.25"],
                        help="Class names in order")
    parser.add_argument("--title", type=str, 
                        default="Grad-CAM Visualizations by BCS Class",
                        help="Figure title")
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = os.path.join(args.gradcam_dir, "summary.png")
    
    create_clean_summary_figure(
        args.gradcam_dir, 
        args.output, 
        args.class_names,
        args.title
    )


if __name__ == "__main__":
    main()

