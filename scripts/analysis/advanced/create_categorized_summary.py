#!/usr/bin/env python3
"""Create summary images for categorized error types."""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


def extract_overlay_from_gradcam(img_path):
    """Extract overlay panel from Grad-CAM image."""
    img = mpimg.imread(img_path)
    h, w = img.shape[:2]
    panel_width = w // 3
    overlay = img[:, 2*panel_width:]  # Last panel is overlay
    return overlay


def create_category_summary(csv_path, gradcam_dir, output_dir, category, max_examples=4):
    """Create summary image for a specific error category."""
    df = pd.read_csv(csv_path)
    
    # Filter by category
    category_df = df[df['category'] == category]
    
    if len(category_df) == 0:
        print(f"No examples found for category: {category}")
        return
    
    # Take up to max_examples
    examples = category_df.head(max_examples)
    n_examples = len(examples)
    
    # Create grid (2x2 for up to 4 examples)
    cols = 2
    rows = (n_examples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    class_names = ["3.25", "3.5", "3.75", "4.0", "4.25"]
    
    for idx, row in examples.iterrows():
        if idx >= len(axes):
            break
        
        gradcam_path = os.path.join(gradcam_dir, row['filename'])
        
        if os.path.exists(gradcam_path):
            overlay = extract_overlay_from_gradcam(gradcam_path)
            axes[idx].imshow(overlay)
            axes[idx].axis('off')
            
            # Get prediction from filename or need to load from model
            # For now, just show true class
            true_bcs = row['true_bcs']
            sample_num = row['sample_num']
            axes[idx].set_title(f"Sample #{sample_num}\nTrue: BCS {true_bcs}", 
                               fontsize=12, fontweight='bold')
        else:
            axes[idx].axis('off')
    
    # Hide unused axes
    for idx in range(n_examples, len(axes)):
        axes[idx].axis('off')
    
    category_title = category.replace('_', ' ').title()
    plt.suptitle(f"Error Category: {category_title}", fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    output_path = os.path.join(output_dir, f"errors_{category}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved {category_title} summary to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create summary images for categorized error types",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", type=str,
                        default="outputs/roi_robustness_padding_05/error_analysis/error_categorization.csv",
                        help="Path to categorization CSV")
    parser.add_argument("--gradcam-dir", type=str,
                        default="outputs/roi_robustness_padding_05/gradcam_incorrect",
                        help="Directory containing Grad-CAM images")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/roi_robustness_padding_05/error_analysis",
                        help="Output directory for summaries")
    parser.add_argument("--category", type=str, default=None,
                        help="Specific category to process (default: all)")
    parser.add_argument("--max-examples", type=int, default=4,
                        help="Maximum examples per category (default: 4)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load CSV
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        print("Please run analyze_error_types.py first and fill in the categorization.")
        return
    
    df = pd.read_csv(args.csv)
    
    # Get unique categories (excluding empty ones)
    categories = df[df['category'].notna() & (df['category'] != '')]['category'].unique()
    
    if len(categories) == 0:
        print("No categories found in CSV. Please fill in the 'category' column.")
        print(f"Categories should be: extreme_pose, mud_dirt, borderline, or non_tailhead")
        return
    
    if args.category:
        categories = [args.category] if args.category in categories else []
    
    print(f"Creating summaries for categories: {categories}")
    
    for category in categories:
        create_category_summary(args.csv, args.gradcam_dir, args.output_dir, 
                               category, args.max_examples)


if __name__ == "__main__":
    main()

