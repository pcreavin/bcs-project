#!/usr/bin/env python3
"""
Generate additional error analysis visualizations:
1. 2x2 grid of high-confidence error images
2. Adjacent vs non-adjacent error statistics
"""
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Rectangle

def plot_high_confidence_errors(predictions_csv, output_path, n_images=4):
    """Create a 2x2 grid showing high-confidence errors."""
    df = pd.read_csv(predictions_csv)
    
    # Get top N high-confidence errors
    errors = df[~df['is_correct']].sort_values('confidence', ascending=False).head(n_images)
    
    if len(errors) == 0:
        print("No errors found!")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(errors.iterrows()):
        if idx >= n_images:
            break
            
        # Load image
        img_path = row['image_path']
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load image: {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Display image
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        # Add title with prediction info
        true_label = row['true_class_name']
        pred_label = row['pred_class_name']
        confidence = row['confidence']
        
        title = f"True: {true_label} | Pred: {pred_label}\nConfidence: {confidence:.1%}"
        axes[idx].set_title(title, fontsize=12, fontweight='bold', pad=10)
        
        # Add colored border (red for error)
        rect = Rectangle((0, 0), img.shape[1]-1, img.shape[0]-1, 
                         linewidth=4, edgecolor='red', facecolor='none')
        axes[idx].add_patch(rect)
    
    # Hide unused subplots
    for idx in range(len(errors), n_images):
        axes[idx].axis('off')
    
    plt.suptitle('High-Confidence Misclassifications', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved high-confidence error visualization to {output_path}")

def calculate_adjacent_error_stats(predictions_csv, class_names, output_path):
    """Calculate and visualize adjacent vs non-adjacent errors."""
    df = pd.read_csv(predictions_csv)
    
    # Get only errors
    errors = df[~df['is_correct']].copy()
    
    if len(errors) == 0:
        print("No errors found!")
        return
    
    # Calculate distance between true and predicted (in class indices)
    errors['error_distance'] = abs(errors['predicted_label'] - errors['true_label'])
    
    # Categorize errors
    adjacent_errors = errors[errors['error_distance'] == 1]
    non_adjacent_errors = errors[errors['error_distance'] > 1]
    
    total_errors = len(errors)
    n_adjacent = len(adjacent_errors)
    n_non_adjacent = len(non_adjacent_errors)
    
    pct_adjacent = (n_adjacent / total_errors) * 100 if total_errors > 0 else 0
    pct_non_adjacent = (n_non_adjacent / total_errors) * 100 if total_errors > 0 else 0
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart
    sizes = [n_adjacent, n_non_adjacent]
    labels = [f'Adjacent (±1 class)\n{n_adjacent} errors ({pct_adjacent:.1f}%)',
              f'Non-Adjacent (>1 class)\n{n_non_adjacent} errors ({pct_non_adjacent:.1f}%)']
    colors = ['#ff9999', '#cc0000']
    explode = (0.05, 0.05)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax1.set_title('Error Type Distribution', fontsize=14, fontweight='bold', pad=20)
    
    # Bar chart by error distance
    error_dist_counts = errors['error_distance'].value_counts().sort_index()
    
    colors_bar = ['orange' if d == 1 else 'red' for d in error_dist_counts.index]
    
    ax2.bar(error_dist_counts.index, error_dist_counts.values, color=colors_bar, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Error Distance (|Predicted - True|)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Errors', fontsize=12, fontweight='bold')
    ax2.set_title('Error Distribution by Distance', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.set_xticks(range(int(error_dist_counts.index.max()) + 1))
    
    # Add text annotations on bars
    for i, (dist, count) in enumerate(error_dist_counts.items()):
        ax2.text(dist, count + max(error_dist_counts.values) * 0.02, str(count), 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved adjacent error analysis to {output_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("ADJACENT vs NON-ADJACENT ERROR ANALYSIS")
    print("="*60)
    print(f"Total Errors: {total_errors}")
    print(f"Adjacent Errors (±1 class): {n_adjacent} ({pct_adjacent:.2f}%)")
    print(f"Non-Adjacent Errors (>1 class): {n_non_adjacent} ({pct_non_adjacent:.2f}%)")
    print("\nError Distance Breakdown:")
    for dist, count in error_dist_counts.items():
        pct = (count / total_errors) * 100
        print(f"  Distance {int(dist)}: {count} errors ({pct:.2f}%)")
    print("="*60)
    
    # Save statistics to text file
    stats_path = output_path.replace('.png', '_stats.txt')
    with open(stats_path, 'w') as f:
        f.write("ADJACENT vs NON-ADJACENT ERROR ANALYSIS\n")
        f.write("="*60 + "\n")
        f.write(f"Total Errors: {total_errors}\n")
        f.write(f"Adjacent Errors (±1 class): {n_adjacent} ({pct_adjacent:.2f}%)\n")
        f.write(f"Non-Adjacent Errors (>1 class): {n_non_adjacent} ({pct_non_adjacent:.2f}%)\n")
        f.write("\nError Distance Breakdown:\n")
        for dist, count in error_dist_counts.items():
            pct = (count / total_errors) * 100
            f.write(f"  Distance {int(dist)}: {count} errors ({pct:.2f}%)\n")
    print(f"Saved statistics to {stats_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate additional error visualizations")
    parser.add_argument("--predictions-csv", type=str, required=True, 
                       help="Path to predictions CSV from error analysis")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save visualizations")
    parser.add_argument("--class-names", type=str, nargs='+', 
                       default=["3.25", "3.5", "3.75", "4.0", "4.25"],
                       help="Class names")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate visualizations
    print("Generating high-confidence error visualization...")
    plot_high_confidence_errors(
        args.predictions_csv,
        os.path.join(args.output_dir, "high_confidence_errors_grid.png"),
        n_images=4
    )
    
    print("\nCalculating adjacent error statistics...")
    calculate_adjacent_error_stats(
        args.predictions_csv,
        args.class_names,
        os.path.join(args.output_dir, "adjacent_error_analysis.png")
    )
    
    print("\nAll visualizations complete!")

if __name__ == "__main__":
    main()
