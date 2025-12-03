#!/usr/bin/env python3
"""
Generate detailed error analysis for a specific model on the test set.
Produces plots and a CSV of predictions.
"""
import argparse
import os
import json
import yaml
import sys
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.train.dataset import BcsDataset
from src.models import create_model

def plot_confusion_matrix_custom(cm, class_names, save_path, title="Confusion Matrix", normalize=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_per_class_metrics(y_true, y_pred, class_names, save_path):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    x = np.arange(len(class_names))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, precision, width, label='Precision', color='#4c72b0')
    plt.bar(x, recall, width, label='Recall', color='#55a868')
    plt.bar(x + width, f1, width, label='F1 Score', color='#c44e52')
    
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Per-Class Performance Metrics')
    plt.xticks(x, class_names)
    plt.legend()
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confidence_histogram(df, save_path):
    plt.figure(figsize=(10, 6))
    
    correct = df[df['is_correct']]
    incorrect = df[~df['is_correct']]
    
    plt.hist(correct['confidence'], bins=20, alpha=0.5, label='Correct', color='green', density=True)
    plt.hist(incorrect['confidence'], bins=20, alpha=0.5, label='Incorrect', color='red', density=True)
    
    plt.xlabel('Confidence (Probability of Predicted Class)')
    plt.ylabel('Density')
    plt.title('Confidence Distribution: Correct vs Incorrect Predictions')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_error_magnitude(df, class_names, save_path):
    # Calculate distance between true and predicted class indices
    # Assuming class_names are ordered (which they are for BCS: 3.25, 3.5, 3.75, 4.0, 4.25)
    
    # Map class names to float values to calculate real magnitude
    try:
        class_values = [float(x) for x in class_names]
        val_map = {i: v for i, v in enumerate(class_values)}
        
        true_vals = df['true_label'].map(val_map)
        pred_vals = df['predicted_label'].map(val_map)
        
        diffs = pred_vals - true_vals
        
        plt.figure(figsize=(10, 6))
        counts = diffs.value_counts().sort_index()
        
        # Color code: Green for 0, Yellow for +/- 0.25, Red for others
        colors = []
        for val in counts.index:
            if val == 0:
                colors.append('green')
            elif abs(val) <= 0.25:
                colors.append('orange')
            else:
                colors.append('red')
                
        counts.plot(kind='bar', color=colors)
        plt.xlabel('Prediction Error (Predicted - True BCS)')
        plt.ylabel('Count')
        plt.title('Distribution of Prediction Errors')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
    except ValueError:
        print("Could not parse class names as floats, skipping error magnitude plot.")

def main():
    parser = argparse.ArgumentParser(description="Generate detailed error analysis")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--test-csv", type=str, default="data/test.csv", help="Path to test CSV")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save analysis")
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        
    # Device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Model Params
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("eval", {})
    
    backbone = model_cfg.get("backbone", "efficientnet_b0")
    num_classes = int(model_cfg.get("num_classes", 5))
    img_size = int(data_cfg.get("img_size", 224))
    class_names = eval_cfg.get("class_names", ["3.25", "3.5", "3.75", "4.0", "4.25"])
    
    # Load Model
    print(f"Loading model: {backbone}")
    model = create_model(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=False, # Not needed for inference
        finetune_mode=model_cfg.get("finetune_mode", "full")
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    
    # Load Data
    print(f"Loading test data: {args.test_csv}")
    test_ds = BcsDataset(args.test_csv, img_size=img_size, train=False, do_aug=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)
    
    # Inference
    print("Running inference...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    # Create DataFrame
    df = pd.DataFrame({
        'true_label': all_labels,
        'predicted_label': all_preds,
        'confidence': [p[pred] for p, pred in zip(all_probs, all_preds)]
    })
    
    # Add metadata from dataset if available
    if hasattr(test_ds, 'df'):
        df['image_path'] = test_ds.df['image_path'].values
        # Add other metadata columns if useful
        
    df['is_correct'] = df['true_label'] == df['predicted_label']
    df['true_class_name'] = df['true_label'].apply(lambda x: class_names[x])
    df['pred_class_name'] = df['predicted_label'].apply(lambda x: class_names[x])
    
    # Save Predictions
    csv_path = os.path.join(args.output_dir, "test_predictions_detailed.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved predictions to {csv_path}")
    
    # Generate Plots
    print("Generating plots...")
    
    # 1. Confusion Matrices
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix_custom(cm, class_names, os.path.join(args.output_dir, "confusion_matrix.png"), title="Confusion Matrix (Counts)")
    plot_confusion_matrix_custom(cm, class_names, os.path.join(args.output_dir, "confusion_matrix_normalized.png"), title="Confusion Matrix (Normalized by True Label)", normalize=True)
    
    # 2. Per-Class Metrics
    plot_per_class_metrics(all_labels, all_preds, class_names, os.path.join(args.output_dir, "per_class_metrics.png"))
    
    # 3. Confidence Histogram
    plot_confidence_histogram(df, os.path.join(args.output_dir, "confidence_histogram.png"))
    
    # 4. Error Magnitude
    plot_error_magnitude(df, class_names, os.path.join(args.output_dir, "error_magnitude.png"))
    
    # 5. Top High Confidence Errors
    top_errors = df[~df['is_correct']].sort_values('confidence', ascending=False).head(10)
    top_errors.to_csv(os.path.join(args.output_dir, "top_10_high_confidence_errors.csv"), index=False)
    print("Saved top 10 high confidence errors.")
    
    print(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
