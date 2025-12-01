#!/usr/bin/env python3
"""Create a 2x2 grid of specific error cases with heatmaps."""
import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import yaml
from torch.utils.data import DataLoader

from src.train.dataset import BcsDataset
from src.models import create_model


def extract_heatmap_from_gradcam(img_path):
    """Extract heatmap panel (middle column) from Grad-CAM image."""
    img = mpimg.imread(img_path)
    h, w = img.shape[:2]
    panel_width = w // 3
    heatmap = img[:, panel_width:2*panel_width]  # Middle panel is heatmap
    return heatmap


def get_prediction_for_sample(model, dataset, sample_idx, device, class_names):
    """Get prediction for a specific sample."""
    model.eval()
    with torch.no_grad():
        image_tensor, true_label = dataset[sample_idx]
        image_tensor = image_tensor.unsqueeze(0).to(device)
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    
    true_bcs = class_names[true_label] if true_label < len(class_names) else str(true_label)
    pred_bcs = class_names[pred_class] if pred_class < len(class_names) else str(pred_class)
    
    return true_bcs, pred_bcs, confidence


def create_error_grid(sample_numbers, gradcam_dir, test_csv, config_path, checkpoint_path, 
                     output_path, class_names, device):
    """Create 2x2 grid of specific error cases."""
    # Load model to get predictions
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    
    backbone = model_cfg.get("backbone", "efficientnet_b0")
    num_classes = int(model_cfg.get("num_classes", 5))
    pretrained = bool(model_cfg.get("pretrained", True))
    finetune_mode = model_cfg.get("finetune_mode", "full")
    img_size = int(data_cfg.get("img_size", 224))
    
    # Create and load model
    from src.models import create_model
    model = create_model(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        finetune_mode=finetune_mode
    )
    model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Load dataset
    test_ds = BcsDataset(test_csv, img_size=img_size, train=False, do_aug=False)
    
    # Load error catalog to get filenames
    catalog_path = os.path.join(os.path.dirname(gradcam_dir), "error_analysis", "error_catalog.json")
    with open(catalog_path, 'r') as f:
        catalog = json.load(f)
    
    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    for grid_idx, sample_num in enumerate(sample_numbers):
        # Find this sample in catalog
        sample_info = None
        for error in catalog:
            if error['sample_num'] == sample_num:
                sample_info = error
                break
        
        if sample_info is None:
            print(f"Warning: Sample {sample_num} not found in catalog")
            axes[grid_idx].axis('off')
            continue
        
        # Get Grad-CAM image path
        gradcam_filename = sample_info['filename']
        gradcam_path = os.path.join(gradcam_dir, gradcam_filename)
        
        if not os.path.exists(gradcam_path):
            print(f"Warning: Grad-CAM image not found: {gradcam_path}")
            axes[grid_idx].axis('off')
            continue
        
        # Extract heatmap
        heatmap = extract_heatmap_from_gradcam(gradcam_path)
        
        # Get true and predicted labels
        true_bcs, pred_bcs, confidence = get_prediction_for_sample(
            model, test_ds, sample_num, device, class_names
        )
        
        # Display heatmap
        axes[grid_idx].imshow(heatmap)
        axes[grid_idx].axis('off')
        
        # Add label with true and predicted
        axes[grid_idx].set_title(f"True: BCS {true_bcs} | Pred: BCS {pred_bcs}", 
                                fontsize=12, fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved error grid to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create 2x2 grid of specific error cases",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sample-numbers", type=int, nargs=4, 
                        default=[21, 16, 23, 24],
                        help="Sample numbers to include (default: 21 16 23 24)")
    parser.add_argument("--gradcam-dir", type=str,
                        default="outputs/roi_robustness_padding_05/gradcam_incorrect",
                        help="Directory containing Grad-CAM images")
    parser.add_argument("--test-csv", type=str, default="data/test.csv",
                        help="Path to test CSV file")
    parser.add_argument("--config", type=str,
                        default="outputs/roi_robustness_padding_05/config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/roi_robustness_padding_05/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--output", type=str,
                        default="outputs/roi_robustness_padding_05/error_analysis/specific_errors_2x2.png",
                        help="Output path for grid")
    parser.add_argument("--class-names", type=str, nargs="+", 
                        default=["3.25", "3.5", "3.75", "4.0", "4.25"],
                        help="Class names in order")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cpu/cuda/mps, default: auto-detect)")
    
    args = parser.parse_args()
    
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
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    create_error_grid(
        args.sample_numbers,
        args.gradcam_dir,
        args.test_csv,
        args.config,
        args.checkpoint,
        args.output,
        args.class_names,
        device
    )


if __name__ == "__main__":
    main()

