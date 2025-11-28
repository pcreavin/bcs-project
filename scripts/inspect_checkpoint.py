#!/usr/bin/env python3
"""Inspect checkpoint structure to understand model architecture."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

def inspect_checkpoint(checkpoint_path: str, config_path: str):
    """Print checkpoint structure details."""
    print("=" * 70)
    print("CHECKPOINT INSPECTION")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint_keys = list(checkpoint.keys())
    
    print(f"\nTotal keys: {len(checkpoint_keys)}")
    print(f"\nFirst 20 keys:")
    for i, key in enumerate(checkpoint_keys[:20]):
        shape = checkpoint[key].shape if hasattr(checkpoint[key], 'shape') else 'N/A'
        print(f"  {i+1}. {key:60s} {str(shape)}")
    
    # Check for different structures
    print("\n" + "=" * 70)
    print("STRUCTURE ANALYSIS")
    print("=" * 70)
    
    has_sequential = any("0." in k or "1." in k for k in checkpoint_keys)
    has_classifier = any("classifier" in k for k in checkpoint_keys)
    has_head = any("head" in k for k in checkpoint_keys)
    
    print(f"Sequential model (0., 1. keys): {has_sequential}")
    print(f"Classifier keys: {has_classifier}")
    print(f"Head keys: {has_head}")
    
    if has_classifier:
        classifier_keys = [k for k in checkpoint_keys if "classifier" in k]
        print(f"\nClassifier-related keys ({len(classifier_keys)}):")
        for key in classifier_keys[:10]:
            shape = checkpoint[key].shape if hasattr(checkpoint[key], 'shape') else 'N/A'
            print(f"  {key:60s} {str(shape)}")
        
        # Check for nested structure
        has_classifier_fc = any("classifier.fc" in k for k in classifier_keys)
        has_classifier_nested = any("classifier.0" in k or "classifier.1" in k for k in classifier_keys)
        print(f"\n  Has classifier.fc: {has_classifier_fc}")
        print(f"  Has classifier.0/1: {has_classifier_nested}")
    
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    model_cfg = cfg.get("model", {})
    num_classes = int(model_cfg.get("num_classes", 5))
    num_thresholds = num_classes - 1
    
    print(f"\nConfig says: num_classes={num_classes}, so num_thresholds={num_thresholds}")
    
    # Find the output layer
    print("\n" + "=" * 70)
    print("OUTPUT LAYER ANALYSIS")
    print("=" * 70)
    
    # Look for the final layer (should have num_thresholds outputs)
    final_layers = []
    for key in checkpoint_keys:
        if "classifier" in key.lower() or "head" in key.lower() or "fc" in key.lower():
            if "weight" in key or "bias" in key:
                final_layers.append(key)
    
    if final_layers:
        print(f"\nPotential output layers:")
        for key in final_layers[:5]:
            shape = checkpoint[key].shape
            print(f"  {key:60s} {str(shape)}")
            if "weight" in key:
                print(f"    -> Output size: {shape[0] if len(shape) >= 1 else 'unknown'}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    inspect_checkpoint(args.checkpoint, args.config)

