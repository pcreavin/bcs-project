"""Run baseline experiments (Logistic Regression, SVM) using frozen features from a timm model.

Usage:
  python scripts/run_baselines.py --config configs/default.yaml

The script will:
 - load train/val CSVs using `src.train.dataset.BcsDataset`
 - create a timm backbone (frozen) via `src.models.factory.create_model`
 - extract features with `src.models.baselines.extract_frozen_features`
 - train all baseline models and save artifacts:
   * Logistic Regression -> outputs/baseline_logreg
   * SVM (linear kernel) -> outputs/baseline_svm_linear
   * SVM (RBF kernel) -> outputs/baseline_svm_rbf
"""

import argparse
import yaml
import os
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

from src.train.dataset import BcsDataset
from src.models.factory import create_model
from src.models.baselines import extract_frozen_features, train_logistic_regression, train_svm


def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def auto_device(cfg_device: str):
    if cfg_device and cfg_device.lower() != 'null':
        return torch.device(cfg_device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloader(csv_path, img_size, batch_size, train, do_aug, num_workers):
    ds = BcsDataset(csv_path, img_size=img_size, train=train, do_aug=do_aug)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=True)


def save_metrics_json(metrics, report, cm, out_dir, name):
    """Save metrics in a JSON format similar to deep learning experiments."""
    # Extract per-class recall (critical metric)
    per_class_recall = {}
    for class_id in range(5):
        class_key = str(class_id)
        if class_key in report:
            per_class_recall[class_id] = report[class_key].get('recall', 0.0)
    
    # Create comprehensive metrics dict
    results = {
        'accuracy': metrics['accuracy'],
        'macro_f1': metrics['macro_f1'],
        'underweight_recall': per_class_recall.get(0, 0.0),  # Class 0 is BCS 3.25 (underweight)
        'per_class_recall': per_class_recall,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    metrics_path = os.path.join(out_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  Saved comprehensive metrics to {metrics_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='configs/default.yaml')
    p.add_argument('--baselines', type=str, default='all', 
                   help='Which baselines to run: all, logreg, svm_linear, svm_rbf')
    p.add_argument('--max-samples', type=int, default=None,
                   help='Limit number of samples for quick smoke test (e.g., 1000)')
    args = p.parse_args()

    cfg = load_config(args.config)

    data_cfg = cfg.get('data', {})
    model_cfg = cfg.get('model', {})
    train_cfg = cfg.get('train', {})

    train_csv = data_cfg.get('train', 'data/train.csv')
    val_csv = data_cfg.get('val', 'data/val.csv')
    img_size = data_cfg.get('img_size', 224)
    do_aug = data_cfg.get('do_aug', False)

    batch_size = train_cfg.get('batch_size', 32)
    num_workers = train_cfg.get('num_workers', 2)
    device = auto_device(train_cfg.get('device', None))

    print('='*60)
    print('BASELINE MODELS: Logistic Regression & SVM')
    print('='*60)
    print(f'Device: {device}')
    print(f'Train CSV: {train_csv}')
    print(f'Val CSV: {val_csv}')
    print(f'Image size: {img_size}')
    print()

    print('Loading dataloaders...')
    train_loader = make_dataloader(train_csv, img_size, batch_size, train=True, do_aug=do_aug, num_workers=num_workers)
    val_loader = make_dataloader(val_csv, img_size, batch_size, train=False, do_aug=False, num_workers=num_workers)

    backbone = model_cfg.get('backbone', 'efficientnet_b0')
    num_classes = model_cfg.get('num_classes', 5)

    print(f'Creating model {backbone} (frozen backbone for feature extraction)')
    # Create model with head_only so backbone is frozen and forward_features works for timm models
    model = create_model(backbone, num_classes=num_classes, pretrained=model_cfg.get('pretrained', True), finetune_mode='head_only')
    model.to(device)
    print()

    print('Extracting features from train set...')
    X_train, y_train = extract_frozen_features(model, train_loader, device)
    print(f'  Train features shape: {X_train.shape}')
    print(f'  Train labels shape: {y_train.shape}')
    
    print('Extracting features from val set...')
    X_val, y_val = extract_frozen_features(model, val_loader, device)
    print(f'  Val features shape: {X_val.shape}')
    print(f'  Val labels shape: {y_val.shape}')
    
    # Limit samples for smoke test if requested
    if args.max_samples is not None:
        print(f'\n[SMOKE TEST MODE] Limiting to {args.max_samples} samples')
        X_train = X_train[:args.max_samples]
        y_train = y_train[:args.max_samples]
        X_val = X_val[:min(args.max_samples // 4, len(X_val))]
        y_val = y_val[:min(args.max_samples // 4, len(y_val))]
        print(f'  Limited train: {X_train.shape}, val: {X_val.shape}')
    
    print()

    # Dictionary to store all results for comparison
    all_results = {}

    # Train Logistic Regression
    if args.baselines in ['all', 'logreg']:
        print('='*60)
        print('Training Logistic Regression baseline...')
        print('='*60)
        out_dir = 'outputs/baseline_logreg'
        os.makedirs(out_dir, exist_ok=True)
        
        metrics, report, cm = train_logistic_regression(X_train, y_train, X_val, y_val, out_dir=out_dir)
        save_metrics_json(metrics, report, cm, out_dir, 'logreg')
        
        print('Logistic Regression Results:')
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        print(f"  Underweight Recall (Class 0): {report['0']['recall']:.4f}")
        print(f'  Saved artifacts to {Path(out_dir).resolve()}')
        print()
        
        all_results['logreg'] = metrics

    # Train SVM with linear kernel
    if args.baselines in ['all', 'svm_linear']:
        print('='*60)
        print('Training SVM (linear kernel) baseline...')
        print('='*60)
        out_dir = 'outputs/baseline_svm_linear'
        os.makedirs(out_dir, exist_ok=True)
        
        metrics, report, cm = train_svm(X_train, y_train, X_val, y_val, out_dir=out_dir, kernel='linear')
        save_metrics_json(metrics, report, cm, out_dir, 'svm_linear')
        
        print('SVM (linear) Results:')
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        print(f"  Underweight Recall (Class 0): {report['0']['recall']:.4f}")
        print(f'  Saved artifacts to {Path(out_dir).resolve()}')
        print()
        
        all_results['svm_linear'] = metrics

    # Train SVM with RBF kernel
    if args.baselines in ['all', 'svm_rbf']:
        print('='*60)
        print('Training SVM (RBF kernel) baseline...')
        print('='*60)
        out_dir = 'outputs/baseline_svm_rbf'
        os.makedirs(out_dir, exist_ok=True)
        
        metrics, report, cm = train_svm(X_train, y_train, X_val, y_val, out_dir=out_dir, kernel='rbf')
        save_metrics_json(metrics, report, cm, out_dir, 'svm_rbf')
        
        print('SVM (RBF) Results:')
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        print(f"  Underweight Recall (Class 0): {report['0']['recall']:.4f}")
        print(f'  Saved artifacts to {Path(out_dir).resolve()}')
        print()
        
        all_results['svm_rbf'] = metrics

    # Print comparison summary
    if len(all_results) > 1:
        print('='*60)
        print('BASELINE COMPARISON SUMMARY')
        print('='*60)
        print(f"{'Model':<20} {'Accuracy':<12} {'Macro F1':<12}")
        print('-'*60)
        for name, metrics in all_results.items():
            print(f"{name:<20} {metrics['accuracy']:<12.4f} {metrics['macro_f1']:<12.4f}")
        print('='*60)


if __name__ == '__main__':
    main()
