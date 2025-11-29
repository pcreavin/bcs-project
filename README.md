# Cattle Body Condition Scoring (BCS) from Photos

CS229 Machine Learning Project: Building a lightweight, explainable model to classify dairy cows into underweight/healthy/overweight from tailhead (rump) photos.

## Project Overview

This project uses transfer learning with EfficientNet to classify dairy cattle body condition scores from images. The model helps farmers get a fast, consistent "second opinion" for welfare monitoring, fertility planning, and feed management.

## Dataset

- **Source**: ScienceDB dairy BCS dataset
- **Labels**: 5 BCS bins (3.25, 3.5, 3.75, 4.0, 4.25)
- **Format**: Images with XML bounding boxes for ROI cropping
- **Splits**: Stratified 70/15/15 train/val/test split (seed=42)
  - **Train**: 37,496 samples (70%)
  - **Validation**: 8,035 samples (15%)
  - **Test**: 8,035 samples (15%)

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training a Model

```bash
# Train with default config
python -m src.train.train --config configs/default.yaml

# Train specific experiment
python -m src.train.train --config configs/ablation_full.yaml

# Run all ablation experiments
bash scripts/run_ablation.sh
```

### Evaluating Models

```bash
# Evaluate on test set
python scripts/evaluate_test.py \
  --checkpoint outputs/ablation_full/best_model.pt \
  --config outputs/ablation_full/config.yaml

# Compare ablation results
python scripts/compare_ablation.py
```

## Project Structure

```
.
├── configs/              # YAML configuration files
│   ├── default.yaml      # Default training config
│   └── ablation_*.yaml   # Experiment configs
├── data/                 # Data directory
│   ├── train.csv         # Training split
│   ├── val.csv           # Validation split
│   └── manifest.csv      # Full dataset manifest
├── src/
│   ├── models/           # Model creation utilities
│   ├── train/            # Training code
│   └── eval/             # Evaluation metrics
├── scripts/              # Utility scripts
│   ├── run_ablation.sh   # Run all ablation experiments
│   ├── compare_ablation.py  # Compare results
│   └── evaluate_test.py  # Test set evaluation
└── outputs/              # Experiment outputs (models, metrics, plots)
```

## Key Features

- **Transfer Learning Ablation**: Compare scratch, head_only, last_block, and full fine-tuning
- **Ordinal Regression**: CORAL-style ordinal head for distance-aware predictions
- **Ensemble Methods**: Combine classification and ordinal models
- **ROI Robustness**: Test model robustness to cropping strategies
- **Comprehensive Metrics**: Accuracy, Macro-F1, per-class recall/precision, confusion matrices

## Configuration

Key config options:

```yaml
model:
  backbone: efficientnet_b0
  num_classes: 5
  pretrained: true
  finetune_mode: full  # Options: head_only, last_block, full, scratch

train:
  batch_size: 32
  epochs: 20
  lr: 3e-4
  early_stopping:
    enabled: true
    patience: 5
    monitor: val_macro_f1
```

## Output Files

Each experiment creates an output directory with:
- `best_model.pt`: Best model checkpoint
- `config.yaml`: Config used (for reproducibility)
- `metrics.json`: Detailed evaluation metrics
- `confusion_matrix.png`: Visualization of confusion matrix

## Author

Paul Creavin - CS229 Machine Learning Project
