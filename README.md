# Cattle Body Condition Scoring (BCS) from Photos

CS229 Machine Learning Project: Building a lightweight, explainable model to classify dairy cows into underweight/healthy/overweight from tailhead (rump) photos.

## Project Overview

This project uses transfer learning with EfficientNet to classify dairy cattle body condition scores from images. The model helps farmers get a fast, consistent "second opinion" for welfare monitoring, fertility planning, and feed management.

## Dataset

- **Source**: ScienceDB dairy BCS dataset
- **Labels**: 5 BCS bins (3.25, 3.5, 3.75, 4.0, 4.25)
- **Format**: Images with XML bounding boxes for ROI cropping
- **Splits**: Stratified train/val split (seed=42)

## Quick Start

### Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Training

```bash
# Single experiment with default config
python -m src.train.train --config configs/default.yaml

# Run specific ablation experiment
python -m src.train.train --config configs/ablation_full.yaml

# Run all ablation experiments
bash scripts/run_ablation.sh
```

### Comparing Results

```bash
python scripts/compare_ablation.py
```

## Project Structure

```
.
├── configs/              # YAML configuration files
│   ├── default.yaml      # Default training config
│   └── ablation_*.yaml   # Transfer learning ablation configs
├── data/                 # Data directory
│   ├── train.csv         # Training split
│   ├── val.csv           # Validation split
│   └── manifest.csv      # Full dataset manifest
├── src/
│   ├── models/           # Model creation utilities
│   ├── train/            # Training code
│   ├── eval/             # Evaluation metrics
│   └── data/             # Data utilities
├── scripts/              # Utility scripts
│   ├── run_ablation.sh   # Run all ablation experiments
│   └── compare_ablation.py  # Compare experiment results
└── outputs/              # Experiment outputs (models, metrics, plots)
```

## Transfer Learning Strategies

The codebase supports 4 transfer learning strategies for ablation study:

1. **scratch**: Train from scratch (no pretrained weights)
2. **head_only**: Freeze backbone, train only classifier head
3. **last_block**: Freeze early layers, unfreeze last block(s)
4. **full**: Fine-tune all parameters (default)

Configure in `configs/*.yaml`:
```yaml
model:
  pretrained: true
  finetune_mode: full  # Options: head_only, last_block, full, scratch
```

## Metrics

The evaluation tracks:
- **Accuracy**: Overall classification accuracy
- **Macro-F1**: Average F1 across all classes (balanced)
- **Underweight Recall**: Recall for class 0 (BCS 3.25) - critical for welfare
- **Per-class Recall/Precision**: Detailed per-class metrics
- **Confusion Matrix**: Visual classification patterns

## Configuration

Key config options in `configs/default.yaml`:

```yaml
model:
  backbone: efficientnet_b0
  num_classes: 5
  pretrained: true
  finetune_mode: full

train:
  batch_size: 32
  epochs: 20
  lr: 3e-4
  early_stopping:
    enabled: true
    patience: 5
    monitor: val_macro_f1
```

See `docs/REFACTORING_SUMMARY.md` for detailed documentation of recent changes.

## Output Files

Each experiment creates an output directory with:
- `best_model.pt`: Best model checkpoint
- `config.yaml`: Config used (for reproducibility)
- `metrics.json`: Detailed evaluation metrics
- `confusion_matrix.png`: Visualization of confusion matrix

## License

[Add your license here]

## Authors

Paul Creavin - CS229 Machine Learning Project

