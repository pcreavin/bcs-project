# Cattle Body Condition Scoring (BCS) from Photos

CS229 Machine Learning Project: Building a lightweight, explainable model to classify dairy cows into underweight/healthy/overweight from tailhead (rump) photos.

## Project Overview

This project uses deep learning with transfer learning to classify dairy cattle body condition scores from images. The model helps farmers get a fast, consistent "second opinion" for welfare monitoring, fertility planning, and feed management. We explore multiple architectures, transfer learning strategies, and ensemble methods to achieve robust classification performance.

### Key Features

- **Multiple Architectures**: EfficientNet (B0-B3), ConvNeXt-Tiny, Swin Transformer, RegNetY-8GF
- **Transfer Learning Ablations**: Scratch, head-only, last-block, and full fine-tuning
- **Ordinal Regression**: CORAL-based ordinal regression for better handling of ordered classes
- **Ensemble Methods**: Majority voting, weighted voting, and consensus-based ensembles
- **ROI Cropping**: Automatic region-of-interest extraction with jitter and padding options
- **Comprehensive Evaluation**: Accuracy, macro-F1, per-class metrics, and underweight recall

## Dataset

- **Source**: ScienceDB dairy BCS dataset
- **Labels**: 5 BCS bins (3.25, 3.5, 3.75, 4.0, 4.25)
- **Format**: Images with XML bounding boxes for ROI cropping
- **Splits**: Stratified 70/15/15 train/val/test split (seed=42)
  - **Train**: 37,496 samples (70%) - used for training
  - **Validation**: 8,035 samples (15%) - used for model selection, early stopping
  - **Test**: 8,035 samples (15%) - held out for final evaluation only

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd bcs-project

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Check device availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test model creation
python -c "from src.models import create_model; m = create_model('efficientnet_b0', 5); print('Model created successfully!')"
```

## Data Preparation

### Step 1: Extract Dataset

If you have the dataset in RAR format:

```bash
# Extract dataset (requires 7-Zip)
7z x dataset.rar -odata/raw
```

### Step 2: Create Manifest

Generate the manifest CSV from raw dataset:

```bash
python scripts/data/make_manifest.py
```

This creates `data/manifest.csv` with image paths, labels, and bounding box coordinates.

### Step 3: Create Train/Val/Test Splits

```bash
python scripts/data/split_manifest_3way.py
```

This creates:

- `data/train.csv` (70%)
- `data/val.csv` (15%)
- `data/test.csv` (15%)

### Step 4: (Optional) Preprocess Images

For faster training, you can preprocess images once:

```bash
python scripts/data/preprocess_images.py \
  --train-csv data/train.csv \
  --val-csv data/val.csv \
  --test-csv data/test.csv \
  --output-dir data/processed_224 \
  --img-size 224
```

Then update your config to use the preprocessed CSVs:

```yaml
data:
  train: data/processed_224/train_processed.csv
  val: data/processed_224/val_processed.csv
  skip_resize: true # Images already resized
```

## Running Experiments

### Quick Start: Single Experiment

```bash
# Run with default config
python -m src.train.train --config configs/default.yaml

# Run specific experiment
python -m src.train.train --config configs/ablation_full.yaml
```

### Transfer Learning Ablation Study

Run all 4 transfer learning strategies:

```bash
bash scripts/training/run_ablation.sh
```

Or run individually:

```bash
python -m src.train.train --config configs/ablation_scratch.yaml
python -m src.train.train --config configs/ablation_head_only.yaml
python -m src.train.train --config configs/ablation_last_block.yaml
python -m src.train.train --config configs/ablation_full.yaml
```

### Enhanced Training Experiments

```bash
# With data augmentation and cosine scheduler
python -m src.train.train --config configs/ablation_full_enhanced.yaml

# With ROI jitter augmentation
python -m src.train.train --config configs/ablation_full_enhanced_b0_224_jitter.yaml

# Higher resolution (320x320)
python -m src.train.train --config configs/ablation_full_enhanced_b0_320.yaml
```

### Model Size Scaling

```bash
# EfficientNet-B1
python -m src.train.train --config configs/ablation_full_enhanced_b1.yaml

# EfficientNet-B2
python -m src.train.train --config configs/ablation_full_enhanced_b2.yaml

# EfficientNet-B3
python -m src.train.train --config configs/ablation_full_enhanced_b3.yaml
```

### Alternative Architectures

```bash
# ConvNeXt-Tiny
python -m src.train.train --config configs/convnext_tiny.yaml

# Swin Transformer Tiny
python -m src.train.train --config configs/swin_tiny.yaml

# RegNetY-8GF
python -m src.train.train --config configs/regnety_8gf.yaml

# Or run all lightweight models
python scripts/training/run_lightweight_models.py --model all
```

### Baseline Models

Train traditional ML baselines using frozen features:

```bash
python scripts/training/run_baselines.py --config configs/default.yaml
```

This trains Logistic Regression on frozen EfficientNet-B0 features.

### Ordinal Regression

```bash
# Standard ordinal regression
python -m src.train.train --config configs/ordinal_b0_224_jitter.yaml

# With weighted thresholds (emphasizes underweight detection)
python -m src.train.train --config configs/ordinal_b0_224_jitter_weighted.yaml
```

### ROI Robustness Experiments

```bash
# 5% fixed padding
python -m src.train.train --config configs/roi_robustness_padding_05.yaml

# 10% fixed padding
python -m src.train.train --config configs/roi_robustness_padding_10.yaml
```

## Evaluation

### Compare All Experiments

```bash
python scripts/evaluation/compare_ablation.py
```

This generates a comparison table of all experiments with metrics.

### Evaluate on Test Set

After selecting your best model:

```bash
python scripts/training/evaluate_test.py \
  --checkpoint outputs/ablation_full/best_model.pt \
  --config outputs/ablation_full/config.yaml \
  --test-csv data/test.csv
```

### Ensemble Evaluation

#### 2-Model Ensemble

```bash
python scripts/evaluation/evaluate_ensemble_general.py \
  --model1-checkpoint outputs/ablation_full_enhanced_b0_224_jitter/best_model.pt \
  --model1-config outputs/ablation_full_enhanced_b0_224_jitter/config.yaml \
  --model2-checkpoint outputs/ablation_full_enhanced_b2/best_model.pt \
  --model2-config outputs/ablation_full_enhanced_b2/config.yaml \
  --ensemble-method weighted \
  --split val
```

#### Multi-Model Ensemble (3+ models)

```bash
python scripts/evaluation/evaluate_ensemble_multiway.py \
  --checkpoints \
    outputs/ablation_full_enhanced_b0_224_jitter/best_model.pt \
    outputs/ablation_full_enhanced_b1/best_model.pt \
    outputs/ablation_full_enhanced_b2/best_model.pt \
  --configs \
    outputs/ablation_full_enhanced_b0_224_jitter/config.yaml \
    outputs/ablation_full_enhanced_b1/config.yaml \
    outputs/ablation_full_enhanced_b2/config.yaml \
  --model-types classification classification classification \
  --ensemble-method majority \
  --split val
```

## Project Structure

```
.
├── configs/                    # YAML configuration files
│   ├── default.yaml            # Default training config
│   ├── ablation_*.yaml         # Transfer learning ablation configs
│   ├── ablation_full_enhanced*.yaml  # Enhanced training configs
│   ├── ordinal_*.yaml          # Ordinal regression configs
│   ├── roi_robustness_*.yaml   # ROI cropping experiments
│   └── *.yaml                  # Architecture-specific configs
│
├── data/                       # Data directory
│   ├── manifest.csv            # Full dataset manifest
│   ├── train.csv               # Training split
│   ├── val.csv                 # Validation split
│   ├── test.csv                # Test split
│   └── processed_*/            # Preprocessed images (optional)
│
├── src/                        # Source code
│   ├── models/                 # Model creation utilities
│   │   ├── factory.py          # Model factory with transfer learning
│   │   └── baselines.py         # Baseline model utilities
│   ├── train/                  # Training code
│   │   ├── dataset.py          # BcsDataset class
│   │   ├── train.py             # Main training script
│   │   └── early_stopping.py   # Early stopping callback
│   └── eval/                    # Evaluation metrics
│       └── metrics.py          # Evaluation functions
│
├── scripts/                     # Utility scripts
│   ├── training/                # Training scripts
│   │   ├── run_ablation.sh      # Run all ablation experiments
│   │   ├── run_baselines.py     # Train baseline models
│   │   ├── run_lightweight_models.py  # Train alternative architectures
│   │   └── evaluate_test.py    # Test set evaluation
│   ├── evaluation/              # Evaluation scripts
│   │   ├── compare_ablation.py  # Compare experiment results
│   │   ├── evaluate_ensemble_general.py  # 2-model ensembles
│   │   └── evaluate_ensemble_multiway.py # Multi-model ensembles
│   ├── data/                    # Data preparation
│   │   ├── make_manifest.py     # Create manifest from raw data
│   │   ├── split_manifest_3way.py  # Create train/val/test splits
│   │   └── preprocess_images.py   # Preprocess images (optional)
│   └── analysis/                # Analysis and visualization
│       ├── export_results_latex.py  # Export results to LaTeX
│       ├── plot_learning_curve.py   # Plot training curves
│       ├── generate_test_error_analysis.py  # Error analysis
│       └── advanced/            # Advanced analysis scripts
│
└── outputs/                     # Experiment outputs
    └── <experiment_name>/      # Per-experiment directory
        ├── best_model.pt        # Best model checkpoint
        ├── config.yaml         # Config used (for reproducibility)
        ├── metrics.json        # Detailed evaluation metrics
        └── confusion_matrix.png # Confusion matrix visualization
```

## Configuration

### Transfer Learning Strategies

The codebase supports 4 transfer learning strategies:

1. **scratch**: Train from scratch (no pretrained weights)
2. **head_only**: Freeze backbone, train only classifier head
3. **last_block**: Freeze early layers, unfreeze last block(s)
4. **full**: Fine-tune all parameters (default)

Configure in `configs/*.yaml`:

```yaml
model:
  backbone: efficientnet_b0
  num_classes: 5
  pretrained: true
  finetune_mode: full # Options: head_only, last_block, full, scratch
```

### Data Configuration

```yaml
data:
  train: data/train.csv
  val: data/val.csv
  img_size: 224 # Image size (height/width)
  do_aug: true # Enable data augmentation
  skip_resize: false # Whether images are already resized
  crop_jitter: 0.1 # Random padding around ROI (0.0-1.0)
  crop_padding: 0.05 # Fixed padding around ROI (0.0-1.0)
```

### Training Configuration

```yaml
train:
  batch_size: 32
  epochs: 20
  lr: 3e-4
  seed: 42
  device: cuda # Options: cpu, cuda, mps, or null (auto-detect)
  optimizer: adamw # Options: adamw, adam, sgd
  weight_decay: 1e-4
  scheduler: cosine # Options: cosine, step, or null
  label_smoothing: 0.05
  early_stopping:
    enabled: true
    patience: 5
    monitor: val_macro_f1 # Options: val_acc, val_macro_f1, val_underweight_recall
    mode: max
```

### Evaluation Configuration

```yaml
eval:
  save_confusion_matrix: true
  save_per_class_metrics: true
  class_names: ["3.25", "3.5", "3.75", "4.0", "4.25"]
```

## Metrics

The evaluation tracks comprehensive metrics:

- **Accuracy**: Overall classification accuracy
- **Macro-F1**: Average F1 across all classes (balanced, primary metric)
- **Weighted-F1**: F1 weighted by class frequency
- **Underweight Recall**: Recall for class 0 (BCS 3.25) - critical for welfare
- **Per-class Recall/Precision**: Detailed per-class metrics
- **Confusion Matrix**: Visual classification patterns
- **MAE (BCS)**: Mean absolute error in BCS units (for ordinal models)
- **Ordinal Accuracy**: Accuracy within ±1 class (for ordinal models)

## Experiments Summary

This project includes 18+ training experiments:

### Transfer Learning Ablations (4)

- `ablation_scratch` - Train from scratch
- `ablation_head_only` - Freeze backbone, train head only
- `ablation_last_block` - Freeze early layers, unfreeze last blocks
- `ablation_full` - Full fine-tuning

### Enhanced Training (3)

- `ablation_full_enhanced` - With augmentation + cosine scheduler
- `ablation_full_enhanced_b0_224_jitter` - With ROI jitter
- `ablation_full_enhanced_b0_320` - Higher resolution (320x320)

### Model Size Scaling (3)

- `ablation_full_enhanced_b1` - EfficientNet-B1
- `ablation_full_enhanced_b2` - EfficientNet-B2
- `ablation_full_enhanced_b3` - EfficientNet-B3

### Alternative Architectures (3)

- `convnext_tiny` - ConvNeXt-Tiny
- `swin_tiny` - Swin Transformer Tiny
- `regnety_8gf` - RegNetY-8GF

### Ordinal Regression (2)

- `ordinal_b0_224_jitter` - Standard ordinal regression
- `ordinal_b0_224_jitter_weighted` - With weighted thresholds

### ROI Robustness (2)

- `roi_robustness_padding_05` - 5% fixed padding
- `roi_robustness_padding_10` - 10% fixed padding

### Baselines (1)

- `baseline_logreg` - Logistic Regression on frozen features

### Ensembles (5+)

- Various 2-way, 3-way, 4-way, and 5-way ensembles

## Output Files

Each experiment creates an output directory (`outputs/<experiment_name>/`) with:

- `best_model.pt`: Best model checkpoint (based on validation metric)
- `config.yaml`: Exact config used (for reproducibility)
- `metrics.json`: Detailed evaluation metrics (accuracy, F1, per-class metrics)
- `confusion_matrix.png`: Visualization of confusion matrix
- `learning_curve.png`: Training curves (if generated)

## Analysis and Visualization

### Generate Error Analysis

```bash
python scripts/analysis/generate_test_error_analysis.py \
  --checkpoint outputs/ablation_full/best_model.pt \
  --config outputs/ablation_full/config.yaml
```

### Plot Learning Curves

```bash
python scripts/analysis/plot_learning_curve.py \
  --log logs/training.log \
  --output learning_curve.png
```

## Key Features

### ROI Cropping

The dataset includes bounding box annotations for the region of interest (tailhead). The code supports:

- **Automatic ROI cropping**: Extract tailhead region from full images
- **Jitter augmentation**: Random padding around ROI for robustness
- **Fixed padding**: Consistent padding margin for reproducibility

### Data Augmentation

- Horizontal flip (50% probability)
- Rotation (±10 degrees, 50% probability)
- Normalization (ImageNet statistics)

### Early Stopping

Configurable early stopping based on:

- Validation accuracy
- Validation macro-F1 (default)
- Underweight recall

### Ensemble Methods

- **Majority voting**: Most common prediction wins
- **Weighted voting**: Weight predictions by model performance
- **Consensus**: Only use prediction if enough models agree

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in config:

```yaml
train:
  batch_size: 16 # Reduce from 32
```

### Slow Training on CPU

Training on CPU is very slow. Consider:

- Using a GPU-enabled environment (CUDA or MPS)
- Reducing image size: `img_size: 224` instead of `320`
- Using a smaller model: `efficientnet_b0` instead of `b2` or `b3`

### Missing Dependencies

If you encounter import errors:

```bash
pip install --upgrade -r requirements.txt
```

## Authors

- Paul Creavin - CS229 Machine Learning Project
- Quentin MacFarlane - CS229 Machine Learning Project

## Acknowledgments

- ScienceDB for the dairy BCS dataset
- PyTorch and timm for model architectures
- Stanford CS229 course staff
