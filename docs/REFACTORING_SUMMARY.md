# Code Refactoring Summary

This document summarizes the code cleanup and enhancements made to prepare for the transfer learning ablation study.

## What Was Changed

### 1. **New Module Structure**

#### `src/models/factory.py`
- Centralized model creation with support for 4 transfer learning strategies:
  - `head_only`: Freeze backbone, train only classifier head
  - `last_block`: Freeze early layers, unfreeze last block(s)
  - `full`: Fine-tune all parameters (default)
  - `scratch`: Train from scratch (no pretrained weights)
- Automatically handles architecture-specific layer freezing
- Prints trainable parameter count for debugging

#### `src/eval/metrics.py`
- Comprehensive evaluation with multiple metrics:
  - Accuracy
  - Macro-F1 score
  - Weighted F1 score
  - Per-class recall and precision
  - Underweight recall (class 0)
  - Confusion matrix generation and visualization

#### `src/train/early_stopping.py`
- Early stopping utility to prevent overfitting
- Configurable patience and monitoring metric
- Supports monitoring: `val_acc`, `val_macro_f1`, `val_underweight_recall`

### 2. **Enhanced Training Script** (`src/train/train.py`)

**Improvements:**
- Clean import organization
- Uses model factory instead of hardcoded timm calls
- Comprehensive evaluation with all metrics
- Early stopping integration
- Configurable optimizer (AdamW, Adam, SGD)
- Optional learning rate scheduler (cosine, step)
- Saves confusion matrix as PNG
- Saves detailed metrics as JSON
- Better logging and progress reporting

**Removed:**
- Hardcoded `pretrained=True`
- Simple accuracy-only evaluation
- In-function imports (moved to top)

### 3. **Updated Configuration System**

#### `configs/default.yaml`
- Added `model.pretrained` and `model.finetune_mode`
- Added `train.optimizer`, `train.weight_decay`, `train.scheduler`
- Added `train.early_stopping` section
- Added `eval` section for evaluation settings
- Increased default epochs to 20 (for early stopping)

#### New Ablation Config Files
- `configs/ablation_scratch.yaml`
- `configs/ablation_head_only.yaml`
- `configs/ablation_last_block.yaml`
- `configs/ablation_full.yaml`

### 4. **New Utility Scripts**

#### `scripts/run_ablation.sh`
- Bash script to run all ablation experiments sequentially
- Automatically runs all 4 transfer learning strategies

#### `scripts/compare_ablation.py`
- Compares results from all ablation experiments
- Creates CSV comparison table
- Prints formatted results with best performance highlighted
- Identifies experiments automatically from `outputs/` directory

### 5. **Code Cleanup**

- Removed accidental files (`eval "$(ssh-agent -s)"` files)
- Added proper docstrings
- Organized imports at top of files
- Added `seaborn` to `requirements.txt` (used for confusion matrix plots)

## How to Use

### Running a Single Experiment

```bash
python -m src.train.train --config configs/default.yaml
```

Or with a specific ablation config:

```bash
python -m src.train.train --config configs/ablation_full.yaml
```

### Running All Ablation Experiments

```bash
bash scripts/run_ablation.sh
```

Or run them individually:

```bash
python -m src.train.train --config configs/ablation_scratch.yaml
python -m src.train.train --config configs/ablation_head_only.yaml
python -m src.train.train --config configs/ablation_last_block.yaml
python -m src.train.train --config configs/ablation_full.yaml
```

### Comparing Results

After running experiments, compare results:

```bash
python scripts/compare_ablation.py
```

This will:
- Find all experiment directories in `outputs/`
- Load metrics from each
- Create a comparison CSV file
- Print a formatted comparison table
- Highlight best results

### Output Structure

Each experiment creates a directory in `outputs/` with:
- `best_model.pt`: Best model checkpoint (saved based on macro-F1)
- `config.yaml`: Copy of config used (for reproducibility)
- `metrics.json`: Detailed metrics (accuracy, F1, per-class recall, etc.)
- `confusion_matrix.png`: Visualization of confusion matrix

### Configuring Experiments

Key config options for transfer learning ablation:

```yaml
model:
  pretrained: true          # Use ImageNet weights
  finetune_mode: full       # Options: head_only, last_block, full, scratch

train:
  early_stopping:
    enabled: true
    patience: 5
    monitor: val_macro_f1   # Stop when this metric stops improving
```

## Next Steps

1. **Run the ablation study**: Use `scripts/run_ablation.sh` to compare all strategies
2. **Analyze results**: Use `scripts/compare_ablation.py` to compare metrics
3. **Choose best strategy**: Based on macro-F1 and underweight recall (for welfare)
4. **Optional enhancements**:
   - Add data augmentation (`do_aug: true` in config)
   - Experiment with different learning rates
   - Add Grad-CAM visualization (planned)

## Metrics Focus

Based on your project goals:
- **Macro-F1**: Overall class balance
- **Underweight Recall**: Critical for welfare (class 0 = BCS 3.25)
- **Accuracy**: General performance
- **Confusion Matrix**: Visualize class confusion patterns

The training script automatically tracks all these metrics and saves them for comparison.

