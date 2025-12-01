# Lightweight Model Experiments

This document describes the three lightweight models configured for comparison with EfficientNet-B0.

## Models

### 1. ConvNeXt-Tiny
- **Config**: `configs/convnext_tiny.yaml`
- **Model Name**: `convnext_tiny`
- **Description**: Modern convnet with LayerNorm + large kernels. Often outperforms EfficientNet-B0 on fine-grained tasks while staying lightweight.
- **Parameters**: ~28M (larger than B0 but more efficient)
- **Input Size**: 224×224

### 2. Swin-T Transformer
- **Config**: `configs/swin_tiny.yaml`
- **Model Name**: `swin_tiny_patch4_window7_224`
- **Description**: Hierarchical Vision Transformer. Strong at capturing long-range context, worth testing if compute allows.
- **Parameters**: ~28M
- **Input Size**: 224×224
- **Note**: Vision Transformer architecture - may require different learning rate or training strategy

### 3. RegNetY-8GF
- **Config**: `configs/regnety_8gf.yaml`
- **Model Name**: `regnety_008`
- **Description**: Balanced accuracy/efficiency, includes SE-style channel attention. Good contrast to EfficientNet and often robust.
- **Parameters**: ~39M
- **Input Size**: 224×224

## Running Experiments

### Run a single model:
```bash
python -m src.train.train --config configs/convnext_tiny.yaml
python -m src.train.train --config configs/swin_tiny.yaml
python -m src.train.train --config configs/regnety_8gf.yaml
```

### Run all models (using helper script):
```bash
python scripts/run_lightweight_models.py --model all
```

### Run specific model:
```bash
python scripts/run_lightweight_models.py --model convnext_tiny
python scripts/run_lightweight_models.py --model swin_tiny
python scripts/run_lightweight_models.py --model regnety_8gf
```

## Configuration

All models use the same training configuration:
- **Batch Size**: 32
- **Learning Rate**: 3e-4
- **Optimizer**: AdamW
- **Epochs**: 20 (with early stopping, patience=5)
- **Image Size**: 224×224
- **Transfer Learning**: Full fine-tuning (all parameters trainable)
- **Early Stopping**: Monitors `val_macro_f1` (maximize)

## Expected Outputs

Each experiment will create an output directory:
- `outputs/convnext_tiny/`
- `outputs/swin_tiny/`
- `outputs/regnety_8gf/`

Each directory contains:
- `best_model.pt`: Best model checkpoint
- `config.yaml`: Config used (for reproducibility)
- `metrics.json`: Detailed evaluation metrics
- `confusion_matrix.png`: Visualization of confusion matrix

## Comparison Baseline

Compare results against:
- **Logistic Regression Baseline**: 50.4% accuracy, 51.0% macro-F1
- **EfficientNet-B0**: (run ablation experiments for comparison)

## Notes

- All models verified to exist in `timm` library
- Swin-T is a Vision Transformer - may benefit from different hyperparameters
- RegNetY-8GF is slightly larger but includes attention mechanisms
- ConvNeXt-Tiny uses modern design patterns (LayerNorm, large kernels)








