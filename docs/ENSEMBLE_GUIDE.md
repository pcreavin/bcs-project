# Ensemble Evaluation Guide

This guide explains how to run the ensemble experiments comparing classification baseline + ordinal regression models.

## Overview

The ensemble approach combines predictions from:
1. **Classification baseline**: Standard softmax classifier (better at exact class accuracy)
2. **Ordinal regression**: CORAL-style ordinal model (better at distance-aware predictions)

### Ensemble Rule

The ensemble uses an override strategy:
- **Same prediction**: Both models agree → use the common prediction
- **Differ by ±1 class**: Models disagree by 1 class → use **baseline** (it's better at 0-1 loss)
- **Differ by >1 class**: Models disagree by >1 class → use **ordinal** (more distance-aware)

## Scripts

### 1. Compare Ordinal Decoding Methods

Test different decoding strategies on a trained ordinal model without retraining:

```bash
python scripts/compare_ordinal_decoding.py \
    --checkpoint outputs/ordinal_b0_224_jitter_weighted/best_model.pt \
    --config outputs/ordinal_b0_224_jitter_weighted/config.yaml \
    --method all \
    --split val
```

**Options:**
- `--checkpoint`: Path to ordinal model checkpoint
- `--config`: Path to model config YAML
- `--method`: Decoding method (`threshold_count`, `expected_value`, `max_prob`, or `all`)
- `--split`: Dataset split (`train`, `val`, or `test`)
- `--output-dir`: Optional output directory (default: checkpoint directory)

**Output:**
- Comparison table of metrics for each decoding method
- Confusion matrices for each method
- JSON file with detailed results

### 2. Evaluate Ensemble

Combine baseline + ordinal models using the override rule:

```bash
python scripts/evaluate_ensemble.py \
    --baseline-checkpoint outputs/ablation_full_enhanced_b0_224_jitter/best_model.pt \
    --baseline-config outputs/ablation_full_enhanced_b0_224_jitter/config.yaml \
    --ordinal-checkpoint outputs/ordinal_b0_224_jitter_weighted/best_model.pt \
    --ordinal-config outputs/ordinal_b0_224_jitter_weighted/config.yaml \
    --ordinal-decoding threshold_count \
    --split test
```

**Options:**
- `--baseline-checkpoint`: Path to classification baseline checkpoint
- `--baseline-config`: Path to baseline config YAML
- `--ordinal-checkpoint`: Path to ordinal model checkpoint
- `--ordinal-config`: Path to ordinal config YAML
- `--ordinal-decoding`: Decoding method for ordinal model (`threshold_count`, `expected_value`, `max_prob`)
- `--split`: Dataset split (`train`, `val`, or `test`)
- `--output-dir`: Optional output directory
- `--device`: Optional device override (`cpu`, `cuda`, `mps`)

**Output:**
- Agreement statistics (same predictions, ±1 differences, >1 differences)
- Metrics comparison table (Classification vs Ordinal vs Ensemble)
- Improvement metrics over baseline
- Confusion matrices for all three approaches
- JSON file with detailed results

## Example Workflow

### Step 1: Test Different Decoding Methods

First, find the best decoding method for your ordinal model:

```bash
python scripts/compare_ordinal_decoding.py \
    --checkpoint outputs/ordinal_b0_224_jitter_weighted/best_model.pt \
    --config outputs/ordinal_b0_224_jitter_weighted/config.yaml \
    --method all \
    --split val
```

Look for the method with best:
- Underweight recall
- Overall accuracy
- Ordinal accuracy (±1 class)

### Step 2: Evaluate Ensemble

Then, evaluate the ensemble with the best decoding method:

```bash
python scripts/evaluate_ensemble.py \
    --baseline-checkpoint outputs/ablation_full_enhanced_b0_224_jitter/best_model.pt \
    --baseline-config outputs/ablation_full_enhanced_b0_224_jitter/config.yaml \
    --ordinal-checkpoint outputs/ordinal_b0_224_jitter_weighted/best_model.pt \
    --ordinal-config outputs/ordinal_b0_224_jitter_weighted/config.yaml \
    --ordinal-decoding expected_value \
    --split test
```

### Step 3: Compare Results

Compare the ensemble results against:
- Baseline classification model
- Ordinal model alone
- Check for improvements in:
  - Underweight recall
  - Overall accuracy
  - MAE (mean absolute error)

## Expected Improvements

The ensemble should:
- **Maintain or improve accuracy** over baseline (use baseline when models agree or differ by ±1)
- **Improve underweight recall** when ordinal catches cases baseline misses
- **Reduce large errors** (differ by >1) by using ordinal predictions
- **Show ordinal accuracy** improvement while maintaining exact accuracy

## Output Files

Both scripts create output files in the specified directory:

1. **JSON metrics files**: Detailed metrics for all models/methods
2. **Confusion matrices**: PNG files for visual analysis
3. **Comparison tables**: Printed to console and saved in JSON

## Troubleshooting

### Model Loading Issues

If you see errors loading ordinal model checkpoints:
- The scripts try multiple model structures automatically
- Check the checkpoint keys match the expected structure
- Verify the config file matches the training config

### Device Issues

- Scripts auto-detect device (CUDA > MPS > CPU)
- Use `--device` flag to override if needed

### Memory Issues

- Reduce batch size in DataLoader if running out of memory
- Evaluate on smaller splits first (val before test)


