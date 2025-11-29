# Ensemble Evaluation Guide

## Quick Overview

Ensemble methods combine predictions from multiple trained models to get better results. Different models make different mistakes, so combining them can cancel out individual errors.

**Key Insight**: Diversity matters more than quantity. 3-4 diverse models often outperform 5 similar models.

## Available Scripts

### 1. `evaluate_ensemble.py` - Classification + Ordinal (2 models)

**Purpose**: Combines a classification model with an ordinal regression model

**When to use**: Testing classification vs ordinal head architectures

### 2. `evaluate_ensemble_general.py` - Any 2 Classification Models

**Purpose**: Combines any two classification models (different sizes, resolutions, etc.)

**When to use**: Testing model size/resolution diversity

### 3. `evaluate_ensemble_multiway.py` - 3, 4, 5+ Models ⭐ **NEW**

**Purpose**: Combines 3, 4, 5, or more models

**When to use**: Testing if more models = better performance

**Ensemble Methods**:

- `majority`: Most common prediction wins (default)
- `weighted`: Weight each model by its performance
- `consensus`: Only use prediction if enough models agree (configurable threshold)

## Recommended Combinations

### Priority 1: 2-Way Ensemble - B0 + B2 (Multi-Scale) ⭐

**Why**: Tests model size + resolution diversity. B2 has best underweight recall (0.9593).

```bash
python scripts/evaluate_ensemble_general.py \
  --model1-checkpoint outputs/ablation_full_enhanced_b0_224_jitter/best_model.pt \
  --model1-config outputs/ablation_full_enhanced_b0_224_jitter/config.yaml \
  --model2-checkpoint outputs/ablation_full_enhanced_b2/best_model.pt \
  --model2-config outputs/ablation_full_enhanced_b2/config.yaml \
  --ensemble-method weighted \
  --split val
```

### Priority 2: 3-Way Ensemble - B0 + B1 + B2

**Why**: Tests if combining different model sizes helps.

```bash
python scripts/evaluate_ensemble_multiway.py \
  --checkpoints \
    outputs/ablation_full_enhanced_b0_224_jitter/best_model.pt \
    outputs/ablation_full_enhanced_b1/ablation_full_enhanced_b1/best_model.pt \
    outputs/ablation_full_enhanced_b2/best_model.pt \
  --configs \
    outputs/ablation_full_enhanced_b0_224_jitter/config.yaml \
    outputs/ablation_full_enhanced_b1/ablation_full_enhanced_b1/config.yaml \
    outputs/ablation_full_enhanced_b2/config.yaml \
  --model-types classification classification classification \
  --ensemble-method majority \
  --split val
```

### Priority 3: 4-Way Ensemble - B0 + B1 + B2 + Ordinal

**Why**: Combines model size diversity with head architecture diversity. Likely best performance.

```bash
python scripts/evaluate_ensemble_multiway.py \
  --checkpoints \
    outputs/ablation_full_enhanced_b0_224_jitter/best_model.pt \
    outputs/ablation_full_enhanced_b1/ablation_full_enhanced_b1/best_model.pt \
    outputs/ablation_full_enhanced_b2/best_model.pt \
    outputs/ordinal_b0_224_jitter_weighted/best_model.pt \
  --configs \
    outputs/ablation_full_enhanced_b0_224_jitter/config.yaml \
    outputs/ablation_full_enhanced_b1/ablation_full_enhanced_b1/config.yaml \
    outputs/ablation_full_enhanced_b2/config.yaml \
    outputs/ordinal_b0_224_jitter_weighted/config.yaml \
  --model-types classification classification classification ordinal \
  --ordinal-decoding threshold_count \
  --ensemble-method majority \
  --split val
```

### Priority 4: 5-Way Ensemble - B0 + B1 + B2 + B3 + Ordinal

**Why**: Tests upper bound - may show diminishing returns.

```bash
python scripts/evaluate_ensemble_multiway.py \
  --checkpoints \
    outputs/ablation_full_enhanced_b0_224_jitter/best_model.pt \
    outputs/ablation_full_enhanced_b1/ablation_full_enhanced_b1/best_model.pt \
    outputs/ablation_full_enhanced_b2/best_model.pt \
    outputs/ablation_full_enhanced_b3/best_model.pt \
    outputs/ordinal_b0_224_jitter_weighted/best_model.pt \
  --configs \
    outputs/ablation_full_enhanced_b0_224_jitter/config.yaml \
    outputs/ablation_full_enhanced_b1/ablation_full_enhanced_b1/config.yaml \
    outputs/ablation_full_enhanced_b2/config.yaml \
    outputs/ablation_full_enhanced_b3/config.yaml \
    outputs/ordinal_b0_224_jitter_weighted/config.yaml \
  --model-types classification classification classification classification ordinal \
  --ordinal-decoding threshold_count \
  --ensemble-method majority \
  --split val
```

## How It Works (Brief)

### 2-Model Ensembles

- **Classification + Ordinal**: Uses rule-based combination (same → use it, diff by 1 → classification, diff >1 → ordinal)
- **Any 2 Models**: Uses majority vote or weighted combination

### Multi-Way Ensembles (3+ models)

- **Majority Vote**: Each model votes, most common prediction wins
- **Weighted Vote**: Better models get more influence (specify `--weights`)
- **Consensus**: Only use prediction if enough models agree (specify `--consensus-threshold`)

## Evaluation on Different Splits

Change `--split` parameter:

- `--split train`: Training set (for debugging)
- `--split val`: Validation set (for model selection) ⭐ **Recommended**
- `--split test`: Test set (final evaluation only - use sparingly!)

## Outputs

Each ensemble evaluation creates:

- `ensemble_metrics_{split}.json`: Full metrics for all models and ensemble
- `confusion_matrix_{model}_{split}.png`: Confusion matrices for each model
- `confusion_matrix_ensemble_{split}.png`: Confusion matrix for ensemble

## Comparing Results

After running ensembles, compare with all models:

```bash
python scripts/compare_ablation.py
```

This automatically includes ensemble results in the comparison table.

## Expected Results

- **2-way**: 0.1-0.5% improvement in accuracy/F1
- **3-way**: 0.2-0.7% improvement
- **4-way**: 0.3-0.9% improvement
- **5-way**: 0.3-1.0% improvement (diminishing returns)

**Key Metrics to Watch**:

- **Accuracy**: Overall correctness
- **Macro-F1**: Balanced performance across classes
- **Underweight Recall**: Critical for welfare (BCS 3.25 detection)

## Tips

1. **Start with 2-way**: Easier to debug, faster to run
2. **Check agreement stats**: If models rarely agree, ensemble may not help
3. **Try different methods**: Majority vs weighted vs consensus
4. **Watch for diminishing returns**: 5 models may not be much better than 3-4
5. **Diversity > Quantity**: 3 diverse models > 5 similar models

## Troubleshooting

**Issue**: Models have different image sizes

- **Solution**: Script uses first model's image size. For best results, use models with same resolution or evaluate separately.

**Issue**: Ordinal model not loading

- **Solution**: Make sure `--model-types` includes "ordinal" and `--ordinal-decoding` is specified

**Issue**: Too many models, script is slow

- **Solution**: Start with 2-3 models, add more if needed. Use `--split val` for faster iteration.

## Quick Reference

| Script                          | Models                 | Best For                          |
| ------------------------------- | ---------------------- | --------------------------------- |
| `evaluate_ensemble.py`          | 2 (class + ordinal)    | Testing head architectures        |
| `evaluate_ensemble_general.py`  | 2 (any classification) | Testing size/resolution diversity |
| `evaluate_ensemble_multiway.py` | 3-10 (any mix)         | Testing multi-model combinations  |
