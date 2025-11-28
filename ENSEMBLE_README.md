# Ensemble Experiments - Quick Start

## What Was Created

1. **`src/models/ordinal_utils.py`**: Minimal ordinal decoding utilities for inference
   - `decode_ordinal()` function with 3 methods: `threshold_count`, `expected_value`, `max_prob`

2. **`scripts/compare_ordinal_decoding.py`**: Compare different decoding methods on ordinal models
   - Tests all 3 decoding methods without retraining
   - Generates comparison tables and confusion matrices

3. **`scripts/evaluate_ensemble.py`**: Evaluate ensemble of baseline + ordinal models
   - Implements override rule: same → use it, ±1 → baseline, >1 → ordinal
   - Reports agreement statistics and improvements

4. **`docs/ENSEMBLE_GUIDE.md`**: Detailed documentation

## Quick Start

### 1. Compare Decoding Methods

```bash
python scripts/compare_ordinal_decoding.py \
    --checkpoint outputs/ordinal_b0_224_jitter_weighted/best_model.pt \
    --config outputs/ordinal_b0_224_jitter_weighted/config.yaml \
    --method all \
    --split val
```

### 2. Evaluate Ensemble

```bash
python scripts/evaluate_ensemble.py \
    --baseline-checkpoint outputs/ablation_full_enhanced_b0_224_jitter/best_model.pt \
    --baseline-config outputs/ablation_full_enhanced_b0_224_jitter/config.yaml \
    --ordinal-checkpoint outputs/ordinal_b0_224_jitter_weighted/best_model.pt \
    --ordinal-config outputs/ordinal_b0_224_jitter_weighted/config.yaml \
    --ordinal-decoding threshold_count \
    --split test
```

## Ensemble Rule

- **Same class**: Both models agree → use the prediction
- **Differ by ±1**: Use **baseline** (better at exact accuracy)
- **Differ by >1**: Use **ordinal** (more distance-aware)

## Next Steps

1. Run decoding comparison to find best decoding method
2. Run ensemble evaluation with best decoding method
3. Compare results: accuracy, underweight recall, MAE
4. Test on test set to see if ensemble improves metrics without killing macro-F1


