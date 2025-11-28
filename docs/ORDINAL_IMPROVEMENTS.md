# Ordinal Regression Improvements Guide

## What We've Implemented

### 1. Threshold Weighting
Weight specific thresholds more heavily in the loss function. This is useful for prioritizing certain class boundaries.

**Example:** Weight the first threshold (underweight detection) 2x more:
```yaml
train:
  ordinal_threshold_weights: [2.0, 1.0, 1.0, 1.0]
```

This gives more importance to correctly learning "BCS ≥ 3.5?" threshold.

### 2. Configurable Decoding Methods
Three decoding strategies available:

1. **`threshold_count`** (default): Standard CORAL
   - Count how many thresholds have probability ≥ 0.5
   - Most common approach

2. **`expected_value`**: Expected class index
   - Sum of threshold probabilities gives expected class
   - May provide smoother predictions

3. **`max_prob`**: Maximum probability threshold
   - Uses the threshold with highest probability
   - Alternative approach

## New Config Files

### `configs/ordinal_b0_224_jitter_weighted.yaml`
- Same as baseline but with threshold weights `[2.0, 1.0, 1.0, 1.0]`
- Expected to improve underweight recall

### `configs/ordinal_b0_224_jitter_expected_value.yaml`
- Uses expected_value decoding instead of threshold_count
- May improve overall accuracy

## Running Experiments

### 1. Train with Threshold Weighting
```bash
python -m src.train.train --config configs/ordinal_b0_224_jitter_weighted.yaml
```

### 2. Train with Expected Value Decoding
```bash
python -m src.train.train --config configs/ordinal_b0_224_jitter_expected_value.yaml
```

### 3. Compare Decoding Methods (No Retraining Needed)
Test different decodings on an existing model:

```bash
python scripts/compare_ordinal_decoding.py \
    --checkpoint outputs/ordinal_b0_224_jitter/ordinal_b0_224_jitter/best_model.pt \
    --config outputs/ordinal_b0_224_jitter/ordinal_b0_224_jitter/config.yaml
```

This will:
- Test all 3 decoding methods on the same trained model
- Compare accuracy, F1, underweight recall, MAE, etc.
- Save results to CSV

## Expected Improvements

### Threshold Weighting
- **Goal**: Improve underweight recall (currently 91.50%)
- **Expected**: +2-4% underweight recall
- **Trade-off**: Slight change in other metrics

### Expected Value Decoding
- **Goal**: Improve overall accuracy
- **Expected**: +0.5-1.5% accuracy
- **Benefit**: Smoother predictions, better calibration

## Next Steps After Training

1. **Compare all ordinal variants:**
   ```bash
   python scripts/compare_ablation.py
   ```

2. **Compare decoding methods** (if you trained with threshold_count):
   ```bash
   python scripts/compare_ordinal_decoding.py \
       --checkpoint outputs/ordinal_b0_224_jitter_weighted/.../best_model.pt \
       --config outputs/ordinal_b0_224_jitter_weighted/.../config.yaml
   ```

3. **Evaluate best model on test set:**
   ```bash
   python scripts/evaluate_test.py \
       --checkpoint <best_model_path> \
       --config <best_config_path> \
       --test-csv data/test.csv
   ```

## Configuration Reference

### Threshold Weights
```yaml
train:
  ordinal_threshold_weights: [w0, w1, w2, w3]
  # Weights for thresholds:
  # w0: "BCS ≥ 3.5?" (underweight detection)
  # w1: "BCS ≥ 3.75?"
  # w2: "BCS ≥ 4.0?"
  # w3: "BCS ≥ 4.25?"
```

Common patterns:
- `[2.0, 1.0, 1.0, 1.0]` - Prioritize underweight detection
- `[1.5, 1.0, 1.0, 1.5]` - Prioritize extremes (underweight + overweight)
- `[1.0, 1.0, 1.0, 1.0]` - Equal weights (default)

### Decoding Methods
```yaml
eval:
  ordinal_decoding_method: threshold_count  # or expected_value, max_prob
```

## Tips

1. **Start with threshold weighting** - easiest win for underweight recall
2. **Try decoding methods on existing models** - no retraining needed!
3. **Compare confusion matrices** - see which reduces adjacent errors better
4. **Check ordinal accuracy** - should stay >99% within ±1 class

