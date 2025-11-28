# ROI Robustness Experiments

## Purpose

Address reviewer feedback: "Because you rely on ROI cropping, it may be worth a quick ROI-only vs ROI+context robustness check (or small crop jitter) to ensure the model isn't leaning on background cues."

## Experimental Design

Three experiments comparing different ROI cropping strategies:

1. **ROI-only** (`roi_robustness_roi_only.yaml`)
   - `crop_padding: 0.0`
   - Tight crop to bounding box only
   - **Baseline**: No background context

2. **ROI + 5% context** (`roi_robustness_padding_05.yaml`)
   - `crop_padding: 0.05`
   - Expand ROI by 5% margin on each side
   - Includes some background context

3. **ROI + 10% context** (`roi_robustness_padding_10.yaml`)
   - `crop_padding: 0.10`
   - Expand ROI by 10% margin on each side
   - Includes more background context

## How Padding Works

For a bounding box `(xmin, ymin, xmax, ymax)` with width `w = xmax - xmin` and height `h = ymax - ymin`:

- **Padding amount**: `pad_w = w * crop_padding`, `pad_h = h * crop_padding`
- **Expanded bbox**: 
  - `xmin_new = xmin - pad_w`
  - `xmax_new = xmax + pad_w`
  - `ymin_new = ymin - pad_h`
  - `ymax_new = ymax + pad_h`
- Clamped to image boundaries

## Expected Outcomes

**If model is robust (not relying on background):**
- All three experiments should have **similar accuracy** (~96%)
- Small variations acceptable, but no significant drop

**If model relies on background cues:**
- ROI-only (0.0) will have **lower accuracy** than padded versions
- Padded versions (0.05, 0.10) will show **better accuracy**

**If padding hurts (model needs tight ROI):**
- ROI-only will have **best accuracy**
- Padded versions will show **worse accuracy**

## Running Experiments

### Run all experiments:
```bash
bash scripts/run_roi_robustness.sh
```

### Run individual experiment:
```bash
# ROI-only
python -m src.train.train --config configs/roi_robustness_roi_only.yaml

# ROI + 5% padding
python -m src.train.train --config configs/roi_robustness_padding_05.yaml

# ROI + 10% padding
python -m src.train.train --config configs/roi_robustness_padding_10.yaml
```

## Comparing Results

After running all experiments, compare metrics:

```bash
# Check all metrics files
ls outputs/roi_robustness_*/metrics.json

# Manual comparison or use compare script
python scripts/compare_ablation.py  # (if updated to include ROI experiments)
```

Look for:
- **Accuracy**: Should be stable across padding levels
- **Macro-F1**: Should be similar
- **Underweight Recall**: Critical metric to monitor

## Interpretation

- **Stable performance across all padding levels** → Model is robust, focuses on ROI features
- **Better performance with padding** → Model benefits from background context
- **Worse performance with padding** → Model needs tight ROI, padding adds noise
- **Significant drop in ROI-only** → Model may rely on background cues (concerning)

## Baseline Comparison

Compare against existing baseline:
- `ablation_full_enhanced_b0_224_jitter`: 96.25% accuracy (ROI-only, no padding implemented)

