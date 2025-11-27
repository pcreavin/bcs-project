# Ordinal Regression Experiments Setup Guide

This guide explains how to run ordinal regression experiments using CORAL (Cumulative ORdinal regression with a Logistic link function) on Google Cloud.

## What's Implemented

### 1. Ordinal Head (CORAL)
- Located in `src/models/heads/ordinal_head.py`
- Predicts K-1 threshold logits for K ordered classes
- Supports multiple decoding methods:
  - `threshold_count`: Standard CORAL decoding (default)
  - `expected_value`: Expected class index
  - `max_prob`: Maximum probability threshold

### 2. Ordinal Loss
- Located in `src/models/losses/ordinal_loss.py`
- Binary cross-entropy with logits across all thresholds
- Supports optional threshold weighting (e.g., weight underweight threshold more)

### 3. Updated Training Pipeline
- Model factory supports `head_type: "ordinal"`
- Training script automatically uses ordinal loss for ordinal heads
- Evaluation includes ordinal-specific metrics:
  - MAE in class space
  - MAE in BCS space
  - Ordinal accuracy (within ±1 class)
  - Ordinal accuracy (within ±0.25 BCS)

## Configuration Files

Two ordinal experiment configs are provided:

1. **`configs/ordinal_b0_224.yaml`**
   - EfficientNet-B0
   - 224px images
   - Preprocessed images
   - Data augmentation enabled
   - Cosine scheduler

2. **`configs/ordinal_b0_224_jitter.yaml`**
   - EfficientNet-B0
   - 224px images
   - Original images with ROI cropping + jitter
   - Data augmentation enabled
   - Cosine scheduler
   - Same setup as best baseline model

## Running Experiments on Google Cloud

### Prerequisites

1. **Set up Google Cloud VM:**
   ```bash
   # Create VM with GPU
   gcloud compute instances create ordinal-training \
     --zone=us-central1-a \
     --machine-type=n1-standard-4 \
     --accelerator=type=nvidia-tesla-t4,count=1 \
     --image-family=pytorch-latest-gpu \
     --image-project=deeplearning-platform-release \
     --maintenance-policy=TERMINATE \
     --boot-disk-size=200GB
   ```

2. **SSH into VM:**
   ```bash
   gcloud compute ssh ordinal-training --zone=us-central1-a
   ```

3. **Clone repository and set up environment:**
   ```bash
   # Clone your repo
   git clone <your-repo-url>
   cd bcs-project
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Verify GPU
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Running Ordinal Experiments

#### Experiment 1: Ordinal B0 224 (preprocessed)
```bash
python -m src.train.train --config configs/ordinal_b0_224.yaml
```

This will:
- Train an ordinal regression model
- Use the same preprocessing as `ablation_full_enhanced`
- Save results to `outputs/ordinal_b0_224/`

#### Experiment 2: Ordinal B0 224 with Jitter (best baseline setup)
```bash
python -m src.train.train --config configs/ordinal_b0_224_jitter.yaml
```

This will:
- Train an ordinal regression model
- Use the same setup as your best baseline (`ablation_full_enhanced_b0_224_jitter`)
- Save results to `outputs/ordinal_b0_224_jitter/`

### Monitoring Training

Training output includes:
- Epoch progress with loss
- Validation metrics (accuracy, macro-F1, underweight recall)
- Ordinal-specific metrics (MAE, ordinal accuracy)
- Best model saved based on macro-F1

### Expected Training Time

- **EfficientNet-B0**: ~1-2 hours per experiment (20 epochs with early stopping)
- Actual time depends on GPU and data loading speed

### Outputs

Each experiment creates:
- `outputs/ordinal_*/best_model.pt`: Best model checkpoint
- `outputs/ordinal_*/metrics.json`: Comprehensive metrics including ordinal-specific ones
- `outputs/ordinal_*/confusion_matrix.png`: Confusion matrix visualization
- `outputs/ordinal_*/config.yaml`: Copy of config used

## Comparing Results

After training, compare ordinal vs baseline:

```bash
# Compare all experiments
python scripts/compare_ablation.py

# This will include ordinal experiments in the comparison
```

Key metrics to compare:
- **Accuracy**: Should be similar or slightly better
- **Macro-F1**: Should be similar or slightly better
- **Adjacent-class errors**: Should be reduced (check confusion matrix)
- **MAE**: Lower is better (ordinal-specific)
- **Ordinal accuracy (±1 class)**: Higher is better

## Generating Grad-CAM Visualizations

After training, visualize model attention:

```bash
# Generate Grad-CAM for ordinal model
python scripts/visualize_gradcam.py \
    --checkpoint outputs/ordinal_b0_224_jitter/best_model.pt \
    --data-csv data/val.csv \
    --num-samples 20 \
    --output-dir outputs/ordinal_gradcam
```

Note: Grad-CAM works the same way for ordinal models - it visualizes attention for the predicted class.

## Next Steps After Training

1. **Compare confusion matrices:**
   - Check if adjacent-class errors are reduced
   - Verify non-adjacent errors are minimized

2. **Analyze ordinal metrics:**
   - MAE should be lower than baseline
   - Ordinal accuracy (±1 class) should be high (>98%)

3. **Test set evaluation:**
   ```bash
   python scripts/evaluate_test.py \
       --checkpoint outputs/ordinal_b0_224_jitter/best_model.pt \
       --config configs/ordinal_b0_224_jitter.yaml \
       --test-csv data/test.csv
   ```

4. **Generate final comparison:**
   - Create table comparing baseline vs ordinal
   - Document improvements in adjacent-class accuracy
   - Include Grad-CAM visualizations

## Troubleshooting

### Issue: Model output shape mismatch
- **Cause**: Model is outputting classification logits instead of ordinal logits
- **Fix**: Ensure `head_type: ordinal` is set in config

### Issue: Loss is NaN
- **Cause**: Learning rate too high or numerical instability
- **Fix**: Try lower learning rate (1e-4) or add gradient clipping

### Issue: GPU out of memory
- **Cause**: Batch size too large
- **Fix**: Reduce `batch_size` in config (e.g., 32 → 16)

### Issue: Training slower than expected
- **Cause**: Data loading bottleneck
- **Fix**: Increase `num_workers` or use preprocessed images

## Advanced Options

### Weighted Thresholds

To prioritize underweight detection, uncomment and adjust threshold weights:

```yaml
train:
  ordinal_threshold_weights: [2.0, 1.0, 1.0, 1.0]  # Weight first threshold 2x
```

This gives more weight to the "BCS ≥ 3.5" threshold.

### Custom Decoding Method

To use different decoding in evaluation, modify `src/eval/metrics.py`:

```python
pred = OrdinalHead.decode(outputs, method="expected_value")  # Instead of "threshold_count"
```

## References

- CORAL paper: "Consistent Rank Logits for Ordinal Regression"
- Your baseline: `outputs/ablation_full_enhanced_b0_224_jitter/` (96.25% accuracy)

