# Ordinal Regression - Quick Start Guide

## ✅ Implementation Complete

All ordinal regression (CORAL) components have been implemented:

1. ✅ Ordinal head module (`src/models/heads/ordinal_head.py`)
2. ✅ Ordinal loss function (`src/models/losses/ordinal_loss.py`)
3. ✅ Updated model factory to support ordinal heads
4. ✅ Updated training script for ordinal models
5. ✅ Updated evaluation with ordinal-specific metrics
6. ✅ Config files for experiments
7. ✅ Documentation

## 🚀 What to Run on Google Cloud

### Step 1: Set Up Google Cloud VM

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

# SSH into VM
gcloud compute ssh ordinal-training --zone=us-central1-a
```

### Step 2: Set Up Environment on VM

```bash
# Clone repo (or upload files)
git clone <your-repo-url>
cd bcs-project

# Install dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Step 3: Run Ordinal Experiments

**Experiment 1: Ordinal with preprocessed images**
```bash
python -m src.train.train --config configs/ordinal_b0_224.yaml
```

**Experiment 2: Ordinal with ROI jitter (matches best baseline)**
```bash
python -m src.train.train --config configs/ordinal_b0_224_jitter.yaml
```

Both experiments will:
- Train for up to 20 epochs with early stopping
- Save best model to `outputs/ordinal_*/best_model.pt`
- Generate metrics including ordinal-specific ones (MAE, ordinal accuracy)
- Save confusion matrix for analysis

### Step 4: Compare Results

After training completes, compare with baseline:

```bash
python scripts/compare_ablation.py
```

This will show you how ordinal models compare to your best baseline (96.25% accuracy).

## 📊 Expected Results

Based on your baseline performance:

- **Baseline (classification)**: 96.25% accuracy, ~227 adjacent-class errors
- **Expected ordinal**: Similar or slightly better accuracy, **30-50% fewer adjacent-class errors**

Key improvements to look for:
- Reduced confusion between adjacent classes (3.25↔3.5, 3.5↔3.75, etc.)
- Fewer non-adjacent errors (jump errors)
- Better MAE in class space
- High ordinal accuracy (within ±1 class should be >98%)

## 📝 Config File Details

### Key Settings in Config Files

```yaml
model:
  head_type: ordinal  # This enables ordinal regression

train:
  device: cuda  # For Google Cloud
  # ordinal_threshold_weights: [2.0, 1.0, 1.0, 1.0]  # Optional: uncomment to weight underweight threshold
```

### Two Config Files Created

1. **`configs/ordinal_b0_224.yaml`**
   - Uses preprocessed images (`data/processed_224/`)
   - Matches `ablation_full_enhanced` setup
   
2. **`configs/ordinal_b0_224_jitter.yaml`**
   - Uses original images with ROI jitter
   - Matches your **best baseline** setup
   - **Recommended**: Start with this one

## 🔍 After Training: What to Check

1. **Confusion Matrix**: Compare with baseline - should show fewer off-diagonal errors
2. **MAE Metrics**: Check `mae_class` and `mae_bcs` in metrics.json
3. **Ordinal Accuracy**: `ordinal_accuracy_1` should be very high (>98%)
4. **Adjacent Errors**: Manually count adjacent vs non-adjacent errors in confusion matrix

## 📚 Full Documentation

See `docs/ORDINAL_EXPERIMENTS.md` for complete details, troubleshooting, and advanced options.

## 🎯 Next Steps After Training

1. Evaluate on test set:
   ```bash
   python scripts/evaluate_test.py \
       --checkpoint outputs/ordinal_b0_224_jitter/best_model.pt \
       --config configs/ordinal_b0_224_jitter.yaml \
       --test-csv data/test.csv
   ```

2. Generate Grad-CAM visualizations:
   ```bash
   python scripts/visualize_gradcam.py \
       --checkpoint outputs/ordinal_b0_224_jitter/best_model.pt \
       --data-csv data/val.csv \
       --num-samples 20
   ```

3. Create comparison table and write-up

## ⚠️ Notes

- Training time: ~1-2 hours per experiment on GPU
- Both configs use same hyperparameters as best baseline
- Early stopping will stop if no improvement for 5 epochs
- Models save automatically when validation macro-F1 improves

Good luck! 🚀

