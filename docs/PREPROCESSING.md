# Image Preprocessing Guide

## Should You Preprocess?

**YES, if:**
- ✅ You have limited cloud credits (saves 20-30% training time = saves money)
- ✅ You'll run multiple experiments (resizing happens every epoch otherwise)
- ✅ You have disk space (~2-3 GB for preprocessed images)
- ✅ You want faster iteration during hyperparameter tuning

**NO, if:**
- ❌ You're still experimenting with image sizes
- ❌ You have unlimited compute time
- ❌ Disk space is very limited

## Cost-Benefit Analysis

### Current Setup (On-the-fly resizing):
- **45,531 images** × **20 epochs** = **~910k resize operations**
- Each resize: ~2-5ms
- Total resize time: ~30-45 minutes per training run
- **GPU time wasted on I/O/CPU operations**

### With Preprocessing:
- **45,531 images** resized **once** = **~15-20 minutes** (one-time cost)
- Training: **20-30% faster** (no resizing during training)
- **Saves ~$5-10 per experiment run** (on cloud GPU)

### For Your Budget ($150):
If you run 5 experiments:
- **Without preprocessing**: ~$50-75 total
- **With preprocessing**: ~$35-55 total (saves $15-20)
- **Preprocessing cost**: $0 (can run locally/CPU)

## How to Preprocess

### Step 1: Run Preprocessing Script

```bash
# Preprocess all splits (train/val/test)
python scripts/preprocess_images.py
```

This will:
- Process ~45,531 images
- Take ~15-20 minutes (local CPU)
- Save to `data/processed_224/`
- Create new CSV files: `train_processed.csv`, `val_processed.csv`, `test_processed.csv`

### Step 2: Update Config Files

Update your config files to use preprocessed images:

```yaml
# configs/default.yaml (or ablation configs)
data:
  train: data/processed_224/train_processed.csv  # Changed from data/train.csv
  val: data/processed_224/val_processed.csv      # Changed from data/val.csv
  img_size: 224
  do_aug: false
```

### Step 3: Update Dataset (Optional)

The dataset will automatically skip resizing if images are already 224×224. But you can also update it to skip the resize step entirely for preprocessed images.

## Storage Requirements

- **Preprocessed images**: ~2-3 GB
  - Each 224×224 JPEG: ~20-50 KB
  - 45,531 images × 30 KB avg = ~1.4 GB

- **Original images**: Keep them (you might want to reprocess later)

## Performance Improvement

### Expected Speedup:
- **Training time**: 20-30% faster
- **GPU utilization**: Higher (less CPU bottleneck)
- **DataLoader speed**: 2-3× faster

### Example:
- **Before**: 1.5 hours per experiment
- **After**: 1.0-1.2 hours per experiment
- **Savings**: 20-30 minutes per experiment

## Reverting to Original

If you want to go back to original images:

```yaml
data:
  train: data/train.csv  # Use original CSV
  val: data/val.csv
```

The dataset will automatically resize on-the-fly again.

## Recommendations

**For your milestone work:**
1. ✅ **Preprocess images** (one-time, saves money)
2. ✅ Use preprocessed images for all ablation experiments
3. ✅ Keep original images for flexibility

**Preprocessing command:**
```bash
python scripts/preprocess_images.py
```

This is a **one-time operation** that will save you time and money during training.




