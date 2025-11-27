# GCP Training Debugging Guide

If your training script runs and immediately stops on GCP, follow these steps:

## Step 1: Run Diagnostic Script

First, test your setup:

```bash
python scripts/test_ordinal_setup.py
```

This will check:
- ✅ All imports work
- ✅ Config file is valid
- ✅ Data files exist
- ✅ Device (CUDA) is available
- ✅ Model can be created
- ✅ Loss function works
- ✅ Dataset can load

## Step 2: Run Training with Verbose Output

Run training and capture all output:

```bash
python -m src.train.train --config configs/ordinal_b0_224_jitter.yaml 2>&1 | tee training.log
```

This saves all output to `training.log` so you can see where it fails.

## Common Issues and Fixes

### Issue 1: Script Exits Immediately with No Output

**Possible causes:**
- Python path issues
- Import errors
- Config file not found

**Fix:**
```bash
# Check Python path
cd /path/to/bcs-project
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run with explicit path
python -m src.train.train --config configs/ordinal_b0_224_jitter.yaml
```

### Issue 2: "Config file not found"

**Fix:**
```bash
# Check current directory
pwd
# Should be: /path/to/bcs-project

# List config files
ls -la configs/ordinal*.yaml

# If not found, you might need to use absolute path:
python -m src.train.train --config /path/to/bcs-project/configs/ordinal_b0_224_jitter.yaml
```

### Issue 3: "CUDA not available"

**Check:**
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

**Fix:**
- Make sure you're using a GPU instance
- Check GPU drivers are installed
- Verify PyTorch GPU version is installed

### Issue 4: "Train CSV not found"

**Check:**
```bash
ls -la data/train.csv data/val.csv
```

**Fix:**
- Make sure data files are uploaded to GCP
- Check paths in config file match actual file locations
- Use absolute paths if needed

### Issue 5: "ModuleNotFoundError"

**Fix:**
```bash
# Reinstall requirements
pip install -r requirements.txt

# Or install specific missing package
pip install <package-name>
```

### Issue 6: Model Creation Fails

**Check ordinal head:**
```bash
python -c "from src.models.heads import OrdinalHead; print('OK')"
python -c "from src.models.losses import create_ordinal_loss; print('OK')"
```

**Check model factory:**
```bash
python -c "from src.models import create_model; print('OK')"
```

## Step 3: Test with Minimal Config

Create a minimal test config to isolate the issue:

```yaml
# configs/test_minimal.yaml
data:
  train: data/train.csv
  val: data/val.csv
  img_size: 224
  do_aug: false

model:
  backbone: efficientnet_b0
  num_classes: 5
  pretrained: true
  finetune_mode: full
  head_type: ordinal

train:
  batch_size: 8  # Small batch for testing
  epochs: 1      # Just one epoch
  lr: 3e-4
  seed: 42
  num_workers: 0  # Set to 0 to avoid multiprocessing issues
  device: cuda
  optimizer: adamw
  weight_decay: 1e-4
  scheduler: null
  early_stopping:
    enabled: false

eval:
  class_names: ["3.25", "3.5", "3.75", "4.0", "4.25"]

exp_name: test_minimal
```

Then run:
```bash
python -m src.train.train --config configs/test_minimal.yaml
```

## Step 4: Check for Silent Errors

Add explicit error checking:

```python
# At the start of your training script, add:
import sys
sys.stdout.flush()  # Ensure output is flushed
sys.stderr.flush()
```

## Step 5: Use Debugging Mode

Run Python with verbose output:

```bash
python -v -m src.train.train --config configs/ordinal_b0_224_jitter.yaml 2>&1 | tee verbose.log
```

## Quick Debug Checklist

- [ ] Can you run `python scripts/test_ordinal_setup.py` successfully?
- [ ] Are you in the correct directory (`bcs-project/`)?
- [ ] Are data files present (`data/train.csv`, `data/val.csv`)?
- [ ] Is CUDA available (`python -c "import torch; print(torch.cuda.is_available())"`)?
- [ ] Can you import the modules (`python -c "from src.models import create_model"`)?
- [ ] Is the config file valid (`python -c "import yaml; yaml.safe_load(open('configs/ordinal_b0_224_jitter.yaml'))"`)?

## Getting Help

If still stuck, run this and share the output:

```bash
# Create diagnostic report
python scripts/test_ordinal_setup.py > diagnostic_report.txt 2>&1
cat diagnostic_report.txt

# Check Python environment
which python
python --version
pip list | grep -E "torch|timm|albumentations"

# Check file structure
ls -la
ls -la src/models/
ls -la configs/
ls -la data/
```

## Expected Output

When training starts successfully, you should see:

```
============================================================
Training Configuration
============================================================
Config: configs/ordinal_b0_224_jitter.yaml
Device: cuda | Batch size: 32 | Epochs: 20
Backbone: efficientnet_b0 | Finetune mode: full | Head type: ordinal
Pretrained: True | Learning rate: 0.0003
Early stopping: True (patience=5, monitor=val_macro_f1)
Output directory: outputs/ordinal_b0_224_jitter
============================================================

Building datasets...
  Loading train dataset from: data/train.csv
  Loading val dataset from: data/val.csv
Train samples: 37496 | Val samples: 8035

Creating model...
  Backbone: efficientnet_b0, Head type: ordinal, Classes: 5
Model created: 5,288,548/5,288,548 parameters trainable (100.0%)
  Moving model to device: cuda
  ✓ Model created successfully
  Creating ordinal loss...
  ✓ Ordinal loss created

Starting training...
------------------------------------------------------------
```

If you don't see this output, check the error messages above it.

