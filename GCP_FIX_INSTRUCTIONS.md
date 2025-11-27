# Quick Fix for GCP Import Error

## The Problem
You're getting:
```
ImportError: cannot import name 'create_ordinal_loss' from 'src.models.losses'
```

This means the `__init__.py` file on GCP is missing the export.

## Quick Fix (Option 1 - Run Script)

On your GCP instance, run:

```bash
cd ~/bcs-project
python scripts/fix_gcp_imports.py
```

This will automatically fix the `__init__.py` file.

## Quick Fix (Option 2 - Manual Edit)

Or manually fix it:

```bash
cd ~/bcs-project
cat > src/models/losses/__init__.py << 'EOF'
"""Loss functions for training."""
from .ordinal_loss import OrdinalLoss, create_ordinal_loss

__all__ = ["OrdinalLoss", "create_ordinal_loss"]
EOF
```

## Quick Fix (Option 3 - Copy from Local)

If you have the files locally, you can upload just this file:

```bash
# From your local machine
scp src/models/losses/__init__.py paulcreavin21@<GCP-IP>:~/bcs-project/src/models/losses/__init__.py

# Or using gcloud
gcloud compute scp src/models/losses/__init__.py ordinal-training:~/bcs-project/src/models/losses/__init__.py --zone=us-central1-a
```

## Verify Fix

After fixing, test the import:

```bash
python -c "from src.models.losses import create_ordinal_loss; print('Import successful!')"
```

If that works, then run training again:

```bash
python -m src.train.train --config configs/ordinal_b0_224_jitter.yaml
```

