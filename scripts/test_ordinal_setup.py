"""Test ordinal regression setup before running full training on GCP."""
import sys
import os
import traceback
import torch
import yaml
from pathlib import Path

print("=" * 60)
print("ORDINAL SETUP VALIDATION")
print("=" * 60)

errors = []

# 1. Test imports
print("\n1. Testing imports...")
try:
    import timm, albumentations, cv2, pandas, sklearn, matplotlib, seaborn
    print("   ✓ All base packages installed")
except ImportError as e:
    print(f"   ✗ Missing package: {e}")
    errors.append(f"Import error: {e}")

try:
    from src.models import create_model
    from src.models.losses import create_ordinal_loss
    from src.models.heads import OrdinalHead
    print("   ✓ Ordinal modules import successfully")
except ImportError as e:
    print(f"   ✗ Ordinal module import failed: {e}")
    traceback.print_exc()
    errors.append(f"Ordinal import error: {e}")

# 2. Test config file
print("\n2. Testing ordinal config file...")
config_path = "configs/ordinal_b0_224_jitter.yaml"
try:
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    print(f"   ✓ Config file loaded: {config_path}")
    print(f"     - Backbone: {cfg['model']['backbone']}")
    print(f"     - Head type: {cfg['model'].get('head_type', 'classification')}")
    print(f"     - Device: {cfg['train']['device']}")
except Exception as e:
    print(f"   ✗ Config file error: {e}")
    traceback.print_exc()
    errors.append(f"Config error: {e}")

# 3. Test data files
print("\n3. Testing data files...")
try:
    train_csv = cfg['data']['train']
    val_csv = cfg['data']['val']
    
    if not Path(train_csv).exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")
    if not Path(val_csv).exists():
        raise FileNotFoundError(f"Val CSV not found: {val_csv}")
    
    import pandas as pd
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    print(f"   ✓ Train CSV: {train_csv} ({len(train_df)} samples)")
    print(f"   ✓ Val CSV: {val_csv} ({len(val_df)} samples)")
    
    # Check if first image exists
    if len(train_df) > 0:
        first_img = train_df.iloc[0]['image_path']
        if not Path(first_img).exists():
            print(f"   ⚠ Warning: First image not found: {first_img}")
        else:
            print(f"   ✓ Sample image exists: {first_img}")
except Exception as e:
    print(f"   ✗ Data file error: {e}")
    traceback.print_exc()
    errors.append(f"Data error: {e}")

# 4. Test device
print("\n4. Testing device...")
try:
    device_str = cfg['train']['device']
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available but device='cuda'")
        device = torch.device("cuda")
        print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
    elif device_str == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS not available but device='mps'")
        device = torch.device("mps")
        print(f"   ✓ MPS available")
    else:
        device = torch.device("cpu")
        print(f"   ✓ Using CPU")
except Exception as e:
    print(f"   ✗ Device error: {e}")
    traceback.print_exc()
    errors.append(f"Device error: {e}")

# 5. Test ordinal model creation
print("\n5. Testing ordinal model creation...")
try:
    model_cfg = cfg['model']
    model = create_model(
        backbone=model_cfg['backbone'],
        num_classes=model_cfg['num_classes'],
        pretrained=model_cfg.get('pretrained', True),
        finetune_mode=model_cfg.get('finetune_mode', 'full'),
        head_type=model_cfg.get('head_type', 'classification')
    )
    
    print(f"   ✓ Model created: {model_cfg['backbone']}")
    print(f"     - Head type: {model_cfg.get('head_type', 'classification')}")
    
    # Test forward pass
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
        
    expected_output_size = model_cfg['num_classes'] - 1 if model_cfg.get('head_type') == 'ordinal' else model_cfg['num_classes']
    if output.shape[1] != expected_output_size:
        raise ValueError(f"Expected output size {expected_output_size}, got {output.shape[1]}")
    
    print(f"   ✓ Forward pass successful: output shape {output.shape}")
except Exception as e:
    print(f"   ✗ Model creation/forward pass failed: {e}")
    traceback.print_exc()
    errors.append(f"Model error: {e}")

# 6. Test ordinal loss
print("\n6. Testing ordinal loss...")
try:
    num_classes = cfg['model']['num_classes']
    criterion = create_ordinal_loss(num_classes)
    
    # Test loss computation
    logits = torch.randn(2, num_classes - 1)  # K-1 logits for K classes
    labels = torch.tensor([0, 2])  # Sample labels
    loss = criterion(logits, labels)
    
    print(f"   ✓ Ordinal loss created and computed: {loss.item():.4f}")
except Exception as e:
    print(f"   ✗ Ordinal loss failed: {e}")
    traceback.print_exc()
    errors.append(f"Loss error: {e}")

# 7. Test dataset loading
print("\n7. Testing dataset loading...")
try:
    from src.train.dataset import BcsDataset
    
    train_csv = cfg['data']['train']
    img_size = cfg['data']['img_size']
    
    # Try to load a small dataset
    ds = BcsDataset(train_csv, img_size=img_size, train=False, do_aug=False)
    print(f"   ✓ Dataset created: {len(ds)} samples")
    
    # Try to get one sample
    try:
        x, y = ds[0]
        print(f"   ✓ Sample loaded: shape {x.shape}, label {y}")
    except Exception as e:
        print(f"   ⚠ Warning: Could not load sample: {e}")
        traceback.print_exc()
        errors.append(f"Dataset sample error: {e}")
        
except Exception as e:
    print(f"   ✗ Dataset creation failed: {e}")
    traceback.print_exc()
    errors.append(f"Dataset error: {e}")

# Summary
print("\n" + "=" * 60)
if errors:
    print("✗ VALIDATION FAILED - Found errors:")
    for i, error in enumerate(errors, 1):
        print(f"   {i}. {error}")
    print("=" * 60)
    sys.exit(1)
else:
    print("✓ ALL CHECKS PASSED - Ordinal setup looks good!")
    print("=" * 60)
    print("\nYou can now run:")
    print(f"  python -m src.train.train --config {config_path}")
    sys.exit(0)

