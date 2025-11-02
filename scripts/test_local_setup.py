"""Quick validation script before spending cloud credits."""
import sys
import torch
import yaml
from pathlib import Path

print("=" * 60)
print("LOCAL SETUP VALIDATION")
print("=" * 60)

# 1. Test imports
print("\n1. Testing imports...")
try:
    import timm, albumentations, cv2, pandas, sklearn, matplotlib, seaborn
    print("   ✓ All packages installed")
except ImportError as e:
    print(f"   ✗ Missing package: {e}")
    sys.exit(1)

# 2. Test data files
print("\n2. Testing data files...")
for csv_file in ["data/train.csv", "data/val.csv"]:
    if Path(csv_file).exists():
        import pandas as pd
        df = pd.read_csv(csv_file)
        print(f"   ✓ {csv_file} ({len(df)} samples)")
    else:
        print(f"   ✗ {csv_file} not found")
        sys.exit(1)

# 3. Test configs
print("\n3. Testing config files...")
for config in ["default.yaml", "ablation_full.yaml", "ablation_head_only.yaml",
               "ablation_last_block.yaml", "ablation_scratch.yaml"]:
    try:
        with open(f"configs/{config}") as f:
            yaml.safe_load(f)
        print(f"   ✓ {config}")
    except Exception as e:
        print(f"   ✗ {config}: {e}")
        sys.exit(1)

# 4. Test model creation
print("\n4. Testing model factory...")
try:
    from src.models import create_model
    for mode in ["scratch", "head_only", "last_block", "full"]:
        model = create_model("efficientnet_b0", 5, pretrained=(mode != "scratch"),
                           finetune_mode=mode)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 5)
        print(f"   ✓ {mode} mode")
except Exception as e:
    print(f"   ✗ Model creation failed: {e}")
    sys.exit(1)

# 5. Test dataset
print("\n5. Testing dataset...")
try:
    from src.train.dataset import BcsDataset
    ds = BcsDataset("data/train.csv", img_size=224, train=False, do_aug=False)
    x, y = ds[0]
    print(f"   ✓ Dataset loads (sample shape: {x.shape}, label: {y})")
except Exception as e:
    print(f"   ✗ Dataset failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL CHECKS PASSED - Ready for cloud training!")
print("=" * 60)