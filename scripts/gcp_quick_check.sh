#!/bin/bash
# Quick diagnostic script for GCP

echo "=========================================="
echo "GCP Quick Diagnostic Check"
echo "=========================================="

echo ""
echo "1. Current directory:"
pwd

echo ""
echo "2. Python version:"
python --version

echo ""
echo "3. CUDA availability:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    python -c "import torch; print('CUDA device:', torch.cuda.get_device_name(0))"
fi

echo ""
echo "4. Key packages:"
python -c "import torch; print('torch:', torch.__version__)" 2>/dev/null || echo "torch: NOT FOUND"
python -c "import timm; print('timm:', timm.__version__)" 2>/dev/null || echo "timm: NOT FOUND"

echo ""
echo "5. Config file exists:"
if [ -f "configs/ordinal_b0_224_jitter.yaml" ]; then
    echo "  ✓ configs/ordinal_b0_224_jitter.yaml"
else
    echo "  ✗ configs/ordinal_b0_224_jitter.yaml NOT FOUND"
fi

echo ""
echo "6. Data files:"
if [ -f "data/train.csv" ]; then
    echo "  ✓ data/train.csv ($(wc -l < data/train.csv) lines)"
else
    echo "  ✗ data/train.csv NOT FOUND"
fi
if [ -f "data/val.csv" ]; then
    echo "  ✓ data/val.csv ($(wc -l < data/val.csv) lines)"
else
    echo "  ✗ data/val.csv NOT FOUND"
fi

echo ""
echo "7. Test imports:"
python -c "from src.models import create_model; print('  ✓ Model import OK')" 2>&1 | head -1 || echo "  ✗ Model import FAILED"
python -c "from src.models.losses import create_ordinal_loss; print('  ✓ Loss import OK')" 2>&1 | head -1 || echo "  ✗ Loss import FAILED"

echo ""
echo "8. Run full diagnostic:"
echo "  python scripts/test_ordinal_setup.py"

echo ""
echo "=========================================="

