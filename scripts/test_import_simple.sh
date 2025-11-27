#!/bin/bash
# Simple test to verify imports work

cd ~/bcs-project

echo "Testing import with PYTHONPATH..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python -c "from src.models.losses import create_ordinal_loss; print('✓ Import successful!')"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ All imports working! You can now run training."
else
    echo ""
    echo "✗ Import failed. Check the error above."
    exit 1
fi

