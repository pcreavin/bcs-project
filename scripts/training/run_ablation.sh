#!/bin/bash
# Run all ablation experiments for transfer learning study

set -e  # Exit on error

echo "=========================================="
echo "Running Transfer Learning Ablation Study"
echo "=========================================="
echo ""

# Array of config files
configs=(
    "configs/ablation_scratch.yaml"
    "configs/ablation_head_only.yaml"
    "configs/ablation_last_block.yaml"
    "configs/ablation_full.yaml"
)

# Run each experiment
for config in "${configs[@]}"; do
    if [ ! -f "$config" ]; then
        echo "Warning: Config file $config not found, skipping..."
        continue
    fi
    
    echo "----------------------------------------"
    echo "Running experiment: $config"
    echo "----------------------------------------"
    
    python -m src.train.train --config "$config"
    
    echo ""
done

echo "=========================================="
echo "All ablation experiments completed!"
echo "=========================================="
echo ""
echo "To compare results, run:"
echo "  python scripts/compare_ablation.py"

