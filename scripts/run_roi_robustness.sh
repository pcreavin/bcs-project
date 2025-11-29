#!/bin/bash
# Run ROI robustness experiments: ROI-only vs ROI+context (with padding)

set -e  # Exit on error

echo "=========================================="
echo "Running ROI Robustness Experiments"
echo "=========================================="
echo ""

# Array of config files
configs=(
    "configs/roi_robustness_roi_only.yaml"
    "configs/roi_robustness_padding_05.yaml"
    "configs/roi_robustness_padding_10.yaml"
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
echo "All ROI robustness experiments completed!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  - outputs/roi_robustness_roi_only/"
echo "  - outputs/roi_robustness_padding_05/"
echo "  - outputs/roi_robustness_padding_10/"
echo ""
echo "Compare results to check if model relies on background context."


