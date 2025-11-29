#!/bin/bash
# Run all pending ensemble experiments on GCP
# Run this from the bcs-project directory on your GCP VM

set -e  # Exit on error

echo "=========================================="
echo "Running Ensemble Experiments on GCP"
echo "=========================================="
echo ""

# Ensure we're in the right directory
cd ~/bcs-project || exit 1

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set device to CUDA (GCP)
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "Priority 1: B0 + B2 Ensemble (Weighted)"
echo "=========================================="
python scripts/evaluate_ensemble_general.py \
  --model1-checkpoint outputs/ablation_full_enhanced_b0_224_jitter/best_model.pt \
  --model1-config outputs/ablation_full_enhanced_b0_224_jitter/config.yaml \
  --model2-checkpoint outputs/ablation_full_enhanced_b2/best_model.pt \
  --model2-config outputs/ablation_full_enhanced_b2/config.yaml \
  --ensemble-method weighted \
  --split val

echo ""
echo "=========================================="
echo "Priority 2: B0 + B1 + B2 Ensemble (Majority)"
echo "=========================================="
python scripts/evaluate_ensemble_multiway.py \
  --checkpoints \
    outputs/ablation_full_enhanced_b0_224_jitter/best_model.pt \
    outputs/ablation_full_enhanced_b1/ablation_full_enhanced_b1/best_model.pt \
    outputs/ablation_full_enhanced_b2/best_model.pt \
  --configs \
    outputs/ablation_full_enhanced_b0_224_jitter/config.yaml \
    outputs/ablation_full_enhanced_b1/ablation_full_enhanced_b1/config.yaml \
    outputs/ablation_full_enhanced_b2/config.yaml \
  --model-types classification classification classification \
  --ensemble-method majority \
  --split val

echo ""
echo "=========================================="
echo "Priority 3: B0 + B1 + B2 + Ordinal Ensemble (Majority)"
echo "=========================================="
python scripts/evaluate_ensemble_multiway.py \
  --checkpoints \
    outputs/ablation_full_enhanced_b0_224_jitter/best_model.pt \
    outputs/ablation_full_enhanced_b1/ablation_full_enhanced_b1/best_model.pt \
    outputs/ablation_full_enhanced_b2/best_model.pt \
    outputs/ordinal_b0_224_jitter_weighted/best_model.pt \
  --configs \
    outputs/ablation_full_enhanced_b0_224_jitter/config.yaml \
    outputs/ablation_full_enhanced_b1/ablation_full_enhanced_b1/config.yaml \
    outputs/ablation_full_enhanced_b2/config.yaml \
    outputs/ordinal_b0_224_jitter_weighted/config.yaml \
  --model-types classification classification classification ordinal \
  --ordinal-decoding threshold_count \
  --ensemble-method majority \
  --split val

echo ""
echo "=========================================="
echo "Priority 4: B0 + B1 + B2 + B3 + Ordinal Ensemble (Majority)"
echo "=========================================="
python scripts/evaluate_ensemble_multiway.py \
  --checkpoints \
    outputs/ablation_full_enhanced_b0_224_jitter/best_model.pt \
    outputs/ablation_full_enhanced_b1/ablation_full_enhanced_b1/best_model.pt \
    outputs/ablation_full_enhanced_b2/best_model.pt \
    outputs/ablation_full_enhanced_b3/best_model.pt \
    outputs/ordinal_b0_224_jitter_weighted/best_model.pt \
  --configs \
    outputs/ablation_full_enhanced_b0_224_jitter/config.yaml \
    outputs/ablation_full_enhanced_b1/ablation_full_enhanced_b1/config.yaml \
    outputs/ablation_full_enhanced_b2/config.yaml \
    outputs/ablation_full_enhanced_b3/config.yaml \
    outputs/ordinal_b0_224_jitter_weighted/config.yaml \
  --model-types classification classification classification classification ordinal \
  --ordinal-decoding threshold_count \
  --ensemble-method majority \
  --split val

echo ""
echo "=========================================="
echo "All ensemble experiments completed!"
echo "=========================================="
echo ""
echo "Results saved in outputs/ directory"
echo "Check ensemble_metrics_val.json files for detailed metrics"

