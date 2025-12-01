#!/bin/bash
# Script to run lightweight models on GCP

# Ensure we are in the project root
cd "$(dirname "$0")/.."

echo "Starting training for lightweight models..."

# 1. ConvNeXt Tiny
echo "Running ConvNeXt Tiny..."
python -m src.train.train --config configs/convnext_tiny.yaml

# 2. RegNetY 8GF
echo "Running RegNetY 8GF..."
python -m src.train.train --config configs/regnety_8gf.yaml

# 3. Swin Tiny
echo "Running Swin Tiny..."
python -m src.train.train --config configs/swin_tiny.yaml

echo "All training runs completed!"
