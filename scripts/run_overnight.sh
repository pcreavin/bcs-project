#!/bin/bash
# Run ablation_full and ablation_scratch experiments overnight
# Uses preprocessed images for faster training

set -e  # Exit on error

echo "=========================================="
echo "OVERNIGHT EXPERIMENT RUN"
echo "=========================================="
echo "Starting: $(date)"
echo ""

# Create logs directory
mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Start caffeinate to prevent sleep (runs for 8 hours)
echo "Starting caffeinate (prevents sleep for 8 hours)..."
caffeinate -d -t 28800 &
CAFFEINATE_PID=$!
echo "Caffeinate PID: $CAFFEINATE_PID"
echo ""

# Run first experiment: ablation_full
echo "=========================================="
echo "Starting: ablation_full"
echo "=========================================="
python -m src.train.train --config configs/ablation_full.yaml > "logs/full_${TIMESTAMP}.log" 2>&1
FULL_EXIT_CODE=$?

if [ $FULL_EXIT_CODE -eq 0 ]; then
    echo "✓ ablation_full completed successfully"
else
    echo "✗ ablation_full failed with exit code $FULL_EXIT_CODE"
    echo "Check logs/full_${TIMESTAMP}.log for details"
fi

echo ""
echo "Full experiment finished at: $(date)"
echo ""

# Run second experiment: ablation_scratch
echo "=========================================="
echo "Starting: ablation_scratch"
echo "=========================================="
python -m src.train.train --config configs/ablation_scratch.yaml > "logs/scratch_${TIMESTAMP}.log" 2>&1
SCRATCH_EXIT_CODE=$?

if [ $SCRATCH_EXIT_CODE -eq 0 ]; then
    echo "✓ ablation_scratch completed successfully"
else
    echo "✗ ablation_scratch failed with exit code $SCRATCH_EXIT_CODE"
    echo "Check logs/scratch_${TIMESTAMP}.log for details"
fi

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=========================================="
echo "Finished: $(date)"
echo ""

# Kill caffeinate
kill $CAFFEINATE_PID 2>/dev/null || true

# Summary
echo "Summary:"
echo "  ablation_full: $([ $FULL_EXIT_CODE -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
echo "  ablation_scratch: $([ $SCRATCH_EXIT_CODE -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
echo ""
echo "Logs saved to:"
echo "  - logs/full_${TIMESTAMP}.log"
echo "  - logs/scratch_${TIMESTAMP}.log"
echo ""
echo "Results saved to:"
echo "  - outputs/ablation_full/"
echo "  - outputs/ablation_scratch/"
echo ""

# Exit with error if any experiment failed
if [ $FULL_EXIT_CODE -ne 0 ] || [ $SCRATCH_EXIT_CODE -ne 0 ]; then
    exit 1
fi

exit 0





