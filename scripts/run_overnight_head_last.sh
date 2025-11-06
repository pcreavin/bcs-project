#!/bin/bash
# Run ablation_head_only and ablation_last_block experiments overnight
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

# Run first experiment: ablation_head_only
echo "=========================================="
echo "Starting: ablation_head_only"
echo "=========================================="
python -m src.train.train --config configs/ablation_head_only.yaml > "logs/head_only_${TIMESTAMP}.log" 2>&1
HEAD_EXIT_CODE=$?

if [ $HEAD_EXIT_CODE -eq 0 ]; then
    echo "✓ ablation_head_only completed successfully"
else
    echo "✗ ablation_head_only failed with exit code $HEAD_EXIT_CODE"
    echo "Check logs/head_only_${TIMESTAMP}.log for details"
fi

echo ""
echo "Head_only experiment finished at: $(date)"
echo ""

# Run second experiment: ablation_last_block
echo "=========================================="
echo "Starting: ablation_last_block"
echo "=========================================="
python -m src.train.train --config configs/ablation_last_block.yaml > "logs/last_block_${TIMESTAMP}.log" 2>&1
LAST_EXIT_CODE=$?

if [ $LAST_EXIT_CODE -eq 0 ]; then
    echo "✓ ablation_last_block completed successfully"
else
    echo "✗ ablation_last_block failed with exit code $LAST_EXIT_CODE"
    echo "Check logs/last_block_${TIMESTAMP}.log for details"
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
echo "  ablation_head_only: $([ $HEAD_EXIT_CODE -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
echo "  ablation_last_block: $([ $LAST_EXIT_CODE -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
echo ""
echo "Logs saved to:"
echo "  - logs/head_only_${TIMESTAMP}.log"
echo "  - logs/last_block_${TIMESTAMP}.log"
echo ""
echo "Results saved to:"
echo "  - outputs/ablation_head_only/"
echo "  - outputs/ablation_last_block/"
echo ""

# Exit with error if any experiment failed
if [ $HEAD_EXIT_CODE -ne 0 ] || [ $LAST_EXIT_CODE -ne 0 ]; then
    exit 1
fi

exit 0

