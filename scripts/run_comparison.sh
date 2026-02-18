#!/bin/bash
# Run the full BioGPT vs Pythia comparison pipeline.
#
# Prerequisites:
#   1. Prepared data in data/ (run: python -m biogpt.prepare_data)
#   2. Trained BioGPT checkpoint (run: python -m biogpt.train)
#
# Usage:
#   bash scripts/run_comparison.sh

set -e

echo "============================================"
echo "  BioGPT vs Pythia-160M Comparison Pipeline"
echo "============================================"

# Step 1: Prepare data (if not already done)
if [ ! -f "data/train.bin" ]; then
    echo ""
    echo "Step 1: Preparing data from The Pile..."
    PYTHONUNBUFFERED=1 python -m biogpt.prepare_data \
        --target_tokens 2000000000 \
        --out_dir data
else
    echo ""
    echo "Step 1: Data already prepared (data/train.bin exists)"
fi

# Step 2: Train BioGPT (if not already done)
if [ ! -f "checkpoints/biogpt_final.pt" ]; then
    echo ""
    echo "Step 2: Training BioGPT..."
    PYTHONUNBUFFERED=1 python -m biogpt.train \
        --data_dir data \
        --out_dir checkpoints \
        --target_tokens 2000000000 \
        --n_passes 1
else
    echo ""
    echo "Step 2: BioGPT checkpoint found (checkpoints/biogpt_final.pt)"
fi

# Step 3: Run comparison
echo ""
echo "Step 3: Running head-to-head comparison..."
PYTHONUNBUFFERED=1 python -m biogpt.compare \
    --data_dir data \
    --biogpt_ckpt checkpoints/biogpt_final.pt \
    --output_dir results

echo ""
echo "Done! Results saved to results/full_comparison.json"
