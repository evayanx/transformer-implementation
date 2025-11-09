#!/bin/bash

# Transformer Implementation Training Script
# Usage: ./scripts/run.sh

set -e  # Exit on error

# Set random seed for reproducibility
SEED=42

echo "Setting up environment..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

echo "Creating directories..."
mkdir -p ../data
mkdir -p ../checkpoints
mkdir -p ../results

echo "Starting training with seed $SEED..."
python src/train.py --seed $SEED --batch_size 32 --num_epochs 50 --learning_rate 0.0001

echo "Training completed!"
echo "Results saved to ../results/"
echo "Model checkpoints saved to ../checkpoints/"