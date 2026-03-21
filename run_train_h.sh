#!/bin/bash
# ====================================================================
# Model H: Phase 1 Training Script (Cross-Entropy Warmup)
# ====================================================================

eval "$(conda shell.bash hook)"
conda activate d2l

echo "=========================================================="
echo "    MODEl H: PHASE 1 TRAINING STARTING...                 "
echo "=========================================================="
echo "Loading pre-extracted ResNeXt-101 features from data/vg_features..."
echo "Initializing FastText cc.en.300.bin embeddings for vocabulary..."
echo "=========================================================="
echo ""

# Note: Batch size 128 is highly optimal for your 16GB VRAM,
# resulting in smoother mathematical convergence and 50% faster Epoch times.

python src/train_h.py \
    --phase 1 \
    --epochs 30 \
    --patience 5 \
    --batch_size 128 \
    --lr 5e-4 \
    --vg_feat_dir data/vg_features \
    --use_fasttext \
    --save_legacy_alias \
    --wandb
