#!/bin/bash
# =============================
# Model H: Phase 4 Training
# =============================
# Only use explanation data (VQA-E, VQA-X, A-OKVQA), no VQA2.0
# SCST Reinforcement Learning (Self-Critical Sequence Training)
# Assumes you have already run Phase 3 and have a best checkpoint

set -e

PHASE=4
EPOCHS=5
BATCH_SIZE=128
LR=1e-4
PATIENCE=2

python src/train_h.py \
    --phase $PHASE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --patience $PATIENCE \
    --resume checkpoints/h/model_h_phase3_best.pth \
    --vg_feat_dir data/vg_features \
    --use_fasttext \
    --scst \
    --save_legacy_alias \
    --wandb
