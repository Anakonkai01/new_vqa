#!/bin/bash
# =============================
# Model H: Phase 3 Training
# =============================
# Only use explanation data (VQA-E, VQA-X, A-OKVQA), no VQA2.0
# Scheduled sampling enabled
# Assumes you have already run Phase 2 and have a best checkpoint

set -e

PHASE=3
EPOCHS=30
BATCH_SIZE=128
LR=2e-4
PATIENCE=5
SS_K=5

python src/train_h.py \
    --phase $PHASE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --patience $PATIENCE \
    --warmup_epochs 0 \
    --resume checkpoints/h/model_h_phase2_best.pth \
    --vg_feat_dir data/vg_features \
    --use_fasttext \
    --scheduled_sampling \
    --ss_k $SS_K \
    --save_legacy_alias \
    --wandb
