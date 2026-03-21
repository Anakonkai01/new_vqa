#!/bin/bash
# ====================================================================
# Model H: Phase 2 Training Script (Teacher Forcing + Replay Buffer)
# ====================================================================
# PREREQUISITE: Phase 1 must be completed first. This script resumes
# from the best Phase 1 checkpoint: checkpoints/h/model_h_phase1_best.pth
#
# Key changes vs Phase 1:
#   - Lower LR (1e-4): brain is already warmed up, don't disrupt it
#   - Replay Sampler: 80% explanation + 20% VQA v2.0 to prevent catastrophic forgetting
#   - Higher patience (7): Phase 2 val loss fluctuates more due to distribution shift
#   - Warmup disabled: resume from stable weights, warmup not needed
# ====================================================================

eval "$(conda shell.bash hook)"
conda activate d2l

echo "=========================================================="
echo "    MODEL H: PHASE 2 TRAINING STARTING...                "
echo "    Replay Buffer: 80% Explanation + 20% VQA v2.0        "
echo "    Resume from: checkpoints/h/model_h_phase1_best.pth   "
echo "=========================================================="
echo ""

python src/train_h.py \
    --phase 2 \
    --epochs 50 \
    --patience 10 \
    --batch_size 128 \
    --lr 1e-4 \
    --warmup_epochs 0 \
    --resume checkpoints/h/model_h_phase1_best.pth \
    --vg_feat_dir data/vg_features \
    --use_fasttext \
    --save_legacy_alias \
    --wandb
