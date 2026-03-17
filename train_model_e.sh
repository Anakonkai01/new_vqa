#!/usr/bin/env bash
# =============================================================================
# Model E — Full 4-Phase Training Script
# ConvNeXt-Base + DCAN + MUTAN + LayerNorm-LSTM + Highway-BiLSTM + GloVe 840B
#
# Usage:
#   bash train_model_e.sh               # run all phases
#   bash train_model_e.sh 1             # run only Phase 1
#   bash train_model_e.sh 2             # run only Phase 2 (requires phase1 checkpoint)
#   bash train_model_e.sh 3             # run only Phase 3
#   bash train_model_e.sh 4             # run only Phase 4 (SCST)
#
# Tracking: add WANDB=1 to enable W&B logging
#   WANDB=1 bash train_model_e.sh
#   WANDB=1 WANDB_PROJECT=my-project bash train_model_e.sh
#
# Hardware notes:
#   12 GB VRAM (e.g. RTX 3080/4080): --batch_size 64 --accum_steps 2
#   24 GB VRAM (e.g. RTX 3090/4090): --batch_size 128
#   40 GB VRAM (A100): --batch_size 256
# =============================================================================

set -e   # exit on any error

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BATCH_SIZE="${BATCH_SIZE:-128}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DROPOUT="${DROPOUT:-0.3}"
WANDB_PROJECT="${WANDB_PROJECT:-vqa-e}"

# W&B flag
WANDB_FLAG=""
if [ "${WANDB:-0}" = "1" ]; then
    WANDB_FLAG="--wandb --wandb_project $WANDB_PROJECT"
fi

# Common flags for all phases
COMMON="--model E \
  --dcan \
  --use_mutan \
  --layer_norm \
  --q_highway \
  --glove --glove_dim 300 \
  --batch_size $BATCH_SIZE \
  --accum_steps $ACCUM_STEPS \
  --num_workers $NUM_WORKERS \
  --dropout $DROPOUT \
  --weight_decay 1e-5 \
  --grad_clip 5.0 \
  --label_smoothing 0.1 \
  --no_compile \
  $WANDB_FLAG"

RUN_PHASE="${1:-all}"

# =============================================================================
# Phase 1 — Baseline (10 epochs, LR warmup, frozen ConvNeXt)
# All new modules trained from scratch; ConvNeXt frozen at ImageNet weights
# =============================================================================
run_phase1() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  PHASE 1 — Baseline (10 epochs, lr=1e-3, ConvNeXt frozen)"
    echo "════════════════════════════════════════════════════════════"
    python src/train.py $COMMON \
        --epochs 10 \
        --lr 1e-3 \
        --warmup_epochs 3 \
        --augment \
        --early_stopping 5 \
        --phase 1 \
        --wandb_run_name "model_e_phase1" \
        --wandb_tags "modelE,phase1,frozen-cnn"
    echo "Phase 1 complete → checkpoints/model_e_best.pth"
}

# =============================================================================
# Phase 2 — Fine-tune CNN (5 epochs, unfreeze ConvNeXt top layers)
# layer3+4 of ConvNeXt unfrozen with 10× smaller LR to avoid forgetting
# =============================================================================
run_phase2() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  PHASE 2 — CNN Fine-tune (5 epochs, lr=5e-4, ConvNeXt unfrozen)"
    echo "════════════════════════════════════════════════════════════"
    RESUME="checkpoints/model_e_resume.pth"
    if [ ! -f "$RESUME" ]; then
        echo "ERROR: $RESUME not found. Run Phase 1 first."
        exit 1
    fi
    python src/train.py $COMMON \
        --epochs 5 \
        --lr 5e-4 \
        --warmup_epochs 0 \
        --finetune_cnn \
        --cnn_lr_factor 0.1 \
        --augment \
        --early_stopping 3 \
        --resume "$RESUME" \
        --phase 2 \
        --wandb_run_name "model_e_phase2" \
        --wandb_tags "modelE,phase2,finetune-cnn"
    echo "Phase 2 complete → checkpoints/model_e_best.pth"
}

# =============================================================================
# Phase 3 — Scheduled Sampling (5 epochs)
# Gradually replace teacher forcing with model's own predictions
# Reduces exposure bias between training and inference
# =============================================================================
run_phase3() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  PHASE 3 — Scheduled Sampling (5 epochs, lr=2e-4)"
    echo "════════════════════════════════════════════════════════════"
    RESUME="checkpoints/model_e_resume.pth"
    if [ ! -f "$RESUME" ]; then
        echo "ERROR: $RESUME not found. Run Phase 2 first."
        exit 1
    fi
    python src/train.py $COMMON \
        --epochs 5 \
        --lr 2e-4 \
        --warmup_epochs 0 \
        --finetune_cnn \
        --cnn_lr_factor 0.1 \
        --scheduled_sampling \
        --ss_k 5 \
        --augment \
        --early_stopping 3 \
        --resume "$RESUME" \
        --phase 3 \
        --wandb_run_name "model_e_phase3" \
        --wandb_tags "modelE,phase3,scheduled-sampling"
    echo "Phase 3 complete → checkpoints/model_e_best.pth"
}

# =============================================================================
# Phase 4 — SCST Reinforcement Learning (3 epochs)
# Mixed CE + REINFORCE — directly optimizes BLEU-4
# Only run after Phase 3 has converged
# =============================================================================
run_phase4() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  PHASE 4 — SCST RL (3 epochs, lr=5e-5, mixed CE+REINFORCE)"
    echo "════════════════════════════════════════════════════════════"
    RESUME="checkpoints/model_e_resume.pth"
    if [ ! -f "$RESUME" ]; then
        echo "ERROR: $RESUME not found. Run Phase 3 first."
        exit 1
    fi
    python src/train.py $COMMON \
        --epochs 3 \
        --lr 5e-5 \
        --warmup_epochs 0 \
        --finetune_cnn \
        --cnn_lr_factor 0.1 \
        --scst \
        --scst_lambda 0.5 \
        --resume "$RESUME" \
        --phase 4 \
        --wandb_run_name "model_e_phase4" \
        --wandb_tags "modelE,phase4,scst-rl"
    echo "Phase 4 complete → checkpoints/model_e_best.pth"
}

# =============================================================================
# Dispatch
# =============================================================================
case "$RUN_PHASE" in
    1)   run_phase1 ;;
    2)   run_phase2 ;;
    3)   run_phase3 ;;
    4)   run_phase4 ;;
    all)
        run_phase1
        run_phase2
        run_phase3
        run_phase4
        echo ""
        echo "All 4 phases complete!"
        echo "Best checkpoint: checkpoints/model_e_best.pth"
        ;;
    *)
        echo "Usage: bash train_model_e.sh [1|2|3|4|all]"
        exit 1
        ;;
esac
