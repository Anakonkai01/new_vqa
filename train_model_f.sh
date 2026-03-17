#!/usr/bin/env bash
# =============================================================================
# Model F — Full 4-Phase Training Script
# RTX 5070 Ti (16 GB VRAM, Blackwell SM 12.0, BF16 native)
#
# Architecture: BUTD (pre-extracted Faster R-CNN 2048-d) + MHCA+img_mask
#               + MUTAN + LayerNorm-BiLSTM + GloVe 840B
#
# Same flag logic as train_model_e.sh — see comments there for rationale:
#   --mix_vqa    → Phase 1 ONLY
#   --curriculum → Phases 2-4 ONLY
#   --focal      → Phases 1-3 ONLY (disabled in Phase 4 SCST)
#
# Prerequisites:
#   python src/scripts/extract_butd_features.py \
#       --image_dir data/raw --output_dir data/butd_features \
#       --splits train2014 val2014
#   python src/scripts/1_build_vocab.py
#
# Usage:
#   bash train_model_f.sh               # all 4 phases
#   bash train_model_f.sh 1             # Phase 1 only
#   BATCH_SIZE=192 WANDB=1 bash train_model_f.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BATCH_SIZE="${BATCH_SIZE:-192}"      # no CNN in forward pass → higher batch safe
ACCUM_STEPS="${ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-16}"
DROPOUT="${DROPOUT:-0.3}"
WANDB_PROJECT="${WANDB_PROJECT:-vqa-e}"

WANDB_FLAG=""
if [ "${WANDB:-0}" = "1" ]; then
    WANDB_FLAG="--wandb --wandb_project ${WANDB_PROJECT}"
fi

COMMON="
  --model F
  --use_mutan
  --layer_norm
  --dropconnect
  --q_highway
  --char_cnn
  --glove --glove_dim 300
  --coverage --coverage_lambda 0.5
  --augment
  --batch_size ${BATCH_SIZE}
  --accum_steps ${ACCUM_STEPS}
  --num_workers ${NUM_WORKERS}
  --dropout ${DROPOUT}
  --weight_decay 1e-4
  --grad_clip 2.0
  --label_smoothing 0.1
  ${WANDB_FLAG}"

RUN_PHASE="${1:-all}"

# =============================================================================
# Phase 1 — Baseline (15 epochs, LR=1e-3)
# mix_vqa: ON | curriculum: OFF (blocked) | focal: ON
# =============================================================================
run_phase1() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  PHASE 1 — Baseline (15 ep, lr=1e-3)"
    echo "  mix_vqa: ON | curriculum: OFF (blocked) | focal: ON"
    echo "  Batch: ${BATCH_SIZE}×${ACCUM_STEPS} = $((BATCH_SIZE * ACCUM_STEPS)) effective"
    echo "════════════════════════════════════════════════════════════"
    # shellcheck disable=SC2086
    python src/train.py ${COMMON} \
        --epochs 15 \
        --lr 1e-3 \
        --warmup_epochs 3 \
        --mix_vqa --mix_vqa_fraction 0.7 \
        --focal \
        --early_stopping 5 \
        --phase 1 \
        --wandb_run_name "model_f_phase1_b${BATCH_SIZE}" \
        --wandb_tags "modelF,phase1,butd,mix-vqa"
    echo "✓ Phase 1 done → checkpoints/model_f_best.pth"
}

# =============================================================================
# Phase 2 — Extended Training (10 epochs, LR=5e-4)
# No CNN fine-tune for Model F (features are pre-extracted).
# mix_vqa: OFF | curriculum: ON | focal: ON
# =============================================================================
run_phase2() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  PHASE 2 — Extended Training (10 ep, lr=5e-4)"
    echo "  mix_vqa: OFF | curriculum: ON | focal: ON"
    echo "════════════════════════════════════════════════════════════"
    RESUME="checkpoints/model_f_resume.pth"
    if [ ! -f "${RESUME}" ]; then
        echo "ERROR: ${RESUME} not found. Run Phase 1 first."
        exit 1
    fi
    # shellcheck disable=SC2086
    python src/train.py ${COMMON} \
        --epochs 10 \
        --lr 5e-4 \
        --warmup_epochs 0 \
        --curriculum \
        --focal \
        --early_stopping 3 \
        --resume "${RESUME}" \
        --phase 2 \
        --wandb_run_name "model_f_phase2_b${BATCH_SIZE}" \
        --wandb_tags "modelF,phase2,butd,curriculum"
    echo "✓ Phase 2 done → checkpoints/model_f_best.pth"
}

# =============================================================================
# Phase 3 — Scheduled Sampling (7 epochs, LR=2e-4)
# mix_vqa: OFF | curriculum: ON | focal: ON
# =============================================================================
run_phase3() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  PHASE 3 — Scheduled Sampling (7 ep, lr=2e-4)"
    echo "  mix_vqa: OFF | curriculum: ON | focal: ON"
    echo "════════════════════════════════════════════════════════════"
    RESUME="checkpoints/model_f_resume.pth"
    if [ ! -f "${RESUME}" ]; then
        echo "ERROR: ${RESUME} not found. Run Phase 2 first."
        exit 1
    fi
    # shellcheck disable=SC2086
    python src/train.py ${COMMON} \
        --epochs 7 \
        --lr 2e-4 \
        --warmup_epochs 0 \
        --curriculum \
        --focal \
        --scheduled_sampling \
        --ss_k 5 \
        --early_stopping 3 \
        --resume "${RESUME}" \
        --phase 3 \
        --wandb_run_name "model_f_phase3_b${BATCH_SIZE}" \
        --wandb_tags "modelF,phase3,scheduled-sampling,curriculum"
    echo "✓ Phase 3 done → checkpoints/model_f_best.pth"
}

# =============================================================================
# Phase 4 — SCST RL (3 epochs, LR=5e-5)
# mix_vqa: OFF | curriculum: ON | focal: OFF (plain CE for RL stability)
# =============================================================================
run_phase4() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  PHASE 4 — SCST RL (3 ep, lr=5e-5)"
    echo "  mix_vqa: OFF | curriculum: ON | focal: OFF (plain CE)"
    echo "════════════════════════════════════════════════════════════"
    RESUME="checkpoints/model_f_resume.pth"
    if [ ! -f "${RESUME}" ]; then
        echo "ERROR: ${RESUME} not found. Run Phase 3 first."
        exit 1
    fi
    # shellcheck disable=SC2086
    python src/train.py ${COMMON} \
        --epochs 3 \
        --lr 5e-5 \
        --warmup_epochs 0 \
        --curriculum \
        --scst \
        --scst_lambda 0.5 \
        --scst_bleu_weight 0.5 \
        --scst_meteor_weight 0.5 \
        --scst_length_bonus 0.0 \
        --resume "${RESUME}" \
        --phase 4 \
        --wandb_run_name "model_f_phase4_b${BATCH_SIZE}" \
        --wandb_tags "modelF,phase4,scst-rl,curriculum"
    echo "✓ Phase 4 done → checkpoints/model_f_best.pth"
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
echo ""
echo "VQA-E Model F (BUTD) — RTX 5070 Ti (Blackwell, BF16)"
echo "Batch: ${BATCH_SIZE} | Workers: ${NUM_WORKERS} | Dropout: ${DROPOUT}"
echo "Phase schedule: 15 + 10 + 7 + 3 = 35 total epochs"

case "${RUN_PHASE}" in
    1)    run_phase1 ;;
    2)    run_phase2 ;;
    3)    run_phase3 ;;
    4)    run_phase4 ;;
    all)
        run_phase1
        run_phase2
        run_phase3
        run_phase4
        echo ""
        echo "══════════════════════════════════════════════"
        echo "  All 4 phases complete (35 epochs total)"
        echo "  Best checkpoint: checkpoints/model_f_best.pth"
        echo "══════════════════════════════════════════════"
        ;;
    *)
        echo "Usage: bash train_model_f.sh [1|2|3|4|all]"
        exit 1
        ;;
esac
