#!/usr/bin/env bash
# =============================================================================
# Model E — Full 4-Phase Training Script
# RTX 5070 Ti (16 GB VRAM, Blackwell SM 12.0, BF16 native)
#
# Architecture: ConvNeXt-Base + MHCA + MUTAN + LayerNorm-BiLSTM + GloVe 840B
#
# Flag logic per phase (critical — do NOT move to COMMON blindly):
#
#   --mix_vqa    PHASE 1 ONLY.  Mixes 70% VQA v2.0 to widen vocab coverage.
#                               Mutually exclusive with --curriculum (train.py L440-450).
#                               If both present, curriculum is SKIPPED with a warning.
#                               Phases 2-4 must be pure VQA-E for explanation quality.
#
#   --curriculum PHASES 2-4.   Question-type progressive curriculum.
#                               Cannot activate in Phase 1 (blocked by --mix_vqa).
#                               Activates after vocab is established in Phase 1.
#
#   --focal      PHASES 1-3.   SequenceFocalLoss for rare-token emphasis.
#                               NOT used in Phase 4: SCST uses plain CE as the
#                               supervised anchor — focal loss destabilizes RL.
#                               (Also: train.py L517-520 silently disables focal
#                               if --pgn is also active; keep logic explicit here.)
#
# Usage:
#   bash train_model_e.sh               # all 4 phases
#   bash train_model_e.sh 1             # Phase 1 only
#   bash train_model_e.sh 2             # Phase 2 (requires phase 1 checkpoint)
#   bash train_model_e.sh 3             # Phase 3
#   bash train_model_e.sh 4             # Phase 4 SCST
#
# Overrides:
#   BATCH_SIZE=192 bash train_model_e.sh
#   WANDB=1 bash train_model_e.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ──────────────────────────────────────────────────────────────────
BATCH_SIZE="${BATCH_SIZE:-128}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-16}"
DROPOUT="${DROPOUT:-0.3}"
WANDB_PROJECT="${WANDB_PROJECT:-vqa-e}"

WANDB_FLAG=""
if [ "${WANDB:-0}" = "1" ]; then
    WANDB_FLAG="--wandb --wandb_project ${WANDB_PROJECT}"
fi

# ── COMMON: flags that are identical across ALL 4 phases ─────────────────────
# DO NOT add --mix_vqa, --curriculum, or --focal here — they are phase-specific.
COMMON="
  --model E
  --use_mutan
  --layer_norm
  --dropconnect
  --q_highway
  --char_cnn
  --no_compile
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

if [[ "$RUN_PHASE" == "all" || "$RUN_PHASE" == "1" ]]; then
    echo "============================================================"
    echo "  BUILDING UNIFIED VOCABULARY (VQA-E + VQA v2.0)"
    echo "============================================================"
    python src/scripts/1_build_vocab.py
fi

# =============================================================================
# Phase 1 — Baseline (15 epochs, LR=1e-3, ConvNeXt FROZEN)
#
# Flags:
#   --mix_vqa       ON  → widens vocab by mixing 70% VQA v2.0
#   --curriculum    OFF → blocked by --mix_vqa (train.py L449-450)
#   --focal         ON  → SequenceFocalLoss for rare-token emphasis
#
# Target: new modules (MHCA, MUTAN, decoder) fully converge.
# 15 epochs (up from 10) because with LR warmup=3, effective training
# starts at epoch 4. Early stopping patience=5 will abort if stuck.
# =============================================================================
run_phase1() {
    PHASE1_TOTAL=20
    PHASE1_RESUME_FLAG=""
    PHASE1_WARMUP=3
    PHASE1_EPOCHS=${PHASE1_TOTAL}

    RESUME_P1="checkpoints/model_e_resume.pth"
    if [ -f "${RESUME_P1}" ]; then
        DONE=$(python -c "
import torch, sys
ckpt = torch.load('${RESUME_P1}', map_location='cpu', weights_only=False)
done = ckpt.get('epoch', 0)
remaining = ${PHASE1_TOTAL} - done
if remaining <= 0:
    sys.exit(0)
print(done, remaining)
" 2>/dev/null) || true

        if [ -z "$DONE" ]; then
            echo "  Phase 1 already complete (${PHASE1_TOTAL} epochs done). Skipping."
            return 0
        fi

        DONE_EP=$(echo "$DONE" | awk '{print $1}')
        PHASE1_EPOCHS=$(echo "$DONE" | awk '{print $2}')
        PHASE1_RESUME_FLAG="--resume ${RESUME_P1}"
        PHASE1_WARMUP=0   # warmup already done; cosine scheduler resumes from saved state
        echo "  Auto-detected resume: ${DONE_EP} epochs done → ${PHASE1_EPOCHS} remaining"
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  PHASE 1 — Baseline (${PHASE1_EPOCHS} ep remaining / ${PHASE1_TOTAL} total)"
    echo "  mix_vqa: ON | curriculum: OFF (blocked) | focal: ON"
    echo "  Batch: ${BATCH_SIZE}×${ACCUM_STEPS} = $((BATCH_SIZE * ACCUM_STEPS)) effective"
    echo "════════════════════════════════════════════════════════════"
    # shellcheck disable=SC2086
    python src/train.py ${COMMON} \
        --epochs ${PHASE1_EPOCHS} \
        --lr 1e-3 \
        --warmup_epochs ${PHASE1_WARMUP} \
        --mix_vqa --mix_vqa_fraction 0.7 \
        --focal \
        --early_stopping 5 \
        --phase 1 \
        ${PHASE1_RESUME_FLAG} \
        --wandb_run_name "model_e_phase1_b${BATCH_SIZE}" \
        --wandb_tags "modelE,phase1,frozen-cnn,mix-vqa"
    echo "✓ Phase 1 done → checkpoints/model_e_best.pth"
}

# =============================================================================
# Phase 2 — Fine-tune ConvNeXt (12 epochs, LR=5e-4)
#
# Flags:
#   --mix_vqa       OFF → pure VQA-E only; model focuses on explanation quality
#   --curriculum    ON  → question-type curriculum now active
#   --focal         ON  → still use focal for rare token emphasis
#
# 12 epochs (up from 5): ConvNeXt-Base is deep. At LR×0.1=5e-5 for the CNN,
# gradient updates per epoch are small. More epochs needed for CNN to adapt
# its ImageNet representations toward "Why/How" visual concepts.
#
# If val_loss is decreasing slowly after epoch 6-8, try --cnn_lr_factor 0.2.
# =============================================================================
run_phase2() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  PHASE 2 — CNN Fine-tune (max 20 ep / early_stop=5, lr=5e-4, ConvNeXt unfrozen)"
    echo "  mix_vqa: OFF | curriculum: ON | focal: ON"
    echo "════════════════════════════════════════════════════════════"
    RESUME="checkpoints/model_e_resume.pth"
    if [ ! -f "${RESUME}" ]; then
        echo "ERROR: ${RESUME} not found. Run Phase 1 first."
        exit 1
    fi
    # shellcheck disable=SC2086
    python src/train.py ${COMMON} \
        --epochs 20 \
        --lr 5e-4 \
        --warmup_epochs 0 \
        --curriculum \
        --focal \
        --finetune_cnn \
        --cnn_lr_factor 0.1 \
        --early_stopping 5 \
        --reset_best_val_loss \
        --resume "${RESUME}" \
        --phase 2 \
        --wandb_run_name "model_e_phase2_b${BATCH_SIZE}" \
        --wandb_tags "modelE,phase2,finetune-cnn,curriculum"
    echo "✓ Phase 2 done → checkpoints/model_e_best.pth"
}

# =============================================================================
# Phase 3 — Scheduled Sampling (15 epochs max, LR=2e-4)
#
# Flags:
#   --mix_vqa       OFF → pure VQA-E
#   --curriculum    ON
#   --focal         ON
#
# 15 epochs, patience=7: SS disrupts training for ~5 epochs as the LSTM
# adapts from 100% teacher forcing to seeing its own imperfect output.
# patience=3 is too aggressive — train loss must rise before falling again.
# ε schedule: 0.833 (ep0) → 0.375 (ep14) over 15 relative epochs.
#
# Resume priority: model_e_epoch45.pth (Phase 2 best, val=2.8734) if available,
# otherwise model_e_resume.pth. This handles the case where Phase 3 was run
# with patience=3 and resume.pth was overwritten with degraded Phase 3 state.
# =============================================================================
run_phase3() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  PHASE 3 — Scheduled Sampling (15 ep max, lr=2e-4, ε: 0.833→0.375)"
    echo "  mix_vqa: OFF | curriculum: ON | focal: ON | patience=7"
    echo "════════════════════════════════════════════════════════════"
    # Resume priority: epoch51 (P3 best, Phase 5 PGN start) > epoch45 (P2 best) > resume.pth
    if [ -f "checkpoints/model_e_epoch51.pth" ]; then
        RESUME="checkpoints/model_e_epoch51.pth"
        echo "  Using Phase 3 best checkpoint: model_e_epoch51.pth (P3 best — Phase 5 PGN resume)"
    elif [ -f "checkpoints/model_e_epoch45.pth" ]; then
        RESUME="checkpoints/model_e_epoch45.pth"
        echo "  Using Phase 2 best checkpoint: model_e_epoch45.pth (ep36, val=2.8734)"
    else
        RESUME="checkpoints/model_e_resume.pth"
    fi
    if [ ! -f "${RESUME}" ]; then
        echo "ERROR: ${RESUME} not found. Run Phase 2 first."
        exit 1
    fi
    # shellcheck disable=SC2086
    python src/train.py ${COMMON} \
        --epochs 15 \
        --lr 2e-4 \
        --warmup_epochs 0 \
        --curriculum \
        --focal \
        --finetune_cnn \
        --cnn_lr_factor 0.1 \
        --scheduled_sampling \
        --ss_k 5 \
        --pgn \
        --early_stopping 7 \
        --reset_best_val_loss \
        --resume "${RESUME}" \
        --phase 3 \
        --wandb_run_name "model_e_phase3_pgn_b${BATCH_SIZE}" \
        --wandb_tags "modelE,phase3,scheduled-sampling,curriculum,pgn"
    echo "✓ Phase 3 done → checkpoints/model_e_best.pth"
}

# =============================================================================
# Phase 4 — SCST RL (3 epochs, LR=5e-5)
#
# Flags:
#   --mix_vqa       OFF → pure VQA-E
#   --curriculum    ON  → keep difficulty pacing during RL
#   --focal         OFF → SCST uses plain CE as supervised anchor.
#                         Focal loss re-weighting destabilizes REINFORCE gradients.
#                         (Plain CE at 50% weight keeps generation stable.)
#
# CRITICAL: Do NOT extend beyond 5 epochs.
# RL directly optimizes BLEU score — too many epochs causes "Language Drift":
# the model learns BLEU-gaming patterns (degenerate repetition that scores high
# but reads unnaturally). 3 epochs is the safe zone.
# =============================================================================
run_phase4() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  PHASE 4 — SCST RL (3 ep, lr=5e-5, REINFORCE+CE)"
    echo "  mix_vqa: OFF | curriculum: ON | focal: OFF | length_bonus: 0.03"
    echo "════════════════════════════════════════════════════════════"
    RESUME="checkpoints/model_e_resume.pth"
    if [ ! -f "${RESUME}" ]; then
        echo "ERROR: ${RESUME} not found. Run Phase 3 first."
        exit 1
    fi
    # shellcheck disable=SC2086
    python src/train.py ${COMMON} \
        --epochs 5 \
        --lr 5e-5 \
        --warmup_epochs 0 \
        --curriculum \
        --finetune_cnn \
        --cnn_lr_factor 0.1 \
        --scst \
        --scst_lambda 0.5 \
        --scst_bleu_weight 0.5 \
        --scst_meteor_weight 0.5 \
        --scst_length_bonus 0.03 \
        --pgn \
        --reset_best_val_loss \
        --resume "${RESUME}" \
        --phase 4 \
        --wandb_run_name "model_e_phase4_pgn_b${BATCH_SIZE}" \
        --wandb_tags "modelE,phase4,scst-rl,curriculum,pgn"
    echo "✓ Phase 4 done → checkpoints/model_e_best.pth"
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
echo ""
echo "VQA-E Model E — RTX 5070 Ti (Blackwell, BF16)"
echo "Batch: ${BATCH_SIZE} | Workers: ${NUM_WORKERS} | Dropout: ${DROPOUT}"
echo "W&B: ${WANDB:-0}"
echo ""
echo "Phase schedule: 20 + 20 + 7 + 3 (max) — early_stopping terminates each phase early"

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
        echo "  All 4 phases complete (37 epochs total)"
        echo "  Best checkpoint: checkpoints/model_e_best.pth"
        echo "══════════════════════════════════════════════"
        ;;
    *)
        echo "Usage: bash train_model_e.sh [1|2|3|4|all]"
        exit 1
        ;;
esac
