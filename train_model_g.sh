#!/usr/bin/env bash
# =============================================================================
# Model G — 4-Phase Curriculum Training Script
# RTX 5070 Ti (16 GB VRAM, Blackwell SM 12.0, BF16 native)
#
# Architecture: BUTD encoder (G1 geo7) + DCAN + GatedFusion +
#               LSTMDecoderG (G2 three-way PGN + G5 length conditioning) +
#               InfoNCE alignment heads (G3) + OHP reward (G4)
#
# Curriculum (per Architecture Specification v2, Section 6):
#   Phase 1 — Alignment   (15 ep): 40% VQA v2.0 + 30% VQA-E + 30% A-OKVQA
#             L = FocalLoss + 0.5*Lcov + 0.1*LInfoNCE
#             LR = 1e-3, warmup 2 ep, cosine decay
#
#   Phase 2 — Mastery     (10 ep): 100% expl data + 20% VQA v2.0 replay
#             L = same as Phase 1 (no SS)
#             LR = 5e-4
#
#   Phase 3 — Correction  ( 7 ep): same as Phase 2 + scheduled sampling
#             L = same  (SS epsilon: k=5 decay)
#             LR = 2e-4
#
#   Phase 4 — Optimization ( 3 ep): VQA-E + VQA-X only + SCST RL + OHP
#             L = 0.5*CE + 0.5*SCST + 0.1*LInfoNCE
#             LR = 5e-5, batch_size=64
#
# Flag logic per phase (do NOT move phase-specific flags to COMMON):
#   --focal         PHASES 1-3. FocalSequenceLoss with per-sample T-norm.
#                   NOT in Phase 4: SCST uses plain CE as supervised anchor.
#   --scst          PHASE 4 only. REINFORCE with greedy baseline.
#   --ohp           PHASE 4 only. G4 Object Hallucination Penalty in SCST reward.
#   --scheduled_sampling  PHASE 3 only.
#
# Data mixing is handled internally by train_g.py (VQAGenerativeDataset +
# build_mixed_sampler / build_replay_sampler).  The shell script only needs
# to pass --phase to tell train_g.py which mixing strategy to apply.
#
# Prerequisites:
#   1. Pre-extract BUTD features with 7-dim spatial geometry (G1):
#      python src/scripts/extract_features_model_f.py --geo7 \
#          --output_dir data/features/butd_g1
#   2. Build merged dataset JSON:
#      python src/scripts/merge_sources.py \
#          --output data/processed/merged_train_filtered.json
#   3. Build vocabulary:
#      python src/scripts/1_build_vocab_v2.py
#
# Usage:
#   bash train_model_g.sh               # all 4 phases
#   bash train_model_g.sh 1             # Phase 1 only
#   bash train_model_g.sh 2             # Phase 2 (requires Phase 1 checkpoint)
#   bash train_model_g.sh 3             # Phase 3
#   bash train_model_g.sh 4             # Phase 4 SCST + OHP
#
# Overrides:
#   BATCH_SIZE=96 bash train_model_g.sh
#   WANDB=1 bash train_model_g.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ───────────────────────────────────────────────────────────────────
BATCH_SIZE="${BATCH_SIZE:-128}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-16}"
DROPOUT="${DROPOUT:-0.3}"
WANDB_PROJECT="${WANDB_PROJECT:-vqa-g}"

BUTD_FEAT_DIR="${BUTD_FEAT_DIR:-data/features/butd_g1}"
MERGED_JSON="${MERGED_JSON:-data/processed/merged_train_filtered.json}"
VOCAB_Q_PATH="${VOCAB_Q_PATH:-data/processed/vocab_questions.json}"
VOCAB_A_PATH="${VOCAB_A_PATH:-data/processed/vocab_answers.json}"

WANDB_FLAG=""
if [ "${WANDB:-0}" = "1" ]; then
    WANDB_FLAG="--wandb --wandb_project ${WANDB_PROJECT}"
fi

# ── COMMON: flags identical across ALL phases ─────────────────────────────────
# Phase-specific flags (--focal, --scst, --ohp, --scheduled_sampling) are NOT here.
COMMON="
  --model G
  --no_compile
  --geo7
  --pgn3
  --infonce
  --infonce_beta 0.1
  --infonce_tau 0.07
  --len_cond
  --layer_norm
  --dropconnect
  --glove --glove_dim 300
  --coverage --coverage_lambda 0.5
  --batch_size ${BATCH_SIZE}
  --accum_steps ${ACCUM_STEPS}
  --num_workers ${NUM_WORKERS}
  --dropout ${DROPOUT}
  --weight_decay 1e-4
  --grad_clip 2.0
  --label_smoothing 0.1
  --butd_feat_dir ${BUTD_FEAT_DIR}
  --merged_json ${MERGED_JSON}
  --vocab_q_path ${VOCAB_Q_PATH}
  --vocab_a_path ${VOCAB_A_PATH}
  ${WANDB_FLAG}"

RUN_PHASE="${1:-all}"
shift || true
EXTRA_ARGS="$*"

# =============================================================================
# Phase 1 — Alignment (15 epochs, LR=1e-3)
#
# Data:  40% VQA v2.0 + 30% VQA-E + 30% A-OKVQA  (mixed by train_g.py)
# Loss:  FocalSequenceLoss + 0.5*Lcov + 0.1*LInfoNCE
# LR:    1e-3 with warmup=2 and cosine decay
# Goal:  G1-G5 modules align visual and language representations;
#        InfoNCE contrastive alignment bootstraps the projection heads.
#
# 15 epochs: warmup takes 2 ep, 3-source mixture needs extra epochs to
# converge.  Early stopping patience=5 aborts if stuck.
# Milestone checkpoint saved at epoch 15.
# =============================================================================
run_phase1() {
    PHASE1_TOTAL=15
    PHASE1_RESUME_FLAG=""
    PHASE1_WARMUP=2
    PHASE1_EPOCHS=${PHASE1_TOTAL}

    RESUME_P1="checkpoints/model_g_resume.pth"
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
        PHASE1_WARMUP=0
        echo "  Auto-detected resume: ${DONE_EP} epochs done → ${PHASE1_EPOCHS} remaining"
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  PHASE 1 — Alignment (${PHASE1_EPOCHS} ep remaining / ${PHASE1_TOTAL} total)"
    echo "  Data: 40% VQAv2 + 30% VQA-E + 30% A-OKVQA | focal: ON | InfoNCE: ON"
    echo "  Batch: ${BATCH_SIZE}×${ACCUM_STEPS} = $((BATCH_SIZE * ACCUM_STEPS)) effective"
    echo "════════════════════════════════════════════════════════════"
    # shellcheck disable=SC2086
    python src/train.py ${COMMON} \
        --epochs ${PHASE1_EPOCHS} \
        --lr 1e-3 \
        --warmup_epochs ${PHASE1_WARMUP} \
        --focal \
        --early_stopping 5 \
        --phase 1 \
        ${PHASE1_RESUME_FLAG} \
        --wandb_run_name "model_g_phase1_b${BATCH_SIZE}" \
        --wandb_tags "modelG,phase1,alignment,infonce,geo7,pgn3" \
        ${EXTRA_ARGS}
    echo "✓ Phase 1 done → checkpoints/model_g_best.pth"
}

# =============================================================================
# Phase 2 — Mastery (10 epochs, LR=5e-4)
#
# Data:  100% explanation data + 20% VQA v2.0 replay  (train_g.py)
# Loss:  FocalSequenceLoss + 0.5*Lcov + 0.1*LInfoNCE  (no SS yet)
# LR:    5e-4, no warmup (cosine resumes from Phase 1 state)
# Goal:  Model masters explanation-format sequences with full G2/G5 features.
#        Replay fraction (20%) prevents catastrophic forgetting of short answers.
# =============================================================================
run_phase2() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  PHASE 2 — Mastery (max 10 ep, lr=5e-4, expl + 20% replay)"
    echo "  focal: ON | InfoNCE: ON | no scheduled sampling"
    echo "════════════════════════════════════════════════════════════"
    RESUME="checkpoints/model_g_resume.pth"
    if [ ! -f "${RESUME}" ]; then
        echo "ERROR: ${RESUME} not found. Run Phase 1 first."
        exit 1
    fi
    # shellcheck disable=SC2086
    python src/train.py ${COMMON} \
        --epochs 10 \
        --lr 5e-4 \
        --warmup_epochs 0 \
        --focal \
        --early_stopping 4 \
        --reset_best_val_loss \
        --resume "${RESUME}" \
        --phase 2 \
        --wandb_run_name "model_g_phase2_b${BATCH_SIZE}" \
        --wandb_tags "modelG,phase2,mastery,replay" \
        ${EXTRA_ARGS}
    echo "✓ Phase 2 done → checkpoints/model_g_best.pth"
}

# =============================================================================
# Phase 3 — Correction (7 epochs max, LR=2e-4)
#
# Data:  same as Phase 2 (expl + 20% replay)
# Loss:  same + Scheduled Sampling (epsilon decays with k=5)
# LR:    2e-4
# Goal:  Reduce exposure bias from teacher forcing.
#        SS epsilon: 0.913 (ep0) → 0.700 (ep6) — gentle decay over 7 epochs.
#
# patience=5: SS disrupts training for ~3 ep as LSTM adapts.
# patience=3 is too aggressive for explanation-length sequences.
# =============================================================================
run_phase3() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  PHASE 3 — Correction (7 ep max, lr=2e-4, SS ε decay k=5)"
    echo "  focal: ON | scheduled_sampling: ON | patience=5"
    echo "════════════════════════════════════════════════════════════"
    RESUME="checkpoints/model_g_resume.pth"
    if [ ! -f "${RESUME}" ]; then
        echo "ERROR: ${RESUME} not found. Run Phase 2 first."
        exit 1
    fi
    # shellcheck disable=SC2086
    python src/train.py ${COMMON} \
        --epochs 7 \
        --lr 2e-4 \
        --warmup_epochs 0 \
        --focal \
        --scheduled_sampling \
        --ss_k 5 \
        --early_stopping 5 \
        --reset_best_val_loss \
        --resume "${RESUME}" \
        --phase 3 \
        --wandb_run_name "model_g_phase3_b${BATCH_SIZE}" \
        --wandb_tags "modelG,phase3,correction,scheduled-sampling" \
        ${EXTRA_ARGS}
    echo "✓ Phase 3 done → checkpoints/model_g_best.pth"
}

# =============================================================================
# Phase 4 — Optimization (3 epochs, LR=5e-5, batch=64)
#
# Data:  VQA-E + VQA-X only  (train_g.py Phase 4 dataset selection)
# Loss:  0.5*CE + 0.5*SCST + 0.1*LInfoNCE
# SCST:  REINFORCE with greedy baseline; reward = 0.5*BLEU4 + 0.5*METEOR
# G4:    OHP penalty in SCST sample reward (delta=0.5, weight=0.3)
# LR:    5e-5, batch_size=64 (reduced for SCST memory)
# focal: OFF — plain CE as supervised anchor (focal destabilizes REINFORCE)
#
# CRITICAL: Do NOT extend beyond 5 epochs.
# RL directly optimizes BLEU/METEOR — more epochs causes language drift
# (degenerate repetition that scores high but is unnatural).
# =============================================================================
run_phase4() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  PHASE 4 — Optimization (3 ep, lr=5e-5, SCST+OHP, bs=64)"
    echo "  focal: OFF | scst: ON | ohp: ON | data: VQA-E + VQA-X only"
    echo "════════════════════════════════════════════════════════════"
    RESUME="checkpoints/model_g_resume.pth"
    if [ ! -f "${RESUME}" ]; then
        echo "ERROR: ${RESUME} not found. Run Phase 3 first."
        exit 1
    fi
    # shellcheck disable=SC2086
    python src/train.py ${COMMON} \
        --epochs 3 \
        --lr 5e-5 \
        --batch_size 64 \
        --warmup_epochs 0 \
        --scst \
        --scst_lambda 0.5 \
        --scst_bleu_weight 0.5 \
        --scst_meteor_weight 0.5 \
        --ohp \
        --ohp_weight 0.3 \
        --ohp_threshold 0.5 \
        --reset_best_val_loss \
        --resume "${RESUME}" \
        --phase 4 \
        --wandb_run_name "model_g_phase4_scst_ohp_b64" \
        --wandb_tags "modelG,phase4,scst-rl,ohp,g4" \
        ${EXTRA_ARGS}
    echo "✓ Phase 4 done → checkpoints/model_g_best.pth"
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
echo ""
echo "VQA Model G — RTX 5070 Ti (Blackwell, BF16)"
echo "Batch: ${BATCH_SIZE} | Workers: ${NUM_WORKERS} | Dropout: ${DROPOUT}"
echo "BUTD feats: ${BUTD_FEAT_DIR}"
echo "Merged JSON: ${MERGED_JSON}"
echo "W&B: ${WANDB:-0}"
echo ""
echo "Phase schedule: 15 + 10 + 7 + 3 (max) — early_stopping terminates each phase"

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
        echo "══════════════════════════════════════════════════════"
        echo "  All 4 phases complete (35 epochs max)"
        echo "  Best checkpoint: checkpoints/model_g_best.pth"
        echo "══════════════════════════════════════════════════════"
        ;;
    *)
        echo "Usage: bash train_model_g.sh [1|2|3|4|all]"
        exit 1
        ;;
esac
