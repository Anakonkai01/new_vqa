#!/bin/bash
# Phase 3 — scheduled sampling. Cần checkpoints/h/model_h_phase2_best.pth.
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

VG="data/vg_features/"
MERGED="data/processed/merged_train_filtered.json"
VQ="data/processed/vocab_questions.json"
VA="data/processed/vocab_answers.json"

python src/train_h.py \
    --phase 3 \
    --epochs 30 \
    --patience 15 \
    --lr 2e-4 \
    --warmup_epochs 0 \
    --batch_size 128 \
    --dropout 0.5 \
    --num_workers 8 \
    --vg_feat_dir "${VG}" \
    --merged_json "${MERGED}" \
    --vocab_q_path "${VQ}" \
    --vocab_a_path "${VA}" \
    --infonce \
    --use_fasttext \
    --scheduled_sampling \
    --ss_k 5 \
    --resume checkpoints/h/model_h_phase2_best.pth \
    --wandb \
    --wandb_project "vqa-model-h" \
    --wandb_run_name "model_h_phase3" \
    --save_legacy_alias \
    --no_compile
