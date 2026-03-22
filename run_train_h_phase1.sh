#!/bin/bash
# Phase 1 only — đồng bộ với train_model_h.sh (train từ đầu, không --resume).
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

VG="data/vg_features/"
MERGED="data/processed/merged_train_filtered.json"
VQ="data/processed/vocab_questions.json"
VA="data/processed/vocab_answers.json"

python src/train_h.py \
    --phase 1 \
    --epochs 50 \
    --patience 15 \
    --lr 1e-3 \
    --warmup_epochs 2 \
    --batch_size 128 \
    --dropout 0.5 \
    --num_workers 8 \
    --vg_feat_dir "${VG}" \
    --merged_json "${MERGED}" \
    --vocab_q_path "${VQ}" \
    --vocab_a_path "${VA}" \
    --infonce \
    --use_fasttext \
    --wandb \
    --wandb_project "vqa-model-h" \
    --wandb_run_name "model_h_phase1" \
    --save_legacy_alias \
    --no_compile
