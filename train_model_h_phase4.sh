#!/bin/bash
# Phase 4 — SCST. Cần checkpoints/h/model_h_phase3_best.pth. Batch nhỏ hơn để tránh OOM.
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VG="data/vg_features/"
MERGED="data/processed/merged_train_filtered.json"
VQ="data/processed/vocab_questions.json"
VA="data/processed/vocab_answers.json"

python src/train_h.py \
    --phase 4 \
    --epochs 50 \
    --patience 15 \
    --lr 1e-5 \
    --warmup_epochs 0 \
    --batch_size 48 \
    --dropout 0.5 \
    --num_workers 8 \
    --vg_feat_dir "${VG}" \
    --merged_json "${MERGED}" \
    --vocab_q_path "${VQ}" \
    --vocab_a_path "${VA}" \
    --use_fasttext \
    --infonce \
    --scst \
    --ohp_lambda 0.1 \
    --select_on_official_val \
    --official_val_max_samples 2048 \
    --official_val_batch_size 64 \
    --official_val_num_workers 4 \
    --resume checkpoints/h/model_h_phase3_best.pth \
    --wandb \
    --wandb_project "vqa-model-h" \
    --wandb_run_name "model_h_phase4" \
    --save_legacy_alias \
    --no_compile
