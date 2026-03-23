#!/bin/bash
# Phase 4 — SCST. Cần checkpoints/h/model_h_phase3_best.pth.
# Batch nhỏ hơn Phase 3 (48 vs 96) vì SCST chạy 2 decode pass (greedy + sample).
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VG="data/vg_features/"
MERGED="data/processed/merged_train_filtered.json"
VQ="data/processed/vocab_questions.json"
VA="data/processed/vocab_answers.json"
VQA2Q="data/annotations/vqa_v2/v2_OpenEnded_mscoco_train2014_questions.json"
VQA2A="data/annotations/vqa_v2/v2_mscoco_train2014_annotations.json"

python src/train_h.py \
    --phase 4 \
    --epochs 50 \
    --patience 20 \
    --lr 5e-5 \
    --warmup_epochs 0 \
    --batch_size 80 \
    --dropout 0.5 \
    --num_workers 8 \
    --vg_feat_dir "${VG}" \
    --merged_json "${MERGED}" \
    --vocab_q_path "${VQ}" \
    --vocab_a_path "${VA}" \
    --vqa_v2_questions "${VQA2Q}" \
    --vqa_v2_annotations "${VQA2A}" \
    --use_fasttext \
    --infonce \
    --scst \
    --scst_bleu 0.3 \
    --scst_meteor 0.3 \
    --ohp_lambda 0.1 \
    --entropy_mac_coef 0.05 \
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
