#!/bin/bash
# Phase 1 only — đồng bộ với train_model_h.sh (train từ đầu, không --resume).
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VG="data/vg_features/"
MERGED="data/processed/merged_train_filtered.json"
VQ="data/processed/vocab_questions.json"
VA="data/processed/vocab_answers.json"
VQA2Q="data/annotations/vqa_v2/v2_OpenEnded_mscoco_train2014_questions.json"
VQA2A="data/annotations/vqa_v2/v2_mscoco_train2014_annotations.json"

python src/train_h.py \
    --phase 1 \
    --epochs 50 \
    --patience 15 \
    --lr 5e-4 \
    --warmup_epochs 4 \
    --batch_size 96 \
    --dropout 0.5 \
    --num_workers 8 \
    --vg_feat_dir "${VG}" \
    --merged_json "${MERGED}" \
    --vocab_q_path "${VQ}" \
    --vocab_a_path "${VA}" \
    --vqa_v2_questions "${VQA2Q}" \
    --vqa_v2_annotations "${VQA2A}" \
    --infonce \
    --use_fasttext \
    --select_on_official_val \
    --official_val_max_samples 2048 \
    --official_val_batch_size 64 \
    --official_val_num_workers 4 \
    --wandb \
    --wandb_project "vqa-model-h" \
    --wandb_run_name "model_h_phase1" \
    --save_legacy_alias \
    --no_compile
