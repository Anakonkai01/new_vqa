#!/bin/bash
# ==============================================================================
# Model H — full curriculum (Phase 1→4 + eval). Chạy từ thư mục gốc repo:
#   cd /path/to/new_vqa && bash train_model_h.sh
# ==============================================================================

set -e

cd "$(dirname "$0")"

export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# torch.compile + long backward kernels can trigger cudaErrorLaunchTimeout on some setups (esp. GPU shared with display).
# --no_compile below avoids that; re-enable compile only if you train headless and never see timeouts.

# --- CẤU HÌNH ĐƯỜNG DẪN ---
VG_FEAT_DIR="data/vg_features/"
MERGED_JSON="data/processed/merged_train_filtered.json"
VOCAB_Q="data/processed/vocab_questions.json"
VOCAB_A="data/processed/vocab_answers.json"
VQA2_Q="data/annotations/vqa_v2/v2_OpenEnded_mscoco_train2014_questions.json"
VQA2_A="data/annotations/vqa_v2/v2_mscoco_train2014_annotations.json"

# --- CẤU HÌNH PHẦN CỨNG (GIÁO SƯ ĐÃ TỐI ƯU) ---
WORKERS=8
BASE_BATCH=96        # 15GB VRAM-safe baseline for CE phases
RL_BATCH=48          # Safer SCST batch on 15GB VRAM
EVAL_BATCH=64        # Official-val / eval selector batch to avoid VRAM spikes
DROPOUT=0.5
PATIENCE=15


echo "======================================================================"
echo "🚀 KHỞI ĐỘNG MASTER PIPELINE: GENERATIVE VQA MODEL H"
echo "======================================================================"

# ==============================================================================
# PHASE 1: ALIGNMENT WARM-UP
# ==============================================================================
echo -e "\n\n>>> [PHASE 1] BẮT ĐẦU HUẤN LUYỆN ALIGNMENT..."
python src/train_h.py \
    --phase 1 \
    --epochs 50 \
    --patience ${PATIENCE} \
    --lr 5e-4 \
    --warmup_epochs 4 \
    --batch_size ${BASE_BATCH} \
    --dropout ${DROPOUT} \
    --num_workers ${WORKERS} \
    --vg_feat_dir ${VG_FEAT_DIR} \
    --merged_json ${MERGED_JSON} \
    --vocab_q_path ${VOCAB_Q} \
    --vocab_a_path ${VOCAB_A} \
    --vqa_v2_questions ${VQA2_Q} \
    --vqa_v2_annotations ${VQA2_A} \
    --infonce \
    --use_fasttext \
    --select_on_official_val \
    --official_val_max_samples 2048 \
    --official_val_batch_size ${EVAL_BATCH} \
    --official_val_num_workers ${WORKERS} \
    --wandb \
    --wandb_project "vqa-model-h" \
    --wandb_run_name "model_h_phase1_auto" \
    --save_legacy_alias \
    --no_compile

echo ">>> [EVAL 1] ĐÁNH GIÁ PHASE 1..."
python src/evaluate_h.py \
    --checkpoint checkpoints/h/model_h_phase1_best.pth \
    --vg_feat_dir ${VG_FEAT_DIR} \
    --vocab_q_path ${VOCAB_Q} \
    --vocab_a_path ${VOCAB_A} \
    --datasets vqa_e vqa_x aokvqa \
    --beam_width 5 \
    --batch_size ${EVAL_BATCH} \
    --num_workers ${WORKERS} \
    --use_fasttext \

# ==============================================================================
# PHASE 2: MASTERY & FASTTEXT INTEGRATION
# ==============================================================================
echo -e "\n\n>>> [PHASE 2] BẮT ĐẦU HUẤN LUYỆN MASTERY (FASTTEXT)..."
python src/train_h.py \
    --phase 2 \
    --epochs 50 \
    --patience ${PATIENCE} \
    --lr 1e-4 \
    --warmup_epochs 2 \
    --batch_size ${BASE_BATCH} \
    --dropout ${DROPOUT} \
    --num_workers ${WORKERS} \
    --vg_feat_dir ${VG_FEAT_DIR} \
    --merged_json ${MERGED_JSON} \
    --vocab_q_path ${VOCAB_Q} \
    --vocab_a_path ${VOCAB_A} \
    --vqa_v2_questions ${VQA2_Q} \
    --vqa_v2_annotations ${VQA2_A} \
    --infonce \
    --use_fasttext \
    --select_on_official_val \
    --official_val_max_samples 2048 \
    --official_val_batch_size ${EVAL_BATCH} \
    --official_val_num_workers ${WORKERS} \
    --resume checkpoints/h/model_h_phase1_best.pth \
    --wandb \
    --wandb_project "vqa-model-h" \
    --wandb_run_name "model_h_phase2_auto" \
    --save_legacy_alias \
    --no_compile

echo ">>> [EVAL 2] ĐÁNH GIÁ PHASE 2..."
python src/evaluate_h.py \
    --checkpoint checkpoints/h/model_h_phase2_best.pth \
    --vg_feat_dir ${VG_FEAT_DIR} \
    --vocab_q_path ${VOCAB_Q} \
    --vocab_a_path ${VOCAB_A} \
    --datasets vqa_e vqa_x aokvqa \
    --use_fasttext \
    --beam_width 5 \
    --batch_size ${EVAL_BATCH} \
    --num_workers ${WORKERS} \

# ==============================================================================
# PHASE 3: SCHEDULED SAMPLING (BRIDGE TO RL)
# ==============================================================================
echo -e "\n\n>>> [PHASE 3] BẮT ĐẦU SCHEDULED SAMPLING..."
python src/train_h.py \
    --phase 3 \
    --epochs 50 \
    --patience ${PATIENCE} \
    --lr 1.5e-4 \
    --warmup_epochs 0 \
    --batch_size ${BASE_BATCH} \
    --dropout ${DROPOUT} \
    --num_workers ${WORKERS} \
    --vg_feat_dir ${VG_FEAT_DIR} \
    --merged_json ${MERGED_JSON} \
    --vocab_q_path ${VOCAB_Q} \
    --vocab_a_path ${VOCAB_A} \
    --vqa_v2_questions ${VQA2_Q} \
    --vqa_v2_annotations ${VQA2_A} \
    --infonce \
    --use_fasttext \
    --scheduled_sampling \
    --ss_k 5 \
    --min_decode_len 8 \
    --select_on_official_val \
    --official_val_max_samples 2048 \
    --official_val_batch_size ${EVAL_BATCH} \
    --official_val_num_workers ${WORKERS} \
    --resume checkpoints/h/model_h_phase2_best.pth \
    --wandb \
    --wandb_project "vqa-model-h" \
    --wandb_run_name "model_h_phase3_ss_auto" \
    --save_legacy_alias \
    --no_compile

echo ">>> [EVAL 3] ĐÁNH GIÁ PHASE 3..."
python src/evaluate_h.py \
    --checkpoint checkpoints/h/model_h_phase3_best.pth \
    --vg_feat_dir ${VG_FEAT_DIR} \
    --vocab_q_path ${VOCAB_Q} \
    --vocab_a_path ${VOCAB_A} \
    --datasets vqa_e vqa_x aokvqa \
    --use_fasttext \
    --beam_width 5 \
    --batch_size ${EVAL_BATCH} \
    --num_workers ${WORKERS} \

# ==============================================================================
# PHASE 4: SELF-CRITICAL SEQUENCE TRAINING (RL)
# ==============================================================================
echo -e "\n\n>>> [PHASE 4] BẮT ĐẦU HUẤN LUYỆN SCST (CIDER REWARD)..."
python src/train_h.py \
    --phase 4 \
    --epochs 50 \
    --patience 20 \
    --lr 5e-5 \
    --warmup_epochs 0 \
    --batch_size ${RL_BATCH} \
    --dropout ${DROPOUT} \
    --num_workers ${WORKERS} \
    --vg_feat_dir ${VG_FEAT_DIR} \
    --merged_json ${MERGED_JSON} \
    --vocab_q_path ${VOCAB_Q} \
    --vocab_a_path ${VOCAB_A} \
    --vqa_v2_questions ${VQA2_Q} \
    --vqa_v2_annotations ${VQA2_A} \
    --use_fasttext \
    --infonce \
    --scst \
    --scst_bleu 0.3 \
    --scst_meteor 0.3 \
    --ohp_lambda 0.1 \
    --entropy_mac_coef 0.05 \
    --select_on_official_val \
    --official_val_max_samples 2048 \
    --official_val_batch_size ${EVAL_BATCH} \
    --official_val_num_workers ${WORKERS} \
    --resume checkpoints/h/model_h_phase3_best.pth \
    --wandb \
    --wandb_project "vqa-model-h" \
    --wandb_run_name "model_h_phase4_scst_auto" \
    --save_legacy_alias \
    --no_compile

echo ">>> [EVAL 4] ĐÁNH GIÁ PHASE 4 (FINAL)..."
python src/evaluate_h.py \
    --checkpoint checkpoints/h/model_h_phase4_best.pth \
    --vg_feat_dir ${VG_FEAT_DIR} \
    --vocab_q_path ${VOCAB_Q} \
    --vocab_a_path ${VOCAB_A} \
    --datasets vqa_e vqa_x aokvqa \
    --use_fasttext \
    --beam_width 5 \
    --batch_size ${EVAL_BATCH} \
    --num_workers ${WORKERS}

echo -e "\n======================================================================"
echo "🎉 MASTER PIPELINE HOÀN TẤT. VUI LÒNG KIỂM TRA BÁO CÁO TRÊN WANDB."
echo "======================================================================"
