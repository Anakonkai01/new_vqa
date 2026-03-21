#!/bin/bash
# train_model_h.sh - Advanced CLI Launcher for VQA Model H

# Default configurations
PHASE="all"
RESUME_CKPT=""
WANDB_FLAG=""
FEAT_DIR="data/features/vg_h1"
MERGED_JSON="data/processed/merged_train_filtered.json"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --phase) PHASE="$2"; shift ;;
        --resume) RESUME_CKPT="--resume $2"; shift ;;
        --wandb) WANDB_FLAG="--wandb" ;;
        --feat_dir) FEAT_DIR="$2"; shift ;;
        --json) MERGED_JSON="$2"; shift ;;
        -h|--help)
            echo "Usage: ./train_model_h.sh [options]"
            echo "Options:"
            echo "  --phase <extract|1|2|3|4|all> Select phase to run (default: all)"
            echo "  --resume <path>               Path to checkpoint (e.g., checkpoints/h/...pth)"
            echo "  --wandb                  Enable W&B logging"
            echo "  --feat_dir <path>        Custom VG features directory (default: data/features/vg_h1)"
            echo "  --json <path>            Merged dataset JSON (default: data/processed/merged_train_filtered.json)"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "========================================================"
echo "🚀 VQA Model H - Ultimate LSTM Architecture Pipeline 🚀"
echo "========================================================"
echo "[Config] Phase: $PHASE | W&B: ${WANDB_FLAG:-OFF} | Resume: ${RESUME_CKPT:-None}"

# Define execution blocks
run_extract() {
    echo "=== Phase 0: Feature Extraction (ResNeXt-101-32x8d + Grid) ==="
    echo "This step extracts Visual Genome features natively via Detectron2."
    
    # Check for train2014
    if [ -d "data/images/train2014" ]; then
        echo "Extracting train2014 images..."
        python src/scripts/extract_vg_features.py --image_dir data/images/train2014 --output_dir "$FEAT_DIR" --batch_size 10
    else
        echo "[WARN] Directory data/images/train2014 not found. Skipping."
    fi

    # Check for val2014
    if [ -d "data/images/val2014" ]; then
        echo "Extracting val2014 images..."
        python src/scripts/extract_vg_features.py --image_dir data/images/val2014 --output_dir "$FEAT_DIR" --batch_size 10
    else
        echo "[WARN] Directory data/images/val2014 not found. Skipping."
    fi

    # Check for test2015
    if [ -d "data/images/test2015" ]; then
        echo "Extracting test2015 images..."
        python src/scripts/extract_vg_features.py --image_dir data/images/test2015 --output_dir "$FEAT_DIR" --batch_size 10
    else
        echo "[WARN] Directory data/images/test2015 not found. Skipping."
    fi
    
    echo "--------------------------------------------------------"
    echo "Feature Extraction Hoàn tất. Sẵn sàng cho Phase 1."
}

run_phase_1() {
    echo "=== Phase 1: Alignment (MAC Network) (Max 50 Epochs - Early Stop) ==="
    python src/train_h.py --phase 1 --epochs 50 --patience 3 \
        --lr 0.001 --warmup_epochs 2 --batch_size 256 \
        --vg_feat_dir "$FEAT_DIR" \
        --merged_json "$MERGED_JSON" \
        --use_fasttext --infonce --save_legacy_alias $WANDB_FLAG $RESUME_CKPT
    
    echo "--------------------------------------------------------"
    echo "Phase 1 Hoàn tất. KHUYẾN NGHỊ: Mở wandb hoặc chạy src/evaluate.py để kiểm tra trước."
}

run_phase_2() {
    echo "=== Phase 2: Mastery (Max 50 Epochs - Early Stop) ==="
    # If resuming a specific phase from CLI, use the user's resume flag. 
    # Otherwise (in 'all' mode), automatically resume from Phase 1's output.
    local LOCAL_RESUME=$RESUME_CKPT
    if [ -z "$LOCAL_RESUME" ] && [ "$PHASE" == "all" ]; then
        LOCAL_RESUME="--resume checkpoints/h/model_h_phase1_resume.pth"
    fi

    python src/train_h.py --phase 2 --epochs 50 --patience 3 \
        --lr 0.0005 --batch_size 256 \
        --vg_feat_dir "$FEAT_DIR" \
        --merged_json "$MERGED_JSON" \
        --use_fasttext --infonce --save_legacy_alias $WANDB_FLAG $LOCAL_RESUME
        
    echo "--------------------------------------------------------"
    echo "Phase 2 Hoàn tất. KHUYẾN NGHỊ: Đảm bảo model đã bắt đầu sinh ra câu giải thích dài."
}

run_phase_3() {
    echo "=== Phase 3: Correction (Scheduled Sampling) (Max 30 Epochs - Early Stop) ==="
    local LOCAL_RESUME=$RESUME_CKPT
    if [ -z "$LOCAL_RESUME" ] && [ "$PHASE" == "all" ]; then
        LOCAL_RESUME="--resume checkpoints/h/model_h_phase2_resume.pth"
    fi

    python src/train_h.py --phase 3 --epochs 30 --patience 3 \
        --lr 0.0002 --batch_size 256 \
        --vg_feat_dir "$FEAT_DIR" \
        --merged_json "$MERGED_JSON" \
        --use_fasttext --infonce --scheduled_sampling --save_legacy_alias $WANDB_FLAG $LOCAL_RESUME
        
    echo "--------------------------------------------------------"
    echo "Phase 3 Hoàn tất. Sẵn sàng cho quá trình khó nhất: Đào tạo tăng cường (RL)."
}

run_phase_4() {
    echo "=== Phase 4: Optimization (SCST Exact Match) (Max 15 Epochs - Early Stop) ==="
    local LOCAL_RESUME=$RESUME_CKPT
    if [ -z "$LOCAL_RESUME" ] && [ "$PHASE" == "all" ]; then
        LOCAL_RESUME="--resume checkpoints/h/model_h_phase3_resume.pth"
    fi

    python src/train_h.py --phase 4 --epochs 15 --patience 2 \
        --lr 0.00005 --batch_size 64 \
        --vg_feat_dir "$FEAT_DIR" \
        --merged_json "$MERGED_JSON" \
        --use_fasttext --infonce --scst --save_legacy_alias $WANDB_FLAG $LOCAL_RESUME
        
    echo "--------------------------------------------------------"
    echo "Phase 4 Hoàn tất. Model H đạt SOTA."
}

# Execution Logic
case $PHASE in
    extract) run_extract ;;
    1) run_phase_1 ;;
    2) run_phase_2 ;;
    3) run_phase_3 ;;
    4) run_phase_4 ;;
    all)
        run_extract
        read -p "Nhấn [Enter] để tự tin đi tiếp sang Phase 1..."
        run_phase_1
        read -p "Nhấn [Enter] để tự tin đi tiếp sang Phase 2..."
        run_phase_2
        read -p "Nhấn [Enter] để đi tiếp sang Phase 3 (Scheduled Sampling)..."
        run_phase_3
        read -p "Nhấn [Enter] để khởi động Phase 4..."
        run_phase_4
        ;;
    *) echo "Invalid Phase: $PHASE. Must be 1, 2, 3, 4, or all."; exit 1 ;;
esac
