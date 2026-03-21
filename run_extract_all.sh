#!/bin/bash
# ====================================================================
# Model H Feature Extraction Automation
# ====================================================================

# Automatically activate the conda environment
eval "$(conda shell.bash hook)"
conda activate d2l

# Safety check for environment
if ! python -c "import detectron2" &> /dev/null; then
    echo "ERROR: The 'd2l' environment is missing Detectron2."
    echo "Please install it: pip install 'git+https://github.com/facebookresearch/detectron2.git'"
    exit 1
fi

echo "=========================================================="
echo "    MODEl H: RESNEXT-101-32x8d FPN FEATURE EXTRACTION     "
echo "=========================================================="
echo "This script will extract BUTD + Grid features + Label Strings"
echo "Total expected size: ~18.6 GB for 123,287 images."
echo "Target Disk Free Space: 63 GB (Safe to proceed!)"
echo "=========================================================="
echo ""

# Prevent PyTorch from heavily fragmenting VRAM during varying image aspect ratio passes
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Extract train2014 (82,783 images)
echo ">>> [1/2] Processing train2014 split..."
python src/scripts/extract_vg_features.py \
    --image_dir data/images/train2014 \
    --output_dir data/vg_features \
    --batch_size 8 \
    --top_k 36

# Extract val2014 (40,504 images)
echo ">>> [2/2] Processing val2014 split..."
python src/scripts/extract_vg_features.py \
    --image_dir data/images/val2014 \
    --output_dir data/vg_features \
    --batch_size 8 \
    --top_k 36

echo "=========================================================="
echo "EXTRACTION 100% COMPLETE!"
echo "Check data/vg_features/ to verify the tensors."
echo "=========================================================="
