#!/bin/bash
# Flexible Model H Evaluation Script
# Lets you select: phase, checkpoint type (best/resume/both), and beam(s).

set -e

# Defaults
PHASE="phase1"
CKPT_MODE="both"         # best | resume | both
BEAMS="3"                # comma-separated, e.g. 1 or 3 or 1,3
DATASETS="vqa_e vqa_x aokvqa"
BATCH_SIZE=256
NUM_WORKERS=16
CHECKPOINTS_DIR="checkpoints/h"
EXTRA_ARGS=()

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

usage() {
    cat <<EOF
Usage: ./run_eval_h_comprehensive.sh [options]

Options:
  --phase PHASE            Phase name. Default: phase1
                           Example: phase1, phase2
  --ckpt MODE              Checkpoint type: best | resume | both (default: both)
  --beam LIST              Comma list of beam widths. Default: 3
                           Example: --beam 1 or --beam 3 or --beam 1,3
  --datasets "LIST"        Space-separated dataset list. Default: "vqa_e vqa_x aokvqa"
  --batch_size N           Batch size. Default: 256
  --num_workers N          DataLoader workers. Default: 16
  --max_samples N          Optional cap for quick testing
  --use_fasttext           Pass through to evaluate_h.py
  -h, --help               Show this help

Examples:
  ./run_eval_h_comprehensive.sh --phase phase1 --ckpt best --beam 1
  ./run_eval_h_comprehensive.sh --phase phase2 --ckpt resume --beam 3
  ./run_eval_h_comprehensive.sh --phase phase1 --ckpt both --beam 1,3 --max_samples 2000
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)
            PHASE="$2"; shift 2 ;;
        --ckpt)
            CKPT_MODE="$2"; shift 2 ;;
        --beam)
            BEAMS="$2"; shift 2 ;;
        --datasets)
            DATASETS="$2"; shift 2 ;;
        --batch_size)
            BATCH_SIZE="$2"; shift 2 ;;
        --num_workers)
            NUM_WORKERS="$2"; shift 2 ;;
        --max_samples)
            EXTRA_ARGS+=("--max_samples" "$2"); shift 2 ;;
        --use_fasttext)
            EXTRA_ARGS+=("--use_fasttext"); shift 1 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1 ;;
    esac
done

if [[ "$CKPT_MODE" != "best" && "$CKPT_MODE" != "resume" && "$CKPT_MODE" != "both" ]]; then
    echo "Invalid --ckpt value: $CKPT_MODE (use best|resume|both)"
    exit 1
fi

IFS=',' read -r -a BEAM_WIDTHS <<< "$BEAMS"
for b in "${BEAM_WIDTHS[@]}"; do
    if ! [[ "$b" =~ ^[0-9]+$ ]]; then
        echo "Invalid beam value: $b"
        exit 1
    fi
done

CHECKPOINTS=()
if [[ "$CKPT_MODE" == "best" || "$CKPT_MODE" == "both" ]]; then
    CHECKPOINTS+=("model_h_${PHASE}_best")
fi
if [[ "$CKPT_MODE" == "resume" || "$CKPT_MODE" == "both" ]]; then
    CHECKPOINTS+=("model_h_${PHASE}_resume")
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Model H Flexible Evaluation${NC}"
echo -e "${BLUE}Phase: ${PHASE}${NC}"
echo -e "${BLUE}Checkpoint mode: ${CKPT_MODE}${NC}"
echo -e "${BLUE}Beams: ${BEAMS}${NC}"
echo -e "${BLUE}Datasets: ${DATASETS}${NC}"
echo -e "${BLUE}Batch: ${BATCH_SIZE} | Workers: ${NUM_WORKERS}${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

TOTAL_RUNS=$((${#CHECKPOINTS[@]} * ${#BEAM_WIDTHS[@]}))
CURRENT_RUN=0

for checkpoint in "${CHECKPOINTS[@]}"; do
    CHECKPOINT_PATH="${CHECKPOINTS_DIR}/${checkpoint}.pth"
    if [[ ! -f "$CHECKPOINT_PATH" ]]; then
        echo -e "${YELLOW}[SKIP] Missing checkpoint: ${CHECKPOINT_PATH}${NC}"
        continue
    fi

    for beam_width in "${BEAM_WIDTHS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        OUTPUT_NAME="eva_${checkpoint}_beam${beam_width}.json"

        echo -e "${YELLOW}[${CURRENT_RUN}/${TOTAL_RUNS}] Running: ${checkpoint} | beam=${beam_width}${NC}"
        echo -e "  Checkpoint: ${CHECKPOINT_PATH}"
        echo -e "  Output: ${CHECKPOINTS_DIR}/${OUTPUT_NAME}"
        echo ""

        python src/evaluate_h.py \
            --checkpoint "${CHECKPOINT_PATH}" \
            --datasets ${DATASETS} \
            --beam_width "${beam_width}" \
            --batch_size "${BATCH_SIZE}" \
            --num_workers "${NUM_WORKERS}" \
            "${EXTRA_ARGS[@]}"

        echo -e "${GREEN}✓ Completed: ${OUTPUT_NAME}${NC}"
        echo ""
    done
done

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ All requested evaluations completed!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Output files:"
for checkpoint in "${CHECKPOINTS[@]}"; do
    for beam_width in "${BEAM_WIDTHS[@]}"; do
        OUTPUT_FILE="${CHECKPOINTS_DIR}/eva_${checkpoint}_beam${beam_width}.json"
        if [[ -f "$OUTPUT_FILE" ]]; then
            FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
            echo -e "  ${GREEN}✓${NC} ${OUTPUT_FILE} (${FILE_SIZE})"
        fi
    done
done
echo ""
