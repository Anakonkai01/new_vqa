# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Visual Question Answering (VQA v2.0) system that generates answers token-by-token via LSTM decoding. Compares 4 neural architectures across 2 dimensions: image encoder (Scratch CNN vs. Pretrained ResNet101) and decoder (No Attention vs. Bahdanau Attention).

## Setup

```bash
pip install torch torchvision nltk tqdm matplotlib Pillow
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Common Commands

### Data Preparation
```bash
python src/scripts/1_build_vocab.py
```

### Training (3 phases — run for all 4 models A, B, C, D)

**Phase 1 — Baseline (10 epochs)**:
```bash
python src/train.py --model A --epochs 10 --lr 1e-3 --batch_size 256 \
    --num_workers 8 --augment --weight_decay 1e-5 --early_stopping 3
```

**Phase 2 — Fine-tune (5 epochs)**:
```bash
# A/C: continue training
python src/train.py --model A --epochs 5 --lr 5e-4 --batch_size 256 \
    --resume checkpoints/model_a_resume.pth --augment --weight_decay 1e-5

# B/D: unfreeze ResNet backbone with differential LR
python src/train.py --model B --epochs 5 --lr 5e-4 --batch_size 256 \
    --resume checkpoints/model_b_resume.pth \
    --finetune_cnn --cnn_lr_factor 0.1 --augment --weight_decay 1e-5
```

**Phase 3 — Scheduled Sampling (5 epochs)**:
```bash
python src/train.py --model A --epochs 5 --lr 2e-4 --batch_size 256 \
    --resume checkpoints/model_a_resume.pth \
    --scheduled_sampling --ss_k 5 --augment --weight_decay 1e-5
```

**Local (12GB VRAM)**: Use `--batch_size 128 --grad_accum_steps 2`.

### Evaluation
```bash
python src/evaluate.py --model_type A --checkpoint checkpoints/model_a_best.pth --beam_width 3
python src/compare.py --models A,B,C,D --epoch 20
python src/plot_curves.py --models A,B,C,D --output checkpoints/curves.png
python src/visualize.py --model_type C --epoch 20 --sample_idx 0  # attention heatmap (C/D only)
```

## Architecture

### The 4 Models

| Model | Image Encoder | Decoder | Notes |
|-------|--------------|---------|-------|
| A | SimpleCNN (scratch) | LSTM (no attention) | Baseline |
| B | ResNet101 (pretrained) | LSTM (no attention) | Phase 2 unfreezes layer3+4 |
| C | SimpleCNNSpatial (scratch, 49 regions) | LSTM + Bahdanau Attention | |
| D | ResNetSpatialEncoder (pretrained, 49 regions) | LSTM + Bahdanau Attention | Expected best |

### Data Flow
```
Image (3,224,224) → CNN Encoder → img_feat (B,1024) or (B,49,1024)
Question tokens   → QuestionEncoder (LSTM + attention pooling) → q_feat (B,1024)
                    → GatedFusion + LayerNorm → fusion (B,1024)
                    → Initialize LSTM decoder h_0 (layer 0 only)
                    → LSTM Decoder (teacher forcing in training)
                    → Answer logits (B, seq_len, vocab_size)
```

### Key Components

**`src/models/`**:
- `encoder_cnn.py`: 4 image encoders — `SimpleCNN`, `SimpleCNNSpatial`, `ResNetEncoder`, `ResNetSpatialEncoder`
- `encoder_question.py`: `QuestionEncoder` — LSTM with attention pooling over all token positions (not just last hidden state)
- `decoder_lstm.py`: `LSTMDecoder` — for A/B, no attention
- `decoder_attention.py`: `BahdanauAttention` + `LSTMDecoderWithAttention` — for C/D, input at each step = concat(embedding, context)
- `vqa_models.py`: `VQAModelA/B/C/D` wrappers + `GatedFusion`

**`src/`**:
- `train.py`: Main entry point — 3-phase training, scheduled sampling, resume logic, early stopping
- `dataset.py`: `VQADataset` + `vqa_collate_fn` — loads COCO images on-the-fly, pads sequences
- `vocab.py`: `Vocabulary` — 4 special tokens (`<pad>=0`, `<start>=1`, `<end>=2`, `<unk>=3`)
- `inference.py`: Greedy and beam search decoding (single and batch variants, with/without attention)
- `evaluate.py`: 7 metrics — VQA Accuracy, Exact Match, BLEU-1/2/3/4, METEOR

### GatedFusion (v3.0)
Replaced Hadamard multiplication:
```
gate = sigmoid(Linear(2H→H)([img_feat; q_feat]))
fusion = gate * img_feat + (1-gate) * q_feat
fusion = LayerNorm(fusion)
```
Only layer 0 of decoder LSTM is initialized from fusion; upper layers start from zeros.

### Training Details
- **Answer sampling**: Training randomly picks 1 of 10 human annotations per sample (per epoch)
- **Scheduled sampling epsilon**: Uses `relative_epoch` (not absolute) — critical bug fix in v3.0
- **Checkpoint strategy**: 3-tier — `_resume.pth` (every epoch, overwritten), `_best.pth` (best val loss), `_epoch{10,15,20}.pth` (milestones, kept)
- **Phase 1→2 resume**: Optimizer state is skipped if parameter group layout changed (frozen→unfrozen)
- **GPU auto-detection**: TF32, BFloat16 AMP (Ampere+), fused AdamW (PyTorch ≥ 2.0)

## Data Layout

```
data/
├── raw/
│   ├── images/train2014/    # ~82K COCO images
│   ├── images/val2014/      # ~40K COCO images
│   └── vqa_data_json/       # VQA v2.0 question/annotation JSONs
└── processed/
    ├── vocab_questions.json
    └── vocab_answers.json
checkpoints/
├── model_{a,b,c,d}_resume.pth
├── model_{a,b,c,d}_best.pth
├── model_{a,b,c,d}_epoch{10,15,20}.pth
└── history_model_{a,b,c,d}.json
```

## Documentation

- `DOCUMENTATION.md`: Full technical reference (41KB, Vietnamese)
- `devlog.md`: Development phases 0–8, 5 bugs fixed with root causes
- `vqa_colab_new.ipynb`: Recommended full pipeline for Colab (A100/Blackwell)
- `vqa_local_training.ipynb`: Local training with gradient accumulation
