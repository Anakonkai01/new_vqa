# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Visual Question Answering (VQA)** project. Input: (image, question) → Output: a generated answer/explanation produced token-by-token by an LSTM decoder. The current state-of-the-art version is **Model E**, which uses VQA-E dataset with multi-task captioning.

The main entry point for the full training pipeline is `train_model_e.ipynb`.

## Common Commands

### Build Vocabulary (run from repo root)
```bash
python src/scripts/1_build_vocab.py
```

### Training (Phases 1, 2, 3)
```bash
# Phase 1 — Base training, freeze CNN backbone
python src/train.py --model E --epochs 15 --lr 1e-3 --batch_size 256 --num_workers 12 --augment

# Phase 2 — Fine-tune CLIP backbone (unfreeze top layers)
python src/train.py --model E --epochs 10 --lr 1e-4 --batch_size 256 \
    --resume checkpoints/model_e_resume.pth --finetune_cnn --cnn_lr_factor 0.1

# Phase 3 — Scheduled Sampling
python src/train.py --model E --epochs 5 --lr 2e-4 --batch_size 256 \
    --resume checkpoints/model_e_resume.pth --scheduled_sampling --ss_k 5
```

### Phase 4 — RL SCST
```bash
python src/train_rl.py --model_type E \
    --base_checkpoint checkpoints/model_e_best.pth \
    --epochs 3 --batch_size 32 --lr 1e-5
```

### Evaluation
```bash
python src/evaluate.py --model_type E --checkpoint checkpoints/model_e_best.pth
python src/evaluate.py --model_type E --beam_width 3  # beam search
python src/llm_eval.py  # Gemini LLM-as-judge evaluation
```

### Model Comparison
```bash
python src/compare.py --models A,B,C,D,E --epoch 20
```

### Visualization
```bash
python src/plot_curves.py --models A,B,C,D,E
python src/visualize.py --model_type E --sample_idx 0  # attention heatmap (C/D/E)
```

## Architecture

### Model Variants

| Model | Image Encoder | Fusion | Decoder |
|-------|--------------|--------|---------|
| **A** | SimpleCNN (scratch, global) | GatedFusion | LSTMDecoder |
| **B** | ResNet101 (pretrained, global) | GatedFusion | LSTMDecoder |
| **C** | SimpleCNNSpatial (scratch, 49 regions) | GatedFusion | LSTMDecoderWithAttention |
| **D** | ResNetSpatialEncoder (pretrained, 49 regions) | GatedFusion | LSTMDecoderWithAttention |
| **E** | CLIPViTEncoder (ViT-B/32, 49 regions) | **FiLMFusion** | LSTMDecoderWithAttention |

### Data Flow (all models)
```
Image → Encoder → img_features (B, 49, 1024) or (B, 1024)
Question → BiLSTM → q_feature (B, 1024)
Fusion(img_mean, q_feature) → h_0, c_0 (decoder init state)
LSTMDecoder → logits (B, seq_len, vocab_size)
```

### Model E Specifics (`src/models/vqa_models.py:VQAModelE`)
- **FiLMFusion** modulates spatial image features using `gamma, beta = MLP(q_feature)`, then applies `γ * img + β`.
- Decoder is initialized via **non-linear projections** (`init_h_proj`, `init_c_proj`) rather than a raw repeat.
- FiLM is applied **twice**: once for global `h_0/c_0` init, and again to the 49 spatial regions passed to the attention decoder.

### Dataset (`src/dataset.py`)
- `VQAEDataset` mixes **VQA-E** (question+explanation) with **COCO Captions** for multi-task learning.
- Task tokens `<task_vqa>` and `<task_cap>` are prepended to decoder sequences so the model distinguishes tasks.
- Multi-task mixing: each epoch pulls a portion of captions to supplement VQA-E samples.

### Vocabulary (`src/vocab.py`)
- Tokenizes with `nltk.word_tokenize`.
- Special tokens: `<pad>=0`, `<start>=1`, `<end>=2`, `<unk>=3`, `<task_vqa>=4`, `<task_cap>=5`.
- Saved to `data/processed/vocab_questions.json` and `data/processed/vocab_answers.json`.

## Critical Technical Notes

1. **`max_len` parameter**: Must be 100 in `inference.py`/`evaluate.py`, 60 in `train_rl.py`. VQA-E explanations are long — short `max_len` causes truncation → wrong loss signals.

2. **SequentialLR Resume Bug (PyTorch)**: `train.py` contains a fast-forward loop (`for _ in range(start_epoch): scheduler.step()`) around line 405. **Never remove this.** Without it, PyTorch re-runs warmup on resume, exploding the learning rate.

3. **Multi-task mixing**: Removing the caption-mixing mechanism from `VQAEDataset` degrades grammar quality. Keep the `<task_vqa>` / `<task_cap>` tokens and mixing ratio intact.

4. **Loss computation**: `CrossEntropyLoss(ignore_index=0)` on reshaped tensors — logits `(B*seq, vocab)`, targets `(B*seq,)`.

5. **Attention decoder returns a tuple**: `VQAModelC/D/E.forward()` returns `(logits, coverage_loss)`. `VQAModelA/B.forward()` returns `logits` only.

6. **RL (Phase 4)**: `src/train_rl.py` implements SCST with REINFORCE. The decoder's `sample()` method generates stochastic samples; greedy baseline is the reward baseline. Batch size 128 for RL due to memory.

## Data Paths

```
data/
├── raw/train2014/ & val2014/   # COCO images
├── vqa_e/                      # VQA-E JSON annotations
├── raw/annotations/            # COCO Captions JSONs
└── processed/                  # Built vocab JSONs (output of 1_build_vocab.py)

checkpoints/
├── model_{a,b,c,d,e}_resume.pth   # overwritten each epoch (resume)
├── model_{a,b,c,d,e}_best.pth     # saved on val loss improvement
└── history_model_*.json           # train/val loss history
```

## Hardware Target
- GPU: NVIDIA RTX 5070 Ti (16GB VRAM)
- Batch size 256 (Phase 1-3), 128 (Phase 4 RL), BFloat16 mixed precision
