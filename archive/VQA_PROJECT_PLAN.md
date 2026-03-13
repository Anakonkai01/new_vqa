# VQA Project Plan

**Dự án:** Visual Question Answering — 4 Architectures Comparison  
**Yêu cầu:** Bài 2 (7đ) — CNN + LSTM-Decoder VQA, Attention vs No Attention, Pretrained vs Scratch  
**Trạng thái:** ✅ Hoàn thành

---

## 1. Mục tiêu dự án

Xây dựng hệ thống VQA generative (sinh câu trả lời token-by-token bằng LSTM Decoder) và so sánh công bằng 4 kiến trúc khác nhau dựa trên 2 trục:

| | Không Attention | Có Attention (Bahdanau) |
|---|---|---|
| **Scratch CNN** | Model A | Model C |
| **Pretrained ResNet101** | Model B | Model D |

---

## 2. Kiến trúc tổng quan

```
Image → CNN Encoder → img_feat
                                  ⊙ Hadamard Fusion → h₀ → LSTM Decoder → Answer tokens
Question → LSTM Q-Encoder → q_feat
```

**4 Models:**

| Model | Image Encoder | Decoder | Đặc điểm |
|-------|--------------|---------|-----------|
| A | SimpleCNN (5 conv blocks, scratch) | LSTMDecoder | Baseline đơn giản nhất |
| B | ResNet101 (pretrained, freeze→unfreeze) | LSTMDecoder | Tận dụng ImageNet features |
| C | SimpleCNNSpatial (49 regions, scratch) | LSTMDecoderWithAttention | Spatial attention, train from scratch |
| D | ResNetSpatialEncoder (49 regions, pretrained) | LSTMDecoderWithAttention | Best: pretrained + attention |

---

## 3. Training Strategy — 3 Phases

### Phase 1: Baseline Teacher Forcing (10 epochs)
- LR = 1e-3, ResNet frozen (B/D)
- Pure teacher forcing
- Mục đích: Decoder + Question Encoder hội tụ

### Phase 2: Fine-tune / Continue (5 epochs)
- LR = 5e-4
- Model A/C: continue training
- Model B/D: unfreeze ResNet layer3+layer4, differential LR (backbone × 0.1)
- Mục đích: Adapt pretrained features cho VQA

### Phase 3: Scheduled Sampling (5 epochs)
- LR = 2e-4
- Inverse-sigmoid epsilon decay (k=5)
- Mục đích: Giảm exposure bias

**Tổng: 20 epochs/model**, batch_size=256 thống nhất, augment + weight_decay + early_stopping.

---

## 4. Checklist công việc

### 4.1 Chuẩn bị dữ liệu
- [x] Tải VQA v2.0 dataset (train2014 + val2014 images + JSON)
- [x] Xây dựng `Vocabulary` class (word2idx, idx2word, save/load JSON)
- [x] Script `1_build_vocab.py` — build question + answer vocab
- [x] `VQADataset` class — load ảnh on-the-fly, variable-length collate
- [x] Official train/val split (train2014 vs val2014, không random_split)

### 4.2 Kiến trúc Models
- [x] `SimpleCNN` — 5 conv blocks → global vector (B, 1024)
- [x] `SimpleCNNSpatial` — 5 conv blocks → 49 spatial regions (B, 49, 1024)
- [x] `ResNetEncoder` — ResNet101 pretrained → global vector, freeze/unfreeze
- [x] `ResNetSpatialEncoder` — ResNet101 → 49 regions, freeze/unfreeze
- [x] `QuestionEncoder` — LSTM encoder, shared bởi 4 models
- [x] `LSTMDecoder` — Teacher forcing decoder, embedding dropout
- [x] `BahdanauAttention` — Additive attention over 49 image regions
- [x] `LSTMDecoderWithAttention` — Attention decoder + decode_step()
- [x] `VQAModelA/B/C/D` wrappers — Hadamard fusion + L2 normalization
- [x] `hadamard_fusion()` — element-wise multiplication

### 4.3 Training Pipeline
- [x] `train.py` — Unified training script với CLI args
- [x] Teacher Forcing (Phase 1 + 2)
- [x] Scheduled Sampling — `ss_forward()` (Phase 3)
- [x] Resume checkpoint — model + optimizer + scheduler + scaler + history
- [x] Differential LR — backbone vs head (Phase 2 Model B/D)
- [x] Optimizer resume safety — handle param group mismatch across phases
- [x] Mixed Precision (AMP) — BF16 trên Ampere+, FP16 + GradScaler fallback
- [x] LR Scheduling — ReduceLROnPlateau(factor=0.5, patience=2)
- [x] Gradient Clipping — clip_grad_norm_(max_norm=5.0)

### 4.4 Anti-Overfitting
- [x] Data Augmentation — RandomHorizontalFlip + ColorJitter (`--augment`)
- [x] Weight Decay — L2 regularization (`--weight_decay 1e-5`)
- [x] Early Stopping — patience-based, copy best→milestone (`--early_stopping 3`)
- [x] Embedding Dropout — Dropout(0.5) trong cả 2 decoder
- [x] LSTM Inter-layer Dropout — dropout=0.5 khi num_layers > 1

### 4.5 GPU Optimizations
- [x] cudnn.benchmark = True
- [x] TF32 matmul + convolutions (Ampere+)
- [x] BFloat16 AMP auto-detect
- [x] Fused Adam — `_fused_adam_available()` guard + `fused=True`
- [x] pin_memory, persistent_workers, prefetch_factor=4

### 4.6 Checkpoint Strategy
- [x] Resume checkpoint — overwritten mỗi epoch (1 file/model)
- [x] Best checkpoint — overwritten khi val loss cải thiện (1 file/model)
- [x] Milestone checkpoints — epochs {10, 15, 20} only (tránh tràn Drive)
- [x] Early stopping → copy best→milestone cho compare.py
- [x] History JSON — train/val loss per epoch, saved mỗi epoch

### 4.7 Evaluation & Inference
- [x] `inference.py` — greedy decode + beam search (single + batch)
- [x] `evaluate.py` — VQA Accuracy, Exact Match, BLEU-1/2/3/4, METEOR
- [x] `compare.py` — side-by-side table cho 4 models, fallback to best checkpoint
- [x] VQA Accuracy — official metric: min(matching/3, 1.0) với 10 human annotations
- [x] Beam Search — configurable width, length-normalized log prob

### 4.8 Visualization & Analysis
- [x] `plot_curves.py` — training/val loss curves cho 4 models
- [x] `visualize.py` — attention heatmap overlay (Model C/D)
- [x] Qualitative analysis — ví dụ correct/incorrect predictions
- [x] Error analysis by question type

### 4.9 Google Colab Integration
- [x] `vqa_colab.ipynb` — full pipeline notebook
- [x] Google Drive mount + sync — checkpoints, vocab, outputs
- [x] Drive restore — khôi phục khi runtime restart
- [x] Kaggle dataset download automation

### 4.10 Documentation
- [x] `DOCUMENTATION.md` — full technical documentation
- [x] `VQA_PROJECT_PLAN.md` — this file
- [x] `devlog.md` — development log
- [x] `README.md` — assignment requirements

---

## 5. Evaluation Metrics

| Metric | Ý nghĩa | Dùng cho |
|--------|---------|---------|
| **VQA Accuracy** | Official VQA challenge metric, partial credit | Primary metric |
| **Exact Match** | Strict string equality | Secondary metric |
| **BLEU-1/2/3/4** | N-gram precision (unigram→4-gram) | NLG quality |
| **METEOR** | Synonym-aware, stemming | Semantic similarity |

### Comparison Template

```
Model    VQA Acc   Exact   BLEU-1   BLEU-2   BLEU-3   BLEU-4   METEOR  Checkpoint
------   -------   -----   ------   ------   ------   ------   ------  ----------
A        XX.XX%    XX.XX%  0.XXXX   0.XXXX   0.XXXX   0.XXXX   0.XXXX  model_a_epoch20.pth
B        XX.XX%    XX.XX%  0.XXXX   0.XXXX   0.XXXX   0.XXXX   0.XXXX  model_b_epoch20.pth
C        XX.XX%    XX.XX%  0.XXXX   0.XXXX   0.XXXX   0.XXXX   0.XXXX  model_c_epoch20.pth
D        XX.XX%    XX.XX%  0.XXXX   0.XXXX   0.XXXX   0.XXXX   0.XXXX  model_d_epoch20.pth
```

So sánh theo 3 mốc: Phase 1 (epoch 10), Phase 2 (epoch 15), Phase 3 (epoch 20).

---

## 6. Các vấn đề đã giải quyết

| # | Vấn đề | Giải pháp |
|---|--------|----------|
| 1 | Optimizer crash khi Phase 1→2 (param groups thay đổi) | Compare group counts, skip restore khi khác |
| 2 | `ss_forward` crash thiếu `decoder.dropout` | Thêm `self.dropout = nn.Dropout(0.5)` vào LSTMDecoder |
| 3 | Overfitting (val loss tăng sau epoch 11) | Augment + weight_decay + early_stopping + dropout |
| 4 | Drive tràn 15GB từ per-epoch checkpoints | Milestone-only saving (epochs 10, 15, 20) |
| 5 | Early stopping → compare.py SKIP (thiếu milestone) | Copy best→milestone + fallback trong compare.py |
| 6 | Batch size không công bằng giữa 4 models | Thống nhất batch_size=256 cho tất cả |

---

## 7. Hyperparameters tổng hợp

| Parameter | Giá trị | Ghi chú |
|-----------|--------|---------|
| hidden_size | 1024 | Image + question feature dimension |
| embed_size | 512 | Word embedding dimension |
| num_layers | 2 | LSTM decoder layers |
| dropout | 0.5 | Embedding + LSTM inter-layer |
| batch_size | 256 | Thống nhất cho 4 models |
| num_workers | 8 | DataLoader workers (Colab) |
| optimizer | Adam | fused=True khi available |
| weight_decay | 1e-5 | L2 regularization |
| scheduler | ReduceLROnPlateau | factor=0.5, patience=2 |
| gradient_clip | 5.0 | max_norm |
| early_stopping | 3 | patience epochs |
| ss_k | 5.0 | Scheduled sampling decay speed |
| cnn_lr_factor | 0.1 | Backbone LR multiplier (Phase 2) |
| augment | HFlip + ColorJitter | Train only |
| max_answer_len | ~20 | Decode max length |

---

## 8. Kết luận

Dự án đã **hoàn thành đầy đủ** tất cả yêu cầu:

1. ✅ **4 kiến trúc** VQA (Scratch/Pretrained × Attention/No-Attention)
2. ✅ **LSTM Decoder** sinh câu trả lời token-by-token (generative)
3. ✅ **So sánh công bằng** — cùng batch_size, cùng augmentation, cùng training phases
4. ✅ **Đánh giá đa chiều** — VQA Accuracy (official) + EM + BLEU + METEOR
5. ✅ **Attention visualization** — heatmap overlay cho Model C/D
6. ✅ **Error analysis** — phân tích theo question type
7. ✅ **Anti-overfitting** — 5 kỹ thuật regularization
8. ✅ **3-phase training** — baseline → fine-tune → scheduled sampling
9. ✅ **Production-ready** — resume, Drive sync, milestone checkpoints, GPU optimization
