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
- [x] `evaluate.py` — Exact Match, BLEU-1/2/3/4, METEOR, **ROUGE-L**, **BERTScore** (8 metrics total)
- [x] `compare.py` — side-by-side table cho 4 models, fallback to best checkpoint, ROUGE-L added
- [x] Beam Search — configurable width, n-gram blocking, length-normalized log prob
- [x] `vqa_evaluate_local.ipynb` — full local evaluation notebook (RTX 5070 Ti optimized)
- [x] `outputs/evaluation_results.json` — final results, n=88,488 samples, greedy + beam

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
- [x] `REPORT.md` — full evaluation report (§15–17) with final results, ROUGE-L, efficiency analysis
- [x] `PRESENTATION.md` — Marp slide deck (15 slides + 2 backup), speaker notes

---

## 5. Evaluation Metrics

| Metric | Ý nghĩa | Primary? |
|--------|---------|---------|
| **BLEU-4** | 4-gram precision (primary NLG quality) | ★ Primary |
| **METEOR** | Synonym-aware, stemming (semantic similarity) | ★ Primary |
| **ROUGE-L** | Longest Common Subsequence F1 | ★ Primary |
| **BERTScore** | Contextual embedding similarity | Reference |
| **Exact Match** | Strict string equality (<6%, reference only) | Reference |
| BLEU-1/2/3 | N-gram precision lower orders | Reference |

### Final Results (Greedy, n=88,488, `outputs/evaluation_results.json`)

| Model | BLEU-4 | METEOR | ROUGE-L | BERTScore | EM | Checkpoint |
|-------|--------|--------|---------|-----------|-----|-----------|
| **A** | 0.0915 | 0.3117 | 0.3828 | 0.9008 | 2.83% | model_a_best.pth |
| **B** | 0.1127 | 0.3561 | 0.4237 | 0.9081 | 4.07% | model_b_best.pth |
| **C** | 0.0988 | 0.3271 | 0.3971 | 0.9034 | 4.18% | model_c_best.pth |
| **D** | **0.1159** | **0.3595** | **0.4270** | **0.9085** | **5.88%** | model_d_best.pth |

**vs. Li et al. (ECCV 2018) best:** BLEU-4 9.40% → ours 11.59% **(+23.3%)** · ROUGE-L 36.33% → ours 42.70% **(+17.5%)**

### Beam Search Results (w=3, n-gram blocking=3)

| Model | BLEU-4 | ROUGE-L | Exact Match |
|-------|--------|---------|-------------|
| A | 0.0926 | 0.3823 | 7.46% |
| B | 0.1137 | 0.4230 | 9.94% |
| C | 0.1005 | 0.3972 | 7.57% |
| **D** | **0.1170** | **0.4269** | **11.07%** |

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
