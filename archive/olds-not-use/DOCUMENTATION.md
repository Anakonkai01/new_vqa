# VQA System — Software Documentation

**Phiên bản:** 2.0.0  
**Ngày cập nhật:** 2025-06-26  
**Repository:** https://github.com/Anakonkai01/new_vqa  
**Branch:** `main`

---

## Mục lục

1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Kiến trúc hệ thống](#2-kiến-trúc-hệ-thống)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Cài đặt & môi trường](#4-cài-đặt--môi-trường)
5. [Mô tả dữ liệu](#5-mô-tả-dữ-liệu)
6. [Mô tả module](#6-mô-tả-module)
7. [Training Pipeline — 3 Phases](#7-training-pipeline--3-phases)
8. [Anti-Overfitting](#8-anti-overfitting)
9. [GPU Optimizations](#9-gpu-optimizations)
10. [Evaluation & Metrics](#10-evaluation--metrics)
11. [Inference & Decoding](#11-inference--decoding)
12. [Google Drive Integration](#12-google-drive-integration)
13. [CLI Reference](#13-cli-reference)
14. [API Reference — Tensor Shapes](#14-api-reference--tensor-shapes)
15. [Checkpoint Strategy](#15-checkpoint-strategy)
16. [Ghi chú kỹ thuật](#16-ghi-chú-kỹ-thuật)

---

## 1. Tổng quan hệ thống

### 1.1 Mục tiêu

Hệ thống **Visual Question Answering (VQA)** nhận đầu vào là một ảnh và một câu hỏi tự nhiên, **sinh ra** câu trả lời token-by-token sử dụng kiến trúc CNN Encoder + LSTM Question Encoder + LSTM Decoder (generative approach).

```
Input:  [Ảnh COCO] + [Câu hỏi tự nhiên]
Output: [Câu trả lời — sinh ra token-by-token bởi LSTM Decoder]
```

### 1.2 Phạm vi

Dự án triển khai và **so sánh công bằng** 4 biến thể kiến trúc dựa trên 2 trục:

| Trục | Lựa chọn A | Lựa chọn B |
|------|-----------|-----------|
| CNN Image Encoder | Train từ đầu (Scratch) | Pretrained ResNet101 |
| Decoder Strategy | Không có Attention | Có Bahdanau Attention |

### 1.3 4 Kiến trúc

| Model | Image Encoder | Attention | Decoder |
|-------|--------------|-----------|---------|
| **A** | SimpleCNN (scratch, 5 conv blocks) | Không | LSTMDecoder |
| **B** | ResNet101 (pretrained, frozen→unfreeze) | Không | LSTMDecoder |
| **C** | SimpleCNNSpatial (scratch, 49 regions) | Bahdanau | LSTMDecoderWithAttention |
| **D** | ResNetSpatialEncoder (pretrained, 49 regions) | Bahdanau | LSTMDecoderWithAttention |

### 1.4 Dataset

- **Nguồn:** VQA v2.0 (Visual QA Challenge) — MS-COCO based
- **Train:** ~443K question-answer pairs trên COCO train2014 (~82K ảnh)
- **Val:** ~214K question-answer pairs trên COCO val2014 (~40K ảnh)
- **Format:** JSON files cho questions + annotations, JPEG ảnh

### 1.5 Training Strategy

3-phase progressive training, tất cả 4 models cùng điều kiện:

| Phase | Epochs | Kỹ thuật | LR |
|-------|--------|----------|-----|
| 1 — Baseline | 10 | Teacher Forcing, ResNet frozen | 1e-3 |
| 2 — Fine-tune | 5 | Unfreeze ResNet L3+L4 (B/D), continue (A/C) | 5e-4 |
| 3 — Scheduled Sampling | 5 | Dần thay GT bằng model prediction | 2e-4 |

**Tổng: 20 epochs/model**, batch_size thống nhất xuyên suốt.

---

## 2. Kiến trúc hệ thống

### 2.1 Tổng quan Pipeline

```
┌────────────────────────────────────────────────────────────────────┐
│                         VQA PIPELINE                               │
│                                                                    │
│  ┌──────────────┐        ┌──────────────────┐                     │
│  │  IMAGE INPUT │        │  QUESTION INPUT  │                     │
│  │ (3, 224, 224)│        │  (max_q_len,)    │                     │
│  └──────┬───────┘        └────────┬─────────┘                     │
│         │                         │                                │
│  ┌──────▼───────┐        ┌────────▼─────────┐                     │
│  │ CNN ENCODER  │        │ QUESTION ENCODER │                     │
│  │ (A/B/C/D)   │        │  (LSTM Encoder)  │                     │
│  └──────┬───────┘        └────────┬─────────┘                     │
│         │                         │                                │
│      img_feat               q_feat (B, 1024)                      │
│         │                         │                                │
│         └──────────┬──────────────┘                                │
│                    │                                               │
│            HADAMARD FUSION = img ⊙ q                              │
│                    │                                               │
│                    │  (B, 1024) → h_0 initial hidden state        │
│            ┌───────▼────────┐                                      │
│            │  LSTM DECODER  │ ← teacher forcing (train)            │
│            │  A/B: no attn  │   autoregressive  (inference)        │
│            │  C/D: + attn   │   scheduled sampling (Phase 3)       │
│            └───────┬────────┘                                      │
│                    │                                               │
│            (B, seq_len, vocab_size)                                │
│                    │                                               │
│            ┌───────▼────────┐                                      │
│            │  ANSWER OUTPUT │ = predicted answer tokens            │
│            └────────────────┘                                      │
└────────────────────────────────────────────────────────────────────┘
```

### 2.2 CNN Output Shapes

| Model | Encoder | Output Shape | Mô tả |
|-------|---------|-------------|-------|
| A | SimpleCNN | `(B, 1024)` | 1 global vector, 5 conv blocks → AdaptiveAvgPool → Linear |
| B | ResNetEncoder | `(B, 1024)` | ResNet101 `[:-1]` → avgpool → Linear(2048→1024) |
| C | SimpleCNNSpatial | `(B, 49, 1024)` | 5 conv blocks → Conv2d(k=1) → 7×7=49 regions |
| D | ResNetSpatialEncoder | `(B, 49, 1024)` | ResNet101 `[:-2]` → Conv2d(k=1) → 49 regions |

### 2.3 Bahdanau Attention (Model C & D)

Tại mỗi decode step `t`:

```
query   = h_t                  (B, hidden)      ← decoder hidden state hiện tại
keys    = img_features         (B, 49, hidden)   ← 49 vùng ảnh
values  = img_features         (B, 49, hidden)

energy  = tanh(W_h(h_t) + W_img(img_features))   → (B, 49, attn_dim)
scores  = v(energy)                               → (B, 49)
alpha   = softmax(scores)                         → (B, 49)
context = Σ(α_i × img_region_i)                   → (B, hidden)

lstm_input = concat(embed_t, context)             → (B, embed + hidden)
```

`alpha` lưu lại để visualize attention heatmap — reshape `(49,)` → `(7, 7)` → upsample lên ảnh.

### 2.4 Hadamard Fusion

```python
fusion = img_feature * q_feature   # element-wise multiplication
```

Kết hợp image và question features. Với attention models (C/D), `img_feature = mean(49 regions)` trước khi fusion.

### 2.5 Teacher Forcing

Trong training, decoder nhận ground-truth token thay vì tự sinh:

```
answer tensor   : [<start>, w1, w2, w3, <end>]

decoder_input   : answer[:, :-1] = [<start>, w1, w2, w3]
decoder_target  : answer[:, 1:]  = [w1, w2, w3, <end>]

Loss = CrossEntropy(logits, target), ignore_index=0 (<pad>)
```

---

## 3. Cấu trúc thư mục

```
vqa_new/
├── data/
│   ├── raw/
│   │   ├── images/
│   │   │   ├── train2014/                   # COCO train images (~82K ảnh)
│   │   │   └── val2014/                     # COCO val images (~40K ảnh)
│   │   └── vqa_json/
│   │       ├── v2_OpenEnded_mscoco_train2014_questions.json
│   │       ├── v2_mscoco_train2014_annotations.json
│   │       ├── v2_OpenEnded_mscoco_val2014_questions.json
│   │       └── v2_mscoco_val2014_annotations.json
│   └── processed/
│       ├── vocab_questions.json             # question vocabulary
│       └── vocab_answers.json               # answer vocabulary
│
├── checkpoints/                             # saved model weights + outputs
│   ├── model_{a,b,c,d}_resume.pth           # resume checkpoint (overwritten each epoch)
│   ├── model_{a,b,c,d}_best.pth             # best val loss checkpoint
│   ├── model_{a,b,c,d}_epoch{10,15,20}.pth  # milestone checkpoints
│   ├── history_model_{a,b,c,d}.json         # train/val loss per epoch
│   ├── training_curves.png                  # output of plot_curves.py
│   ├── attn_model_{c,d}.png                 # attention heatmaps
│   ├── qualitative_analysis.png             # ví dụ dự đoán đúng/sai
│   └── error_analysis_by_type.png           # accuracy by question type
│
├── src/
│   ├── models/
│   │   ├── encoder_cnn.py                   # SimpleCNN, SimpleCNNSpatial,
│   │   │                                    # ResNetEncoder, ResNetSpatialEncoder
│   │   ├── encoder_question.py              # QuestionEncoder (LSTM)
│   │   ├── decoder_lstm.py                  # LSTMDecoder (no attention) + dropout
│   │   ├── decoder_attention.py             # BahdanauAttention + LSTMDecoderWithAttention
│   │   └── vqa_models.py                    # VQAModelA/B/C/D wrappers + hadamard_fusion
│   ├── scripts/
│   │   ├── 1_build_vocab.py                 # build vocab from training data
│   │   └── 2_extract_features.py            # (optional) pre-extract CNN features
│   ├── dataset.py                           # VQADataset + vqa_collate_fn + augmentation
│   ├── vocab.py                             # Vocabulary class (word2idx, idx2word)
│   ├── train.py                             # 3-phase training loop, SS, resume, anti-overfit
│   ├── inference.py                         # greedy decode, beam search, batch decode
│   ├── evaluate.py                          # VQA Accuracy, EM, BLEU-1/2/3/4, METEOR
│   ├── compare.py                           # side-by-side comparison table
│   ├── plot_curves.py                       # training/val loss curves
│   └── visualize.py                         # attention heatmap visualization
│
├── vqa_colab.ipynb                          # Colab notebook (full pipeline)
├── create_dummy_data.py                     # dummy data for pipeline testing
├── DOCUMENTATION.md                         # this file
├── VQA_PROJECT_PLAN.md                      # project plan
├── devlog.md                                # development log
└── README.md                                # assignment requirements
```

---

## 4. Cài đặt & môi trường

### 4.1 Yêu cầu hệ thống

| Thành phần | Tối thiểu | Khuyến nghị (Colab) |
|------------|-----------|-------------------|
| Python | 3.9+ | 3.10+ |
| PyTorch | ≥ 2.0 | 2.x (CUDA 12+) |
| GPU | Bất kỳ NVIDIA sm_70+ | A100 / H100 |
| VRAM | ≥ 8 GB | ≥ 40 GB |
| Disk | ≥ 20 GB | Colab local + Drive |

### 4.2 Cài đặt

```bash
pip install torch torchvision nltk tqdm matplotlib Pillow
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 4.3 Chuẩn bị dữ liệu

**Trên Google Colab (khuyến nghị):**
```bash
# Clone repo
git clone https://github.com/Anakonkai01/new_vqa.git && cd new_vqa

# Tải data từ Kaggle
pip install -q kaggle
kaggle datasets download -d bishoyabdelmassieh/vqa-20-images -p datasets --unzip
kaggle datasets download -d hongnhnnguyntrn/vqa-2-0-val2014 -p datasets --unzip
kaggle datasets download -d hongnhnnguyntrn/vqa2-0-data-json -p datasets --unzip

# Sắp xếp vào cấu trúc project (xem notebook cell chi tiết)
# Build vocab
python src/scripts/1_build_vocab.py
```

---

## 5. Mô tả dữ liệu

### 5.1 Vocabulary

| Token | Index | Ý nghĩa |
|-------|-------|---------|
| `<pad>` | 0 | Padding — không tính loss (`ignore_index=0`) |
| `<start>` | 1 | Bắt đầu chuỗi answer |
| `<end>` | 2 | Kết thúc chuỗi answer |
| `<unk>` | 3 | Token không có trong vocab |
| word_i | 4+ | Các token thực tế |

**Ngưỡng:**
- Question vocab: từ xuất hiện ≥ 3 lần
- Answer vocab: câu trả lời xuất hiện ≥ 5 lần

### 5.2 Dataset Class — `VQADataset`

```python
VQADataset(image_dir, question_json_path, annotations_json_path,
           vocab_q, vocab_a, split='train2014', max_samples=None, augment=False)
```

- Load ảnh on-the-fly (không pre-extract features)
- `split='train2014'|'val2014'` — xác định prefix filename (`COCO_train2014_...` vs `COCO_val2014_...`)
- `augment=True`: RandomHorizontalFlip(0.5) + ColorJitter (chỉ dùng cho train)
- `max_samples`: giới hạn số samples (test pipeline nhanh)

```
__getitem__(idx) → (img_tensor, q_tensor, a_tensor)
  img_tensor : FloatTensor (3, 224, 224) — ImageNet normalized
  q_tensor   : LongTensor  (max_q_len,)
  a_tensor   : LongTensor  (max_a_len,) — [<start>, w1, ..., <end>]
```

### 5.3 Collate Function

`vqa_collate_fn` xử lý variable-length sequences:
- Images: `torch.stack` (cùng shape 3×224×224)
- Questions & Answers: `pad_sequence` với `<pad>=0`

### 5.4 Train / Val Split

Sử dụng **official VQA 2.0 train/val split** — COCO train2014 cho training, COCO val2014 cho validation. Không dùng `random_split`.

---

## 6. Mô tả module

### 6.1 `src/models/encoder_cnn.py`

#### `SimpleCNN` (Model A)
- 5× `conv_block(Conv2d→BN→ReLU→MaxPool)` → `AdaptiveAvgPool(1)` → `Linear(1024→hidden)`
- Output: `(B, hidden_size)` — 1 global vector

#### `SimpleCNNSpatial` (Model C)
- 5× `conv_block` → `Conv2d(kernel=1)` → flatten → permute
- Output: `(B, 49, hidden_size)` — 49 spatial regions, không mean pool

#### `ResNetEncoder` (Model B)
- `ResNet101(pretrained)[:-1]` (bỏ fc, giữ avgpool) → `Linear(2048→hidden)`
- `freeze=True`: đóng băng ResNet weights
- `unfreeze_top_layers()`: mở layer3 + layer4 cho Phase 2 fine-tuning
- `backbone_params()`: trả về params cho differential LR

#### `ResNetSpatialEncoder` (Model D)
- `ResNet101(pretrained)[:-2]` (bỏ avgpool+fc) → `Conv2d(k=1)` → 49 regions
- Tương tự `ResNetEncoder` với `unfreeze_top_layers()` và `backbone_params()`

### 6.2 `src/models/encoder_question.py`

#### `QuestionEncoder`
- `Embedding(vocab_size, embed_size, padding_idx=0)` → `LSTM(num_layers, dropout)` → `h_n[-1]`
- Output: `(B, hidden_size)` — last hidden state
- Dùng bởi tất cả 4 models

### 6.3 `src/models/decoder_lstm.py`

#### `LSTMDecoder` (Model A & B)
- `Embedding` → `Dropout(0.5)` → `LSTM(num_layers, dropout)` → `Linear(hidden→vocab)`
- `forward(encoder_hidden, target_seq)`: teacher forcing, trả về logits
- `self.dropout`: áp dụng lên embedding output để regularize
- LSTM inter-layer dropout=0.5 khi num_layers > 1

### 6.4 `src/models/decoder_attention.py`

#### `BahdanauAttention`
- `W_h(hidden)` + `W_img(img_features)` → tanh → `v` → softmax → context
- Output: `(context, alpha)` — alpha dùng cho visualization

#### `LSTMDecoderWithAttention` (Model C & D)
- LSTM `input_size = embed_size + hidden_size` (concat embedding + attention context)
- `forward(encoder_hidden, img_features, target_seq)`: teacher forcing
- `decode_step(token, hidden, img_features)`: 1 step autoregressive, trả về `(logit, hidden, alpha)`
- Dropout áp dụng cho embedding

### 6.5 `src/models/vqa_models.py`

4 wrapper classes kết hợp encoders + decoder:

| Class | Components | Forward |
|-------|-----------|---------|
| `VQAModelA` | SimpleCNN + QuestionEncoder + LSTMDecoder | img→fusion→decode |
| `VQAModelB` | ResNetEncoder + QuestionEncoder + LSTMDecoder | img→fusion→decode |
| `VQAModelC` | SimpleCNNSpatial + QuestionEncoder + LSTMDecoderWithAttention | img(49)→fusion→decode+attn |
| `VQAModelD` | ResNetSpatialEncoder + QuestionEncoder + LSTMDecoderWithAttention | img(49)→fusion→decode+attn |

Tất cả:
- L2 normalize image features: `F.normalize(feat, p=2, dim=...)`
- Hadamard fusion: `fusion = img_feat * q_feat`
- `h_0 = fusion.unsqueeze(0).repeat(num_layers, 1, 1)`, `c_0 = zeros_like(h_0)`

### 6.6 `src/dataset.py`

Đã mô tả ở §5. Augmentation transforms:
- `augment=False`: Resize(224) → ToTensor → Normalize
- `augment=True`: Resize(224) → RandomHorizontalFlip(0.5) → ColorJitter(0.2, 0.2, 0.2, 0.05) → ToTensor → Normalize

### 6.7 `src/vocab.py`

```python
vocab = Vocabulary()
vocab.word2idx      # dict: word → index
vocab.idx2word      # dict: index → word
vocab.numericalize(text)  # "what color" → [4, 5]
vocab.save(path)    # lưu JSON
vocab.load(path)    # load JSON
```

---

## 7. Training Pipeline — 3 Phases

### 7.1 Phase 1 — Baseline (10 epochs)

**Mục tiêu:** Decoder + Question Encoder hội tụ trước.

```bash
python src/train.py --model A --epochs 10 --lr 1e-3 --batch_size 256 \
    --num_workers 8 --augment --weight_decay 1e-5 --early_stopping 3
```

- Teacher forcing thuần (pure)
- ResNet **frozen** cho Model B, D
- Tất cả 4 models cùng điều kiện → controlled experiment

### 7.2 Phase 2 — Fine-tune / Continue (5 epochs)

**Mục tiêu:** Adapt pretrained features cho VQA domain.

```bash
# Model A, C (continue training, LR giảm)
python src/train.py --model A --epochs 5 --lr 5e-4 --batch_size 256 \
    --resume checkpoints/model_a_resume.pth --augment --weight_decay 1e-5 --early_stopping 3

# Model B, D (unfreeze ResNet layer3+layer4, differential LR)
python src/train.py --model B --epochs 5 --lr 5e-4 --batch_size 256 \
    --resume checkpoints/model_b_resume.pth --finetune_cnn --cnn_lr_factor 0.1 \
    --augment --weight_decay 1e-5 --early_stopping 3
```

**Differential LR (Model B, D):**
- Backbone (layer3+4): `lr × 0.1 = 5e-5` — giữ pretrained knowledge
- Head (decoder + Q-Encoder): `lr = 5e-4` — adapt nhanh hơn

### 7.3 Phase 3 — Scheduled Sampling (5 epochs)

**Mục tiêu:** Giảm exposure bias.

```bash
python src/train.py --model A --epochs 5 --lr 2e-4 --batch_size 256 \
    --resume checkpoints/model_a_resume.pth \
    --scheduled_sampling --ss_k 5 --augment --weight_decay 1e-5 --early_stopping 3
```

**Cơ chế Scheduled Sampling:**
- Mỗi decode step, xác suất `ε` dùng GT token, `(1-ε)` dùng model prediction
- ε decay theo inverse-sigmoid: `ε(epoch) = k / (k + exp(epoch/k))`
- `ss_k=5`: tốc độ decay vừa phải, bắt đầu ~1.0 → giảm dần

### 7.4 Resume Logic

`--resume checkpoints/model_X_resume.pth` khôi phục:
- Model weights
- Optimizer state (nếu param groups khớp)
- Scheduler state
- GradScaler state
- Epoch counter + best val loss + training history

**Phase transition (Phase 1→2):** Optimizer layout thay đổi (frozen→unfreeze thêm param group) → tự động dùng fresh optimizer với LR mới từ CLI args. Không crash.

### 7.5 So sánh sau mỗi Phase

```bash
python src/compare.py --models A,B,C,D --epoch 10   # Phase 1
python src/compare.py --models A,B,C,D --epoch 15   # Phase 2
python src/compare.py --models A,B,C,D --epoch 20   # Phase 3 (final)
```

---

## 8. Anti-Overfitting

| Kỹ thuật | Cách hoạt động | CLI flag |
|----------|---------------|----------|
| **Data Augmentation** | RandomHorizontalFlip + ColorJitter | `--augment` |
| **Weight Decay** | L2 regularization trên model params | `--weight_decay 1e-5` |
| **Early Stopping** | Dừng nếu val loss không cải thiện N epochs | `--early_stopping 3` |
| **Embedding Dropout** | Dropout(0.5) sau embedding layer (cả 2 decoder) | Built-in |
| **LSTM Dropout** | Inter-layer dropout=0.5 (khi num_layers > 1) | Built-in |
| **Gradient Clipping** | `clip_grad_norm_(max_norm=5.0)` | Built-in |
| **LR Scheduling** | ReduceLROnPlateau (factor=0.5, patience=2) | Built-in |

**Early Stopping behavior:** Khi trigger, tự động copy `model_best.pth` → `model_X_epoch{target}.pth` để `compare.py` luôn tìm được checkpoint.

---

## 9. GPU Optimizations

Code tự động detect và áp dụng:

| Optimization | Điều kiện | Hiệu quả |
|---|---|---|
| `cudnn.benchmark = True` | CUDA available | Auto-tune conv algorithms |
| TF32 matmul + conv | Ampere+ (A100, H100...) | ~2× faster, near-FP32 accuracy |
| BFloat16 AMP | `compute_capability >= 8.0` | Wider dynamic range, no GradScaler needed |
| Float16 AMP + GradScaler | Older GPUs | Fallback mixed precision |
| Fused Adam | PyTorch 2.0+ & CUDA | ~10-20% faster optimizer step |
| `pin_memory=True` | CUDA available | Faster CPU→GPU transfer |
| `persistent_workers=True` | num_workers > 0 | Avoid worker respawn overhead |
| `prefetch_factor=4` | num_workers > 0 | Pre-load next batches |

---

## 10. Evaluation & Metrics

### 10.1 Metrics

| Metric | Mô tả | Ý nghĩa |
|--------|-------|---------|
| **VQA Accuracy** | `min(matching_annotations / 3, 1.0)` | Official VQA challenge metric. So sánh prediction với 10 human answers, cho phép partial credit |
| **Exact Match** | prediction == ground truth (strict) | Binary, không partial credit |
| **BLEU-1** | Unigram precision | Đo word overlap |
| **BLEU-2** | Bigram precision | Đo phrase overlap |
| **BLEU-3** | Trigram precision | Đo longer phrase |
| **BLEU-4** | 4-gram precision | Standard NLG metric |
| **METEOR** | Synonym-aware matching + stemming | Hiểu ngữ nghĩa tốt hơn BLEU |

### 10.2 Evaluate từng model

```bash
python src/evaluate.py --model_type A --checkpoint checkpoints/model_a_best.pth
python src/evaluate.py --model_type D --beam_width 3   # beam search
```

### 10.3 So sánh 4 models

```bash
python src/compare.py --models A,B,C,D --epoch 20
python src/compare.py --beam_width 3  # với beam search
```

**Fallback:** Nếu `model_X_epoch{N}.pth` không tồn tại (early stopping), tự động dùng `model_X_best.pth`.

---

## 11. Inference & Decoding

### 11.1 Greedy Decode

Chọn token xác suất cao nhất tại mỗi step. Nhanh nhưng có thể sub-optimal.

```python
# Model A/B (no attention)
greedy_decode(model, img_tensor, q_tensor, vocab_a, max_len=20, device='cpu')

# Model C/D (with attention)
greedy_decode_with_attention(model, img_tensor, q_tensor, vocab_a, max_len=20, device='cpu')
```

### 11.2 Beam Search

Giữ top-k candidates tại mỗi step, trả về sequence có log-probability/length cao nhất.

```python
beam_search_decode(model, img_tensor, q_tensor, vocab_a, beam_width=5)
beam_search_decode_with_attention(model, img_tensor, q_tensor, vocab_a, beam_width=5)
```

### 11.3 Batch Decode

Batch wrappers cho evaluation pipeline:

```python
batch_greedy_decode(model, imgs, qs, vocab_a, device)
batch_greedy_decode_with_attention(model, imgs, qs, vocab_a, device)
batch_beam_search_decode(model, imgs, qs, vocab_a, beam_width=5)
batch_beam_search_decode_with_attention(model, imgs, qs, vocab_a, beam_width=5)
```

---

## 12. Google Drive Integration

Notebook `vqa_colab.ipynb` tự động sync kết quả lên Google Drive:

### 12.1 Cấu trúc Drive

```
MyDrive/VQA_Project/
├── checkpoints/   # resume, best, milestone checkpoints + history JSON
├── vocab/         # vocab_questions.json, vocab_answers.json
└── outputs/       # training_curves.png, attention maps, analysis plots
```

### 12.2 Sync Points

| Thời điểm | Dữ liệu | Mục đích |
|-----------|---------|---------|
| Sau Build Vocab | vocab JSON files | Khôi phục vocab khi restart |
| Sau Phase 1 | resume + best + epoch10 + history | Resume Phase 2 |
| Sau Phase 2 | resume + best + epoch15 + history | Resume Phase 3 |
| Sau Phase 3 | resume + best + epoch20 + history | Final results |
| Sau Plot Curves | training_curves.png | Lưu output |
| Sau Attention Viz | attn_model_*.png | Lưu output |
| Sau Analysis | qualitative + error analysis | Lưu output |

### 12.3 Restore khi Runtime Restart

```python
# Chạy cell restore trong notebook
restore_from_drive('checkpoints', 'checkpoints')
restore_from_drive('vocab', 'data/processed')
```

---

## 13. CLI Reference

### 13.1 `train.py`

```
python src/train.py [OPTIONS]

Options:
  --model {A,B,C,D}        Model architecture (default: A)
  --epochs N               Number of training epochs (default: 10)
  --lr FLOAT               Learning rate (default: 1e-3)
  --batch_size N           Batch size (default: 128)
  --num_workers N          DataLoader workers (default: 4)
  --resume PATH            Resume from checkpoint
  --scheduled_sampling     Enable Scheduled Sampling
  --ss_k FLOAT             SS inverse-sigmoid decay speed (default: 5.0)
  --finetune_cnn           Unfreeze ResNet layer3+4 (B/D only)
  --cnn_lr_factor FLOAT    Backbone LR multiplier (default: 0.1)
  --weight_decay FLOAT     L2 regularization (default: 1e-5)
  --early_stopping N       Patience epochs (0=disabled)
  --augment                Enable data augmentation
```

### 13.2 `evaluate.py`

```
python src/evaluate.py [OPTIONS]

Options:
  --model_type {A,B,C,D}   Model architecture
  --checkpoint PATH        Checkpoint path (default: model_X_epoch10.pth)
  --num_samples N          Limit samples for speed
  --beam_width N           Beam search width (default: 1 = greedy)
```

### 13.3 `compare.py`

```
python src/compare.py [OPTIONS]

Options:
  --epoch N                Epoch checkpoint to load (default: 10)
  --models STR             Comma-separated model list (default: A,B,C,D)
  --num_samples N          Limit samples
  --beam_width N           Beam search width (default: 1)
```

### 13.4 `plot_curves.py`

```
python src/plot_curves.py [OPTIONS]

Options:
  --models STR             Models to plot (default: A,B,C,D)
  --output PATH            Output image path
```

### 13.5 `visualize.py`

```
python src/visualize.py [OPTIONS]

Options:
  --model_type {C,D}       Model with attention
  --epoch N                Checkpoint epoch
  --sample_idx N           Sample index (default: 0)
  --output PATH            Output image path
```

---

## 14. API Reference — Tensor Shapes

| Tensor | Shape | Mô tả |
|--------|-------|-------|
| `images` | `(B, 3, 224, 224)` | Batch ảnh, ImageNet normalized |
| `questions` | `(B, Q)` | Question token indices, padded |
| `answers` | `(B, A)` | Answer tokens `[<start>, ..., <end>]`, padded |
| `decoder_input` | `(B, A-1)` | `answer[:, :-1]` |
| `decoder_target` | `(B, A-1)` | `answer[:, 1:]` |
| `img_feat` (A/B) | `(B, 1024)` | Global image vector |
| `img_features` (C/D) | `(B, 49, 1024)` | Spatial regions |
| `q_feat` | `(B, 1024)` | Question feature |
| `fusion` | `(B, 1024)` | `img_feat ⊙ q_feat` |
| `h_0, c_0` | `(num_layers, B, 1024)` | Initial decoder state |
| `logits` | `(B, seq_len, vocab_size)` | Decoder output |
| `alpha` | `(B, 49)` | Attention weights (C/D) |
| `context` | `(B, 1024)` | Weighted sum of 49 regions |

### Loss Computation

```python
criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore <pad>

loss = criterion(
    logits.view(-1, vocab_size),             # (B*seq_len, vocab_size)
    decoder_target.contiguous().view(-1)     # (B*seq_len,)
)
```

---

## 15. Checkpoint Strategy

### 15.1 Storage-Efficient Design

Tránh tràn bộ nhớ Drive (15GB free) bằng chiến lược tiết kiệm:

| Checkpoint | Tần suất | Ghi đè? | Mục đích |
|-----------|---------|---------|---------|
| `model_X_resume.pth` | Mỗi epoch | Có | Resume training sau disconnect |
| `model_X_best.pth` | Khi val loss cải thiện | Có | Best model cho evaluation |
| `model_X_epoch{10,15,20}.pth` | Milestone epochs only | Không | Cho `compare.py` |

### 15.2 Resume Checkpoint Contents

```python
{
    'epoch': int,                    # last completed epoch
    'model_state_dict': dict,        # model weights
    'optimizer_state_dict': dict,    # optimizer state
    'scheduler_state_dict': dict,    # LR scheduler state
    'scaler_state_dict': dict,       # AMP GradScaler state
    'best_val_loss': float,          # best val loss so far
    'history': {                     # training history
        'train_loss': [float, ...],
        'val_loss': [float, ...]
    }
}
```

### 15.3 Early Stopping + Milestone

Khi early stopping trigger trước milestone epoch:
1. `train.py` copy `model_best.pth` → `model_X_epoch{target}.pth`
2. `compare.py` fallback: nếu epoch-specific không có → dùng `model_best.pth`

---

## 16. Ghi chú kỹ thuật

### 16.1 Quy ước đặt tên

| Đối tượng | Convention | Ví dụ |
|-----------|-----------|-------|
| Model class | PascalCase | `VQAModelA`, `VQAModelD` |
| Checkpoint file | `model_{a,b,c,d}_{type}.pth` | `model_a_best.pth` |
| History file | `history_model_{a,b,c,d}.json` | `history_model_a.json` |
| Encoder file | `encoder_{type}.py` | `encoder_cnn.py` |

### 16.2 Image Normalization

```python
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```
ImageNet statistics — bắt buộc cho cả pretrained ResNet lẫn scratch CNN (consistency).

### 16.3 L2 Feature Normalization

```python
# Model A/B: normalize global vector
img_feature = F.normalize(img_feature, p=2, dim=1)

# Model C/D: normalize each spatial region independently
img_features = F.normalize(img_features, p=2, dim=-1)
```

Loại bỏ ảnh hưởng magnitude, chỉ giữ hướng vector → Hadamard fusion ổn định hơn.

### 16.4 Scheduled Sampling — ss_forward()

`ss_forward()` trong `train.py` bypass `model.forward()`, trực tiếp gọi encoders và decoder step-by-step. Mỗi step:
- Với xác suất ε: dùng GT token (teacher forcing)
- Với xác suất 1-ε: dùng `argmax(logit).detach()` (model prediction)

Model A/B: gọi `model.decoder.dropout(model.decoder.embedding(tok))` + `model.decoder.lstm()` + `model.decoder.fc()`  
Model C/D: gọi `model.decoder.decode_step(tok, hidden, img_features)`

### 16.5 Các lỗi đã gặp và fix

| Lỗi | Nguyên nhân | Fix |
|-----|-------------|-----|
| `ValueError: different number of parameter groups` | Phase 1→2 optimizer layout thay đổi (freeze→unfreeze) | Compare group counts, skip optimizer restore khi khác |
| `AttributeError: 'LSTMDecoder' has no attribute 'dropout'` | ss_forward gọi dropout chưa có | Thêm `self.dropout = nn.Dropout(0.5)` vào LSTMDecoder |
| Val loss tăng sau epoch 11 | Overfitting | 4 regularization techniques (augment, weight_decay, early_stopping, dropout) |
| Drive tràn 15GB | Per-epoch checkpoints quá nhiều | Milestone-only saving (epochs 10, 15, 20) |
| Early stopping → compare.py SKIP | Milestone checkpoint chưa được tạo | Copy best→milestone khi early stop + fallback trong compare.py |
