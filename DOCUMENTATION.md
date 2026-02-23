# VQA System — Software Documentation

**Phiên bản:** 1.0.0  
**Ngày:** 2026-02-23  
**Repository:** https://github.com/Anakonkai01/new_vqa  
**Branch chính:** `experiment/model-a`

---

## Mục lục

1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Kiến trúc hệ thống](#2-kiến-trúc-hệ-thống)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Cài đặt & môi trường](#4-cài-đặt--môi-trường)
5. [Mô tả dữ liệu](#5-mô-tả-dữ-liệu)
6. [Mô tả module](#6-mô-tả-module)
7. [API Reference](#7-api-reference)
8. [Pipeline thực thi](#8-pipeline-thực-thi)
9. [Cấu hình hyperparameters](#9-cấu-hình-hyperparameters)
10. [Hướng dẫn sử dụng](#10-hướng-dẫn-sử-dụng)
11. [Kết quả đánh giá (template)](#11-kết-quả-đánh-giá-template)
12. [Ghi chú kỹ thuật](#12-ghi-chú-kỹ-thuật)

---

## 1. Tổng quan hệ thống

### 1.1 Mục tiêu

Hệ thống **Visual Question Answering (VQA)** nhận đầu vào là một ảnh và một câu hỏi tự nhiên, sinh ra câu trả lời dạng văn bản (generative) sử dụng kiến trúc CNN + LSTM-Decoder.

```
Input:  [Ảnh] + [Câu hỏi dạng text]
Output: [Câu trả lời dạng text — sinh ra token-by-token]
```

### 1.2 Phạm vi

Dự án triển khai và so sánh **4 biến thể kiến trúc** dựa trên 2 trục:

| Trục | Lựa chọn |
|------|----------|
| CNN Image Encoder | Train từ đầu (Scratch) vs Pretrained ResNet101 |
| Decoder Strategy | Không có Attention vs Có Bahdanau Attention |

### 1.3 Dataset

- **Nguồn:** VQA v2 (Visual QA Challenge) — COCO-based
- **Train set:** `v2_OpenEnded_mscoco_train2014_questions.json` + `v2_mscoco_train2014_annotations.json`
- **Ảnh:** MS-COCO `train2014` (~82,783 ảnh, ~13GB)
- **Vocabulary:** Top-K câu trả lời thường gặp nhất

---

## 2. Kiến trúc hệ thống

### 2.1 Tổng quan pipeline

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
│  │  (A/B/C/D)   │        │  (LSTM Encoder)  │                     │
│  └──────┬───────┘        └────────┬─────────┘                     │
│         │                         │                                │
│  No Attn│ (batch, 1024)           │ (batch, 1024)                  │
│  ───────┼─────────────────────────┤                                │
│         │          HADAMARD FUSION│= img ⊙ q                      │
│         │         (batch, 1024)   │                                │
│         │                         │                                │
│  Attn   │ (batch, 49, 1024)       │                                │
│  ───────┼─────────────────────────┤                                │
│         │      mean(dim=1) → 1024 │                                │
│         │      HADAMARD FUSION    │                                │
│         │                         │                                │
│         └──────────┬──────────────┘                                │
│                    │  (batch, 1024) = initial hidden h_0           │
│            ┌───────▼────────┐                                      │
│            │  LSTM DECODER  │ ← teacher forcing during training    │
│            │  (A/B: no attn)│   autoregressive during inference    │
│            │  (C/D: + attn) │                                      │
│            └───────┬────────┘                                      │
│                    │                                               │
│            (batch, seq_len, vocab_size)                            │
│                    │                                               │
│            ┌───────▼────────┐                                      │
│            │     OUTPUT     │ = predicted answer tokens            │
│            └────────────────┘                                      │
└────────────────────────────────────────────────────────────────────┘
```

### 2.2 Chi tiết 4 kiến trúc

| Model | Image Encoder | Decoder | CNN Output Shape | Đặc điểm |
|-------|--------------|---------|-----------------|----------|
| **A** | `SimpleCNN` (scratch) | `LSTMDecoder` | `(batch, 1024)` | Baseline đơn giản nhất |
| **B** | `ResNetEncoder` (pretrained, frozen) | `LSTMDecoder` | `(batch, 1024)` | Feature chất lượng cao |
| **C** | `SimpleCNNSpatial` (scratch) | `LSTMDecoderWithAttention` | `(batch, 49, 1024)` | Học attention từ đầu |
| **D** | `ResNetSpatialEncoder` (pretrained, frozen) | `LSTMDecoderWithAttention` | `(batch, 49, 1024)` | Mạnh nhất lý thuyết |

### 2.3 Bahdanau Attention (Model C và D)

```
Tại mỗi bước decoding t:

  query   = h_t               (batch, hidden_size)    ← hidden state hiện tại
  keys    = img_features       (batch, 49, hidden_size) ← 49 vùng ảnh
  values  = img_features       (batch, 49, hidden_size)

  energy  = tanh(W_h(h_t) + W_img(img_features))     (batch, 49, attn_dim)
  scores  = v(energy)                                 (batch, 49)
  alpha   = softmax(scores)                           (batch, 49)
  context = Σ(alpha_i * img_regions_i)                (batch, hidden_size)

  lstm_input = concat(embed_t, context)               (batch, embed_size + hidden_size)
```

---

## 3. Cấu trúc thư mục

```
vqa_new/
├── data/
│   ├── raw/
│   │   ├── images/
│   │   │   └── train2014/               # COCO images (~13GB)
│   │   └── vqa_json/
│   │       ├── v2_OpenEnded_mscoco_train2014_questions.json
│   │       ├── v2_mscoco_train2014_annotations.json
│   │       ├── v2_OpenEnded_mscoco_val2014_questions.json
│   │       └── v2_mscoco_val2014_annotations.json
│   └── processed/
│       ├── vocab_questions.json         # question vocabulary
│       └── vocab_answers.json           # answer vocabulary
│
├── checkpoints/                         # saved model weights + history
│   ├── model_a_epoch{n}.pth
│   ├── model_b_epoch{n}.pth
│   ├── model_c_epoch{n}.pth
│   ├── model_d_epoch{n}.pth
│   ├── history_model_a.json             # train/val loss per epoch
│   ├── history_model_b.json
│   ├── history_model_c.json
│   ├── history_model_d.json
│   ├── training_curves.png              # output của plot_curves.py
│   ├── attn_model_c.png                 # output của visualize.py
│   └── attn_model_d.png
│
├── src/
│   ├── models/
│   │   ├── encoder_cnn.py               # 4 CNN image encoder classes
│   │   ├── encoder_question.py          # LSTM question encoder
│   │   ├── decoder_lstm.py              # LSTM decoder (no attention)
│   │   ├── decoder_attention.py         # Bahdanau attention + LSTM decoder
│   │   └── vqa_models.py                # 4 VQA wrapper models (A/B/C/D)
│   ├── scripts/
│   │   ├── 1_build_vocab.py             # xây dựng vocabulary từ dữ liệu
│   │   └── 2_extract_features.py        # (optional) pre-extract CNN features
│   ├── dataset.py                       # VQADatasetA + vqa_collate_fn
│   ├── vocab.py                         # Vocabulary class
│   ├── train.py                         # training loop
│   ├── inference.py                     # greedy decode (test 1 sample)
│   ├── evaluate.py                      # đánh giá trên val set
│   ├── compare.py                       # so sánh 4 model cùng lúc
│   ├── plot_curves.py                   # vẽ training/val loss curves
│   └── visualize.py                     # attention heatmap cho C/D
│
├── create_dummy_data.py                 # tạo dummy data để test pipeline
├── devlog.md                            # dev log + context cho chat AI
├── VQA_PROJECT_PLAN.md                  # kế hoạch dự án ban đầu
├── DOCUMENTATION.md                     # file này
└── README.md
```

---

## 4. Cài đặt & môi trường

### 4.1 Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|------------|---------|
| Python | 3.9+ |
| PyTorch | ≥ 1.12 (sm_70+ cho CUDA, hoặc CPU) |
| RAM | ≥ 8GB |
| Disk | ≥ 15GB (cho COCO dataset) |
| GPU (optional) | NVIDIA sm_70+ (Kaggle T4/P100) |

> ⚠️ **NVIDIA MX330 (sm_61) không tương thích CUDA với PyTorch ≥ 1.12.** Dùng CPU local hoặc Kaggle/Colab.

### 4.2 Cài đặt dependencies

```bash
# Tạo conda environment
conda create -n vqa python=3.9
conda activate vqa

# PyTorch (CPU)
pip install torch torchvision

# Các thư viện khác
pip install nltk matplotlib pillow tqdm

# Download NLTK data (cần cho METEOR metric)
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 4.3 Chuẩn bị dữ liệu

**Option A — Dummy data (test pipeline nhanh):**
```bash
python create_dummy_data.py
python src/scripts/1_build_vocab.py
```

**Option B — Real COCO data:**
```bash
# Download ảnh (~13GB)
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip -d data/raw/images/

# VQA annotations đặt vào data/raw/vqa_json/
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

Vocab được lưu dưới dạng JSON:
```json
{
  "word2idx": {"<pad>": 0, "<start>": 1, ...},
  "idx2word": {"0": "<pad>", "1": "<start>", ...}
}
```

### 5.2 Dataset class — `VQADatasetA`

```
__getitem__(idx) → (img_tensor, q_tensor, a_tensor)
  img_tensor : FloatTensor (3, 224, 224) — normalized [0.485,0.456,0.406] ± [0.229,0.224,0.225]
  q_tensor   : LongTensor  (max_q_len,)  — numericalized question tokens
  a_tensor   : LongTensor  (max_a_len,)  — [<start>, w1, w2, ..., <end>]
```

### 5.3 Train / Validation Split

- **Tỉ lệ:** 90% train / 10% validation
- **Seed:** `manual_seed(42)` — **phải nhất quán** giữa `train.py` và `evaluate.py`
- **Lý do:** Đảm bảo val set trong evaluate.py chính xác là tập model chưa thấy khi train

### 5.4 Teacher Forcing

Trong training, decoder nhận ground-truth token thay vì token tự sinh ra:

```
answer tensor   : [<start>, w1, w2, w3, <end>]

decoder_input   : answer[:, :-1] = [<start>, w1, w2, w3]
decoder_target  : answer[:, 1:]  = [w1, w2, w3, <end>]

Loss = CrossEntropy(logits vs decoder_target), ignore_index=0
```

---

## 6. Mô tả module

### 6.1 `src/models/encoder_cnn.py`

Chứa 4 CNN image encoder classes.

#### `SimpleCNN`
- **Mục đích:** Baseline CNN train từ đầu, không dùng pretrained weights
- **Input:** `(batch, 3, 224, 224)`
- **Output:** `(batch, output_size)` — 1 vector đại diện cho ảnh
- **Kiến trúc:** 5× `conv_block(Conv2d → BN → ReLU → MaxPool)` → `AdaptiveAvgPool(1)` → `Linear`
- **Dùng bởi:** `VQAmodelA`

#### `SimpleCNNSpatial`
- **Mục đích:** Giống SimpleCNN nhưng giữ lại 49 vùng spatial cho attention
- **Input:** `(batch, 3, 224, 224)`
- **Output:** `(batch, 49, output_size)` — 49 regional feature vectors
- **Khác SimpleCNN:** Không dùng `AdaptiveAvgPool`, thay bằng `Conv2d(kernel=1)` → `flatten(2)` → `permute`
- **Dùng bởi:** `VQAModelC`

#### `ResNetEncoder`
- **Mục đích:** Sử dụng ResNet101 pretrained ImageNet, frozen
- **Input:** `(batch, 3, 224, 224)`
- **Output:** `(batch, output_size)`
- **Kiến trúc:** `ResNet101[:-1]` (bỏ fc, giữ avgpool) → `flatten` → `Linear(2048 → output_size)`
- **Dùng bởi:** `VQAModelB`

#### `ResNetSpatialEncoder`
- **Mục đích:** ResNet101 pretrained giữ spatial features
- **Input:** `(batch, 3, 224, 224)`
- **Output:** `(batch, 49, output_size)` — 49 pretrained regional features
- **Kiến trúc:** `ResNet101[:-2]` (bỏ avgpool và fc) → `Conv2d(kernel=1)` → `flatten(2)` → `permute`
- **Dùng bởi:** `VQAModelD`

---

### 6.2 `src/models/encoder_question.py`

#### `QuestionEncoder`
- **Mục đích:** Encode câu hỏi thành 1 vector ngữ nghĩa
- **Input:** `(batch, max_q_len)` — token indices
- **Output:** `(batch, hidden_size)` — last hidden state của LSTM
- **Kiến trúc:** `Embedding(vocab_size, embed_size)` → `LSTM(embed_size, hidden_size, num_layers)` → lấy `h_n[-1]`
- **Dùng bởi:** Tất cả 4 model

---

### 6.3 `src/models/decoder_lstm.py`

#### `LSTMDecoder`
- **Mục đích:** LSTM decoder không có attention, dùng teacher forcing
- **Input:** `(h_0, c_0)`, `target_seq (batch, seq_len)`
- **Output:** `logits (batch, seq_len, vocab_size)`
- **Kiến trúc:** `Embedding` → `LSTM` → `Linear(hidden → vocab_size)`
- **`decode_step(token, hidden)`:** 1 bước autoregressive cho inference, trả về `(logit, hidden_new)`
- **Dùng bởi:** `VQAmodelA`, `VQAModelB`

---

### 6.4 `src/models/decoder_attention.py`

#### `BahdanauAttention`
- **Mục đích:** Tính attention weights giữa decoder hidden state và 49 image regions
- **Input:** `hidden (batch, hidden_size)`, `img_features (batch, 49, hidden_size)`
- **Output:** `context (batch, hidden_size)`, `alpha (batch, 49)`
- **Công thức:**
  ```
  energy = tanh(W_h(hidden) + W_img(img_features))   → (batch, 49, attn_dim)
  alpha  = softmax(v(energy), dim=1)                  → (batch, 49)
  context = sum(alpha.unsqueeze(2) * img_features, dim=1)
  ```

#### `LSTMDecoderWithAttention`
- **Mục đích:** LSTM decoder kết hợp Bahdanau Attention
- **Khác LSTMDecoder:** LSTM `input_size = embed_size + hidden_size` (ghép embedding với context vector)
- **`forward(encoder_hidden, img_features, target_seq)`:** Teacher forcing, trả về `logits`
- **`decode_step(token, hidden, img_features)`:** 1 bước autoregressive, trả về `(logit, hidden_new, alpha)`
  - `alpha` dùng để visualize attention heatmap
- **Dùng bởi:** `VQAModelC`, `VQAModelD`

---

### 6.5 `src/models/vqa_models.py`

Wrapper kết hợp encoders + decoder thành end-to-end model.

#### `hadamard_fusion(img_feature, q_feature)`
```python
return img_feature * q_feature   # element-wise multiplication
```
Dùng để kết hợp image và question features thành initial hidden state.

#### `VQAmodelA` / `VQAModelB` / `VQAModelC` / `VQAModelD`
**Chung cho tất cả:**
- **`__init__`:** khởi tạo `i_encoder`, `q_encoder`, `decoder`
- **`forward(images, questions, target_seq)`:** training forward pass
- **`num_layers`:** stored để tạo `h_0` đúng shape

**Riêng cho C/D:** truyền thêm `img_features (batch, 49, hidden)` vào decoder mỗi bước

---

### 6.6 `src/dataset.py`

#### `VQADatasetA`
```python
VQADatasetA(image_dir, question_json_path, annotations_json_path, vocab_q, vocab_a)
```
- Load ảnh on-the-fly (không pre-extract)
- Transform: `Resize(224,224)` → `ToTensor()` → `Normalize(ImageNet mean/std)`
- Answer tokenization: `[<start>] + tokens + [<end>]`

#### `vqa_collate_fn`
- Xử lý variable-length sequences trong một batch
- Pad questions và answers về cùng độ dài với `<pad>=0`
- Trả về `(imgs, questions, answers)` tensors

---

### 6.7 `src/vocab.py`

#### `Vocabulary`
```python
vocab.word2idx     # dict: word → index
vocab.idx2word     # dict: index → word
vocab.numericalize(text)  # str → List[int]
vocab.save(path)          # lưu JSON
vocab.load(path)          # load JSON
```

---

### 6.8 `src/train.py`

**Config (đầu file):**
```python
MODEL_TYPE = 'A'    # thay đổi để train model khác: 'B', 'C', 'D'
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Factory function:**
```python
def get_model(model_type, vocab_q_size, vocab_a_size) → nn.Module
```

**Outputs:**
- `checkpoints/model_{type}_epoch{n}.pth` — model weights sau mỗi epoch
- `checkpoints/history_model_{type}.json` — `{"train_loss": [...], "val_loss": [...]}`, cập nhật sau mỗi epoch

---

### 6.9 `src/inference.py`

```python
greedy_decode(model, img_tensor, q_tensor, vocab_a, max_len=20, device='cpu')
# → str  (dùng cho Model A, B)

greedy_decode_with_attention(model, img_tensor, q_tensor, vocab_a, max_len=20, device='cpu')
# → str  (dùng cho Model C, D)

get_model(model_type, vocab_q_size, vocab_a_size)
# → nn.Module  (factory, dùng chung với evaluate/compare)
```

---

### 6.10 `src/evaluate.py`

```bash
python src/evaluate.py --model_type A
python src/evaluate.py --model_type C --checkpoint checkpoints/model_c_epoch5.pth
python src/evaluate.py --model_type B --num_samples 200
```

**Metrics:**
| Metric | Mô tả |
|--------|-------|
| Exact Match | % câu trả lời khớp hoàn toàn với ground truth |
| BLEU-1 | Unigram precision |
| BLEU-2 | Bigram precision |
| BLEU-3 | Trigram precision |
| BLEU-4 | 4-gram precision (metric chính cho text generation) |
| METEOR | Xét synonym + stemming, tốt hơn BLEU về ngữ nghĩa |

Hàm `evaluate()` trả về dict kết quả — dùng được từ `compare.py`.

---

### 6.11 `src/compare.py`

```bash
python src/compare.py                        # cả 4 model, epoch 10
python src/compare.py --epoch 5
python src/compare.py --models A,C --num_samples 100
```

**Output mẫu:**
```
Model     Exact Match     BLEU-1  Checkpoint
----------------------------------------------------------------------
A          72.30%        0.7412  checkpoints/model_a_epoch10.pth
B          79.15%        0.8103  checkpoints/model_b_epoch10.pth
C          74.80%        0.7680  checkpoints/model_c_epoch10.pth
D          83.42%        0.8560  checkpoints/model_d_epoch10.pth
```
Model thiếu checkpoint → tự động SKIP, không crash.

---

### 6.12 `src/plot_curves.py`

```bash
python src/plot_curves.py                    # vẽ cả 4 model
python src/plot_curves.py --models A,C
python src/plot_curves.py --output results/curves.png
```

- Đọc `checkpoints/history_model_*.json`
- Vẽ 2 subplot: Training Loss | Validation Loss
- Lưu ảnh PNG, không cần display server (`matplotlib.use('Agg')`)

---

### 6.13 `src/visualize.py`

```bash
python src/visualize.py --model_type C               # sample đầu tiên
python src/visualize.py --model_type D --epoch 5
python src/visualize.py --model_type C --sample_idx 10 --output results/attn.png
```

- Chỉ dùng cho Model C và D (có attention)
- Mỗi token được sinh ra → 1 panel gồm ảnh gốc + heatmap `alpha` (jet colormap)
- `alpha (49,)` reshape thành `(7, 7)` → upsample về `(224, 224)` → overlay

---

## 7. API Reference

### 7.1 Tensor shapes tổng hợp

| Tensor | Shape | Ý nghĩa |
|--------|-------|---------|
| `images` | `(B, 3, 224, 224)` | Batch ảnh đã normalize |
| `questions` | `(B, Q)` | Question token indices |
| `answers` | `(B, A)` | Answer token indices, bắt đầu `<start>`, kết thúc `<end>` |
| `decoder_input` | `(B, A-1)` | `answer[:, :-1]` |
| `decoder_target` | `(B, A-1)` | `answer[:, 1:]` |
| `img_feat_flat` | `(B, 1024)` | CNN output không spatial (Model A/B) |
| `img_feat_spatial` | `(B, 49, 1024)` | CNN output giữ spatial (Model C/D) |
| `q_feat` | `(B, 1024)` | LSTM question encoder output |
| `fusion` | `(B, 1024)` | Hadamard product `img ⊙ q` |
| `h_0, c_0` | `(num_layers, B, 1024)` | Initial decoder state |
| `logits` | `(B, seq_len, vocab_size)` | Decoder output |
| `alpha` | `(B, 49)` | Attention weights (C/D only) |
| `context` | `(B, 1024)` | Weighted sum qua 49 vùng |

### 7.2 Loss function

```python
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Reshape 3D → 2D trước khi tính loss:
loss = criterion(
    logits.view(-1, vocab_size),               # (B * seq_len, vocab_size)
    decoder_target.contiguous().view(-1)        # (B * seq_len,)
)
```

`ignore_index=0` → không tính gradient tại vị trí `<pad>`.

---

## 8. Pipeline thực thi

### 8.1 Training

```
1. Load vocab_q, vocab_a từ JSON
2. Khởi tạo VQADatasetA
3. random_split 90/10 với manual_seed(42)
4. Tạo DataLoader (train + val)
5. Khởi tạo model = get_model(MODEL_TYPE, ...)
6. Khởi tạo Adam optimizer + CrossEntropyLoss(ignore_index=0)
7. Với mỗi epoch:
   a. model.train()
   b. Với mỗi batch:
      - Forward: logits = model(imgs, questions, answer[:, :-1])
      - Loss: CE(logits.view(-1,V), answer[:,1:].view(-1))
      - Backward + clip_grad_norm_(max_norm=5.0)
      - optimizer.step()
   c. model.eval() + torch.no_grad()
   d. Tính val_loss tương tự trên val_loader
   e. In "Epoch N | Train Loss | Val Loss"
   f. torch.save(checkpoint)
   g. json.dump(history)  ← sau mỗi epoch
```

### 8.2 Inference (greedy decode)

```
Model A/B:
  1. img_feat = i_encoder(img)              (1, 1024)
  2. q_feat   = q_encoder(q)               (1, 1024)
  3. fusion   = img_feat * q_feat          (1, 1024)
  4. h_0 = fusion.unsqueeze(0).repeat(L,1,1)
  5. token = <start>
  6. Loop:
     embed        = embedding(token)
     output, h    = lstm(embed, hidden)
     logit        = fc(output.squeeze(1))
     pred         = argmax(logit)
     if pred == <end>: break
     append pred → result
  7. return id2word(result)

Model C/D: (thêm attention mỗi bước)
  1. img_feat   = i_encoder(img)            (1, 49, 1024)
  2. q_feat     = q_encoder(q)             (1, 1024)
  3. img_mean   = img_feat.mean(dim=1)     (1, 1024)
  4. fusion     = img_mean * q_feat
  5. h_0 = fusion.unsqueeze(0).repeat(L,1,1)
  6. Loop:
     logit, hidden, alpha = decoder.decode_step(token, hidden, img_feat)
     pred = argmax(logit)
     if pred == <end>: break
     append pred → result
  7. return id2word(result)
```

---

## 9. Cấu hình hyperparameters

| Hyperparameter | Giá trị | Mô tả |
|----------------|---------|-------|
| `embed_size` | 512 | Embedding dimension (câu hỏi + câu trả lời) |
| `hidden_size` | 1024 | LSTM hidden dimension, cũng là output size của CNN encoder |
| `num_layers` | 2 | Số lớp LSTM trong encoder và decoder |
| `attn_dim` | 512 | Attention projection dimension (C/D only) |
| `BATCH_SIZE` | 32 | Số samples mỗi batch |
| `EPOCHS` | 10 | Số epoch training |
| `LEARNING_RATE` | 1e-3 | Adam learning rate |
| `max_norm` | 5.0 | Gradient clipping threshold |
| `SPLIT_SEED` | 42 | Random seed cho train/val split |
| `VAL_RATIO` | 0.1 | Tỉ lệ validation set |
| `freeze_cnn` | `True` | Có freeze ResNet101 weights không (B/D) |
| `max_len` | 20 | Số token tối đa khi inference |

---

## 10. Hướng dẫn sử dụng

### 10.1 Train một model

```bash
# 1. Sửa MODEL_TYPE trong src/train.py
# MODEL_TYPE = 'A'   hoặc 'B', 'C', 'D'

# 2. Chạy training
python src/train.py

# Output:
# checkpoints/model_a_epoch1.pth ... model_a_epoch10.pth
# checkpoints/history_model_a.json
```

### 10.2 Đánh giá một model

```bash
python src/evaluate.py --model_type A
# hoặc chỉ định checkpoint cụ thể:
python src/evaluate.py --model_type C --checkpoint checkpoints/model_c_epoch5.pth
# chỉ chạy N samples:
python src/evaluate.py --model_type B --num_samples 100
```

### 10.3 So sánh tất cả model

```bash
python src/compare.py
# Kết quả: bảng Exact Match + BLEU-1 cho từng model
```

### 10.4 Vẽ training curves

```bash
python src/plot_curves.py
# Lưu vào: checkpoints/training_curves.png
```

### 10.5 Visualize attention (C/D)

```bash
python src/visualize.py --model_type C
python src/visualize.py --model_type D --sample_idx 5
# Lưu vào: checkpoints/attn_model_c.png
```

### 10.6 Test inference thủ công

```bash
# Sửa MODEL_TYPE trong src/inference.py rồi chạy:
python src/inference.py
# Output: Question / Predicted answer
```

### 10.7 Workflow Kaggle (full)

```bash
# Terminal Kaggle:
git clone https://github.com/Anakonkai01/new_vqa && cd new_vqa
pip install torch torchvision nltk matplotlib pillow tqdm
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
python create_dummy_data.py          # hoặc dùng real data
python src/scripts/1_build_vocab.py

# Train 4 model (đổi MODEL_TYPE rồi chạy từng cái):
# MODEL_TYPE = 'A' → python src/train.py
# MODEL_TYPE = 'B' → python src/train.py
# MODEL_TYPE = 'C' → python src/train.py
# MODEL_TYPE = 'D' → python src/train.py

# Download thư mục checkpoints/ về local, rồi:
python src/compare.py
python src/plot_curves.py
python src/visualize.py --model_type C
python src/visualize.py --model_type D
```

---

## 11. Kết quả đánh giá (template)

Điền sau khi train xong trên Kaggle:

| Model | Exact Match | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | Train Time |
|-------|:-----------:|:------:|:------:|:------:|:------:|:------:|:----------:|
| A — Scratch, No Attn | — | — | — | — | — | — | — |
| B — Pretrained, No Attn | — | — | — | — | — | — | — |
| C — Scratch, Attn | — | — | — | — | — | — | — |
| D — Pretrained, Attn | — | — | — | — | — | — | — |

**Kết quả dự kiến:** D > B > C > A (pretrained features > scratch; attention > no attention)

---

## 12. Ghi chú kỹ thuật

### Những lỗi đã gặp và cách fix

| Lỗi | Nguyên nhân | Fix |
|-----|-------------|-----|
| `size mismatch` khi load checkpoint | Vocab size thay đổi giữa 2 lần train | Retrain từ đầu sau khi rebuild vocab |
| `CUDA error: no kernel image` | MX330 sm_61 < sm_70 yêu cầu | `DEVICE = torch.device('cpu')` |
| `ValueError: ResNet101_Weights("Default")` | Không thể khởi tạo enum bằng string | Dùng `ResNet101_Weights.DEFAULT` |
| `RuntimeError: ...contiguous` | Tensor slice cần `.contiguous()` trước `.view()` | `decoder_target.contiguous().view(-1)` |

### Quy ước đặt tên

- Class: `VQAmodelA` (chữ **m thường**) — tên cũ, **không đổi** để tránh break code
- File: `encoder_question.py` (không có 's') — tên file thực tế
- Checkpoint: `model_{a/b/c/d}_epoch{n}.pth`
- History: `history_model_{a/b/c/d}.json`

### Normalization

- Image features: `F.normalize(feat, p=2, dim=1)` (flat) hoặc `dim=-1` (spatial)
- Lý do: loại bỏ ảnh hưởng của magnitude, chỉ giữ hướng vector → fusion ổn định hơn

### `matplotlib.use('Agg')`

Đặt ở đầu `plot_curves.py` và `visualize.py` → không cần X11/display server, chạy được trên Kaggle và headless server.

### History JSON — tại sao ghi sau mỗi epoch

Training trên Kaggle có thể bị ngắt bất cứ lúc nào (timeout, quota). Ghi JSON sau mỗi epoch đảm bảo dữ liệu không mất dù training không hoàn thành đủ 10 epoch.
