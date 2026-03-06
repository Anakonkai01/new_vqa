# CONTEXT HANDOFF — VQA Project (tiếp tục từ đây)

> Tài liệu này tóm tắt toàn bộ ngữ cảnh cần thiết để tiếp tục dự án.
> Cập nhật lần cuối: 2026-03-03 (session 5 — bugs fixed, vocab rebuilt, local training notebook)

---

## 1. ĐỀ BÀI (README.md)

```
Bài 2 (7đ):
Sử dụng model LSTM và model CNN xây dựng kiến trúc cho bài toán Visual Question Answering.

Input : ảnh và câu hỏi
Output: câu trả lời — PHẢI được sinh ra bởi LSTM-decoder (generative, không phải classification)

Xây dựng các loại kiến trúc khác nhau dựa trên:
  1) Không có và có dùng cơ chế Attention
  2) Train từ đầu và có dùng Pretrained model

Đánh giá các mô hình qua các độ đo và so sánh.
```

---

## 2. VẤN ĐỀ ĐÃ PHÁT HIỆN VÀ QUYẾT ĐỊNH ĐÃ ĐƯARA

### Vấn đề gốc rễ
- Codebase ban đầu dùng VQA 2.0 với `multiple_choice_answer` → câu trả lời 1-3 từ ("yes", "no", "2")
- LSTM decoder không có ý nghĩa với output quá ngắn
- Metric accuracy/exact match không phù hợp với generative output

### Quyết định
**Chuyển sang VQA-E dataset** (Li et al., 2018 — "VQA-E: Explaining, Elaborating, and Enhancing Your Answers"):
- Target output: `"yes because the man has glasses on his face"` (~10-25 tokens)
- Cùng COCO 2014 images → không cần download ảnh lại
- Primary metrics: **BLEU-4** và **METEOR** (thay vì accuracy)

---

## 3. TRẠNG THÁI HIỆN TẠI

### ĐÃ HOÀN THÀNH (session 1-5)

#### Session 1-2: VQA-E Migration + Pipeline
- [x] VQA-E dataset migration: `VQAEDataset`, `1_build_vocab.py`, all paths updated
- [x] Metrics: BLEU-4 + METEOR (primary), Exact Match (reference)
- [x] Download VQA-E JSON + Verify format + Rebuild vocab (Q=4546, A=8648)

#### Session 3: Architecture Improvements Round 1 (4 cải tiến)
- [x] **Label Smoothing** — `CrossEntropyLoss(label_smoothing=0.1)` trong `train.py`
- [x] **BiLSTM Question Encoder** — `encoder_question.py`: `bidirectional=True`, hidden_size//2 per direction
- [x] **Gated Fusion** — `vqa_models.py`: `GatedFusion` class thay Hadamard product
- [x] **Dual Attention** — `decoder_attention.py`: `img_attention` + `q_attention` trên cả image và question

#### Session 3: Architecture Improvements Round 2 (4 cải tiến)
- [x] **LR Warmup** — `train.py`: `LinearLR(start_factor=0.1, total_iters=3)` + `ReduceLROnPlateau`
- [x] **GloVe Embeddings** — `glove_utils.py` (new), encoder/decoder: `--glove` flag, 300d→512 projection
- [x] **Weight Tying** — `decoder_lstm.py`, `decoder_attention.py`: `out_proj` + `fc.weight = embedding.weight`
- [x] **BERTScore** — `evaluate.py`, `compare.py`: optional `bert-score` metric with graceful fallback

#### Session 4: Architecture Improvements Round 3 (3 cải tiến)
- [x] **Coverage Mechanism** — `decoder_attention.py`: cumulative attention vector + coverage loss. Giảm lặp từ khi sinh explanation dài. Flag: `--coverage --coverage_lambda 1.0` (model C/D only)
- [x] **Repetition Penalty (N-gram Blocking)** — `inference.py`: `_block_repeated_ngrams()` trong beam search. Default block trigram. Flag: `--no_repeat_ngram 3` (0=disable)
- [x] **Gradient Accumulation** — `train.py`: `--accum_steps N` → effective batch = batch_size × N

#### Session 5: Bug Fixes + Local Training Setup
- [x] **Path bugs fixed** — 6 files dùng sai đường dẫn, đã sửa tất cả:
  - `data/raw/images/train2014` → `data/raw/train2014`
  - `data/raw/images/val2014` → `data/raw/val2014`
  - `data/raw/vqa_e_json/VQA-E_*.json` → `data/vqa_e/VQA-E_*.json`
  - Files: `train.py`, `evaluate.py`, `compare.py`, `inference.py`, `visualize.py`, `1_build_vocab.py`
- [x] **Vocab rebuilt** — vocab cũ bị overwrite bởi VQA 2.0 data (short answers, thiếu explanation words). Đã rebuild từ VQA-E: Q=4546, A=8648, coverage tăng từ 93% → 99.6%
- [x] **AMP API updated** — `torch.cuda.amp` → `torch.amp` (fix FutureWarning trong PyTorch 2.10)
- [x] **Local training notebook** — `vqa_local_training.ipynb` tạo mới cho RTX 3060 12GB

### CÒN LẠI — VIỆC CẦN LÀM TIẾP
1. **Train đủ 4 model** theo 3-phase plan dùng `vqa_local_training.ipynb`
2. **Compare metrics** sau mỗi phase (epoch 10, 15, 20)

---

## 4. KIẾN TRÚC CỦA 4 MODEL

| Model | CNN | Attention | Pretrained |
|---|---|---|---|
| **A** | SimpleCNN (scratch) | Không | Không |
| **B** | ResNet101 | Không | Có (ImageNet) |
| **C** | SimpleCNN Spatial (scratch) | Bahdanau | Không |
| **D** | ResNet101 Spatial | Bahdanau | Có (ImageNet) |

**Pipeline chung:**
```
Image → CNN Encoder → img_feature
Question → BiLSTM Encoder → (q_feature, q_hidden_states)
Gated Fusion = σ(W_g·[img;q]) * tanh(W_img(img)) + (1-σ) * tanh(W_q(q))
Fusion → h_0, c_0 (LSTM decoder initial state)
LSTM Decoder → sinh sequence token by token (teacher forcing khi train)
```

**Model C/D** thêm Dual Attention: decoder attend over 49 spatial image regions + question hidden states tại mỗi decode step. Optional Coverage Mechanism giảm lặp từ.

---

## 5. VQA-E JSON FORMAT (ĐÃ VERIFY — session 5)

**Root là list**, không phải dict có key `annotations`:

```json
[
  {
    "answer_type": "other",
    "explanation": ["Closeup of bins of food that include broccoli and bread.", 0.679],
    "question": "What is the green stuff?",
    "multiple_choice_answer": "broccoli",
    "answers": ["broccoli", "broccoli", ...],
    "question_type": "what is the",
    "img_id": 9
  }
]
```

**Field names thực tế (đã xác nhận):**
- `img_id` (không phải `image_id`)
- `multiple_choice_answer` (không phải `answer`)
- `explanation` = list gồm `[text_string, confidence_float]` — lấy `[0]`

**Code trong `dataset.py` đã đúng:**
```python
img_id     = ann['img_id']
answer     = ann.get('multiple_choice_answer', '')
exp_list   = ann.get('explanation', [])
explanation = exp_list[0] if exp_list and isinstance(exp_list[0], str) else ''
```

**Thống kê dataset:**
- Train: 181,298 samples | Val: 88,488 samples
- Train images: 82,783 | Val images: 40,504 (tại `data/raw/train2014/` và `data/raw/val2014/`)

---

## 6. CẤU TRÚC THƯ MỤC (THỰC TẾ — đã verify session 5)

```
new_vqa/
├── README.md
├── CONTEXT_HANDOFF.md
├── DOCUMENTATION.md
├── vqa_local_training.ipynb     ← notebook train local (RTX 3060) [tạo session 5]
├── vqa_colab_new.ipynb          ← notebook Colab (giữ nguyên)
├── src/
│   ├── vocab.py
│   ├── dataset.py               ← VQAEDataset + VQADataset (cũ, giữ lại)
│   ├── train.py                 ← training loop
│   ├── evaluate.py              ← BLEU-4, METEOR, BERTScore, Exact Match
│   ├── compare.py               ← so sánh 4 model
│   ├── inference.py             ← greedy + beam search (n-gram blocking)
│   ├── visualize.py             ← attention heatmap
│   ├── glove_utils.py           ← GloVe loader
│   ├── plot_curves.py           ← training curves
│   ├── models/
│   │   ├── vqa_models.py        ← VQAModelA/B/C/D + GatedFusion
│   │   ├── encoder_cnn.py       ← SimpleCNN, ResNetEncoder, spatial variants
│   │   ├── encoder_question.py  ← QuestionEncoder (BiLSTM)
│   │   ├── decoder_lstm.py      ← LSTMDecoder (Weight Tying)
│   │   └── decoder_attention.py ← LSTMDecoderWithAttention (Dual Attn + Coverage)
│   └── scripts/
│       ├── 1_build_vocab.py     ← build vocab từ VQA-E JSON
│       └── 2_extract_features.py
├── data/
│   ├── raw/
│   │   ├── train2014/           ← COCO train images (82,783 ảnh)  ← ĐÚNG PATH
│   │   ├── val2014/             ← COCO val images (40,504 ảnh)    ← ĐÚNG PATH
│   │   └── test2015/
│   ├── vqa_e/                   ← VQA-E JSON                       ← ĐÚNG PATH
│   │   ├── VQA-E_train_set.json (181,298 samples)
│   │   └── VQA-E_val_set.json  (88,488 samples)
│   ├── vqa_data_json/           ← VQA 2.0 gốc (giữ tham khảo, không dùng để train)
│   └── processed/
│       ├── vocab_questions.json ← Q vocab 4,546 từ (rebuild từ VQA-E session 5)
│       └── vocab_answers.json   ← A vocab 8,648 từ (rebuild từ VQA-E session 5)
└── checkpoints/                 ← tạo tự động khi train
```

---

## 7. COMMANDS ĐỂ CHẠY

**Chạy từ thư mục `new_vqa/` (project root). Dùng notebook `vqa_local_training.ipynb` để tiện hơn.**

```bash
# ── Phase 1 (10 epochs, LR=1e-3, ResNet frozen) ──────────────────────────────
# RTX 3060 12GB — batch sizes đã chọn để không OOM
python src/train.py --model A --epochs 10 --lr 1e-3 --batch_size 64  --accum_steps 2 --num_workers 6 --augment --early_stopping 5
python src/train.py --model B --epochs 10 --lr 1e-3 --batch_size 32  --accum_steps 4 --num_workers 6 --augment --early_stopping 5
python src/train.py --model C --epochs 10 --lr 1e-3 --batch_size 32  --accum_steps 2 --num_workers 6 --augment --early_stopping 5 --coverage
python src/train.py --model D --epochs 10 --lr 1e-3 --batch_size 16  --accum_steps 4 --num_workers 6 --augment --early_stopping 5 --coverage

# ── Phase 2 (5 epochs, LR=5e-4, unfreeze ResNet layer3+4 cho B/D) ────────────
python src/train.py --model A --epochs 5 --lr 5e-4 --resume checkpoints/model_a_resume.pth --batch_size 64  --accum_steps 2 --num_workers 6 --augment
python src/train.py --model B --epochs 5 --lr 5e-4 --resume checkpoints/model_b_resume.pth --batch_size 32  --accum_steps 4 --num_workers 6 --finetune_cnn
python src/train.py --model C --epochs 5 --lr 5e-4 --resume checkpoints/model_c_resume.pth --batch_size 32  --accum_steps 2 --num_workers 6 --coverage
python src/train.py --model D --epochs 5 --lr 5e-4 --resume checkpoints/model_d_resume.pth --batch_size 16  --accum_steps 4 --num_workers 6 --coverage --finetune_cnn

# ── Phase 3 (5 epochs, LR=2e-4, Scheduled Sampling) ─────────────────────────
python src/train.py --model A --epochs 5 --lr 2e-4 --resume checkpoints/model_a_resume.pth --batch_size 64  --accum_steps 2 --num_workers 6 --scheduled_sampling
python src/train.py --model B --epochs 5 --lr 2e-4 --resume checkpoints/model_b_resume.pth --batch_size 32  --accum_steps 4 --num_workers 6 --scheduled_sampling
python src/train.py --model C --epochs 5 --lr 2e-4 --resume checkpoints/model_c_resume.pth --batch_size 32  --accum_steps 2 --num_workers 6 --scheduled_sampling --coverage
python src/train.py --model D --epochs 5 --lr 2e-4 --resume checkpoints/model_d_resume.pth --batch_size 16  --accum_steps 4 --num_workers 6 --scheduled_sampling --coverage

# ── Evaluate + Compare ────────────────────────────────────────────────────────
python src/evaluate.py --model_type A --checkpoint checkpoints/model_a_best.pth
python src/evaluate.py --model_type A --checkpoint checkpoints/model_a_best.pth --beam_width 5 --no_repeat_ngram 3

python src/compare.py --epoch 10   # sau Phase 1
python src/compare.py --epoch 15   # sau Phase 2
python src/compare.py --epoch 20   # sau Phase 3 (final)
python src/compare.py --epoch 20 --beam_width 3

# ── Plots & Visualization ─────────────────────────────────────────────────────
python src/plot_curves.py
python src/visualize.py --model_type D --checkpoint checkpoints/model_d_best.pth --image_path "..." --question "..."
```

---

## 8. CÁC THAM SỐ QUAN TRỌNG

**Model architecture:**

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `embed_size` | 512 | word embedding dim (300 if GloVe, projected to 512) |
| `hidden_size` | 1024 | LSTM hidden dim |
| `num_layers` | 2 | LSTM layers |
| `attn_dim` | 512 | Bahdanau attention dim (model C/D) |
| `max_len` (inference) | 50 | max tokens sinh ra |
| `vocab_q_size` | **4,546** | Q vocab từ VQA-E (rebuild session 5) |
| `vocab_a_size` | **8,648** | A vocab từ VQA-E MC+explanation (rebuild session 5) |

**Training (RTX 3060 12GB — đã verify không OOM):**

| Model | batch_size | accum_steps | Effective batch | num_workers |
|---|---|---|---|---|
| A (SimpleCNN) | 64 | 2 | 128 | 6 |
| B (ResNet101) | 32 | 4 | 128 | 6 |
| C (SimpleCNN Spatial) | 32 | 2 | 64 | 6 |
| D (ResNet101 Spatial) | 16 | 4 | 64 | 6 |

**Regularization & training config:**

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| Answer vocab threshold | 3 | dùng khi build vocab |
| Grad clip | 5.0 | clip_grad_norm |
| Label smoothing | 0.1 | CrossEntropyLoss |
| Weight decay | 1e-5 | L2 regularization |
| LR Warmup | 3 epochs | LinearLR(start_factor=0.1) |
| ReduceLROnPlateau | factor=0.5, patience=2 | sau warmup |
| Coverage λ | 1.0 | weight for coverage loss (C/D only) |
| N-gram blocking | 3 | trigram blocking in beam search |
| Early stopping | patience=5 | recommended local |

---

## 9. METRICS — Ý NGHĨA VÀ KỲ VỌNG

| Metric | Ý nghĩa | Kỳ vọng |
|---|---|---|
| **BLEU-4** ★ | 4-gram overlap giữa prediction và reference | 0.05-0.20 là hợp lý |
| **METEOR** ★ | n-gram + synonym matching (dùng WordNet) | 0.10-0.30 là hợp lý |
| **BERTScore** ★ | Semantic similarity via BERT embeddings | 0.40-0.70 |
| BLEU-1/2/3 | Unigram/bigram/trigram precision | Tham khảo |
| Exact Match | prediction == ground truth (rất thấp) | <5%, chỉ tham khảo |

**VQA Accuracy đã bị loại bỏ** — metric này dành cho classification, không phù hợp với generative output.

---

## 10. LƯU Ý KỸ THUẬT QUAN TRỌNG

### Teacher Forcing
Train dùng teacher forcing: decoder nhận ground truth token làm input tại mỗi step.
`decoder_input = answer[:, :-1]` → `[<start>, w1, w2, ...]`
`decoder_target = answer[:, 1:]` → `[w1, w2, ..., <end>]`

### Scheduled Sampling (optional)
Dần dần thay GT token bằng prediction của model để giảm exposure bias.
`epsilon(epoch) = k / (k + exp(epoch/k))`, k=5 mặc định.

### Coverage Mechanism (Model C/D only)
- Tích lũy attention weights qua các step: `coverage_t = Σ α_0..α_{t-1}`
- Coverage signal đưa vào Bahdanau Attention: `W_cov @ coverage.unsqueeze(-1)`
- Coverage loss = `Σ min(α_t, coverage_t) / max_len` — phạt khi attend lại vùng đã attend
- Tổng loss = `CE_loss + λ * coverage_loss`, λ mặc định = 1.0
- Chỉ áp dụng cho image attention (không phải question attention)
- `decode_step()` giờ trả về 4 giá trị: `(logit, hidden, img_alpha, coverage)`

### Gradient Accumulation
- Chia loss cho `accum_steps` trước backward: `loss = loss / accum_steps`
- Gọi `optimizer.step()` + `optimizer.zero_grad()` mỗi N mini-batch
- Effective batch size = `batch_size × accum_steps`
- Ví dụ: `--batch_size 16 --accum_steps 4` → effective batch = 64

### N-gram Blocking (Inference)
- `_block_repeated_ngrams()` đặt `log_prob[token] = -inf` cho token tạo trigram lặp
- Mặc định block trigram (n=3), tunable via `--no_repeat_ngram`
- Chỉ áp dụng khi beam search, không ảnh hưởng greedy decode

### Checkpoint Format
- `model_X_resume.pth` — full checkpoint (optimizer, scheduler, scaler, history) — dùng để resume
- `model_X_best.pth` — chỉ model weights — best val loss
- `model_X_epoch10.pth` — milestone epochs (10, 15, 20)

### Mixed Precision
- RTX 3060 (GA106, Ampere, Compute Cap 8.6): **BFloat16 tự động**, không cần GradScaler ✓
- GPU khác (Turing trở xuống): Float16 + GradScaler
- Code tự detect: `_supports_bf16()` kiểm tra `compute_capability >= 8.0`
- API đã update sang `torch.amp.GradScaler('cuda', ...)` và `torch.amp.autocast('cuda', ...)` (fix session 5)

---

## 11. TOÀN BỘ CẢI TIẾN ĐÃ TRIỂN KHAI (11 improvements)

| # | Cải tiến | File chính | Flag / Config |
|---|----------|-----------|---------------|
| 1 | Label Smoothing | `train.py` | `--label_smoothing 0.1` |
| 2 | BiLSTM Question Encoder | `encoder_question.py` | mặc định bật |
| 3 | Gated Fusion | `vqa_models.py` | mặc định bật (Model C/D) |
| 4 | Dual Attention (img + question) | `decoder_attention.py` | mặc định bật |
| 5 | LR Warmup | `train.py` | `--warmup_epochs 3` |
| 6 | GloVe Embeddings | `glove_utils.py`, `train.py` | `--glove --glove_dim 300` |
| 7 | Weight Tying | `decoder_attention.py`, `decoder_lstm.py` | mặc định bật |
| 8 | BERTScore | `evaluate.py` | tự động tính |
| 9 | Coverage Mechanism | `decoder_attention.py`, `vqa_models.py`, `train.py` | `--coverage --coverage_lambda 1.0` |
| 10 | N-gram Blocking | `inference.py`, `evaluate.py` | `--no_repeat_ngram 3` |
| 11 | Gradient Accumulation | `train.py` | `--accum_steps 4` |

---

## 12. NHỮNG GÌ KHÔNG CẦN THAY ĐỔI

- `src/vocab.py` — Vocabulary class không đổi
- `vqa_collate_fn` trong `dataset.py` — padding logic không đổi
- Loss function (CrossEntropyLoss, ignore_index=0)
- `src/VQADataset` cũ — giữ lại (không xóa) phòng trường hợp cần so sánh

---

## 13. BUGS ĐÃ GẶP VÀ ĐÃ FIX (session 5)

Tất cả bugs đã được xác nhận và fix. Không còn gì cần làm.

| Bug | Trạng thái | Chi tiết |
|-----|-----------|---------|
| **Path sai** (6 files) | ✅ Fixed | `data/raw/images/` → `data/raw/` và `data/raw/vqa_e_json/` → `data/vqa_e/` |
| **Vocab sai nguồn** | ✅ Fixed | Vocab cũ build từ VQA 2.0 (short answers), đã rebuild từ VQA-E |
| **AMP deprecated API** | ✅ Fixed | `torch.cuda.amp` → `torch.amp` (PyTorch 2.10) |
| **Field name `answer`** | ✅ Verified | Actual field = `multiple_choice_answer`, code đã dùng đúng |
| **Explanation là list** | ✅ Verified | Format = `[text_string, confidence_float]`, `[0]` lấy đúng text |
| **Vocab size nhỏ** | ✅ Verified & Fixed | Root cause: VQA 2.0 thay vì VQA-E. Sau rebuild: A=8,648, coverage=99.6% |

## 14. MÁY LOCAL — THÔNG SỐ KỸ THUẬT

- **CPU**: Intel i7-14700K (20 cores: 8P + 12E)
- **RAM**: 64 GB
- **GPU**: NVIDIA GeForce RTX 3060 12 GB VRAM
- **CUDA**: 12.8 | **PyTorch**: 2.10.0
- **Compute Capability**: 8.6 (Ampere) → BF16 supported, TF32 enabled
- **Python**: 3.12 | venv: `/home/mayxin/workspace/DeepLearning/.venv`
- **Project root**: `/home/mayxin/workspace/DeepLearning/new_vqa`
- **Notebook**: `vqa_local_training.ipynb` — chạy từ project root
