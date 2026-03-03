# CONTEXT HANDOFF — VQA Project (tiếp tục từ đây)

> Tài liệu này tóm tắt toàn bộ ngữ cảnh cần thiết để tiếp tục dự án.
> Cập nhật lần cuối: 2026-03-03 (session 4 — 11 architecture improvements implemented)

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

### ĐÃ HOÀN THÀNH (session 1-4)

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

### CÒN LẠI — VIỆC CẦN LÀM TIẾP
1. **Train model A 1 epoch** để kiểm tra pipeline: `python src/train.py --model A --epochs 1`
2. **Train đủ 4 model** theo training plan (Phase 1-3) và compare
3. **Download val2014 images** nếu chưa có (cần cho evaluation)

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

## 5. VQA-E JSON FORMAT (dự kiến)

```json
{
  "annotations": [
    {
      "image_id": 458752,
      "question_id": 458752000,
      "question": "What is the person doing?",
      "answer": "surfing",
      "explanation": ["the person is riding a wave on a surfboard"]
    }
  ]
}
```

**Code đã xử lý cả 2 trường hợp:**
- `ann['answer']` (VQA-E chuẩn)
- `ann['multiple_choice_answer']` (fallback nếu format khác)

**Sau khi download, verify bằng:**
```python
import json
with open('data/raw/vqa_e_json/vqa_e_train2014.json') as f:
    data = json.load(f)
print(data['annotations'][0].keys())  # kiểm tra field names
print(data['annotations'][0])          # xem 1 sample
```

Nếu field name khác với dự kiến, sửa trong:
- `src/scripts/1_build_vocab.py` (line ~36-43)
- `src/dataset.py` class `VQAEDataset.__getitem__` (line ~98-102)

---

## 6. CẤU TRÚC THƯ MỤC

```
vqa_new/
├── README.md                    ← đề bài
├── CONTEXT_HANDOFF.md           ← file này
├── DOCUMENTATION.md             ← docs chi tiết
├── src/
│   ├── vocab.py                 ← Vocabulary class (word2idx, idx2word, numericalize)
│   ├── dataset.py               ← VQAEDataset (mới), VQADataset (cũ, giữ lại)
│   ├── train.py                 ← training loop (CE + Label Smoothing + Coverage + Grad Accum)
│   ├── evaluate.py              ← BLEU-4, METEOR, BERTScore, Exact Match
│   ├── compare.py               ← so sánh 4 model trong 1 bảng + BERTScore
│   ├── inference.py             ← greedy + beam search decode (n-gram blocking)
│   ├── visualize.py             ← attention heatmap visualization
│   ├── glove_utils.py           ← GloVe download + embedding matrix builder
│   ├── plot_curves.py           ← training curve plotting
│   ├── models/
│   │   ├── vqa_models.py        ← VQAModelA/B/C/D + GatedFusion
│   │   ├── encoder_cnn.py       ← SimpleCNN, ResNetEncoder, spatial variants
│   │   ├── encoder_question.py  ← QuestionEncoder (BiLSTM + GloVe)
│   │   ├── decoder_lstm.py      ← LSTMDecoder (Weight Tying + GloVe)
│   │   └── decoder_attention.py ← LSTMDecoderWithAttention (Dual Attn + Coverage + Weight Tying)
│   └── scripts/
│       ├── 1_build_vocab.py     ← build vocab từ VQA-E JSON
│       └── 2_extract_features.py← extract ResNet features (optional HDF5)
├── data/
│   ├── raw/
│   │   ├── images/
│   │   │   ├── train2014/       ← COCO train images
│   │   │   └── val2014/         ← COCO val images
│   │   └── vqa_e_json/          ← VQA-E JSON files
│   │       ├── VQA-E_train_set.json
│   │       └── VQA-E_val_set.json
│   └── processed/
│       ├── vocab_questions.json ← Q vocab (4546 từ)
│       └── vocab_answers.json   ← A vocab (8648 từ)
└── checkpoints/                 ← model_X_best.pth, model_X_resume.pth, milestones
```

---

## 7. COMMANDS ĐỂ CHẠY

```bash
# Bước 1: Train từng model (basic)
python src/train.py --model A --epochs 10 --batch_size 128
python src/train.py --model B --epochs 10 --batch_size 128
python src/train.py --model C --epochs 10 --batch_size 64
python src/train.py --model D --epochs 10 --batch_size 64

# Optional flags:
python src/train.py --model A --epochs 10 --glove                    # GloVe embeddings
python src/train.py --model A --epochs 10 --scheduled_sampling       # reduce exposure bias
python src/train.py --model D --epochs 10 --finetune_cnn             # unfreeze ResNet
python src/train.py --model C --epochs 10 --coverage                 # coverage mechanism (C/D only)
python src/train.py --model A --epochs 10 --accum_steps 4            # effective batch = 128×4 = 512
python src/train.py --model A --epochs 10 --augment                  # data augmentation
python src/train.py --model A --epochs 10 --early_stopping 5         # patience=5

# Bước 2: Evaluate 1 model
python src/evaluate.py --model_type A
python src/evaluate.py --model_type A --beam_width 5                 # beam search
python src/evaluate.py --model_type A --beam_width 5 --no_repeat_ngram 3  # + n-gram blocking

# Bước 3: So sánh tất cả model
python src/compare.py --epoch 10
python src/compare.py --epoch 10 --beam_width 3
```

---

## 8. CÁC THAM SỐ QUAN TRỌNG

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `embed_size` | 512 | word embedding dim (300 if GloVe, projected to 512) |
| `hidden_size` | 1024 | LSTM hidden dim |
| `num_layers` | 2 | LSTM layers |
| `attn_dim` | 512 | Bahdanau attention dim (model C/D) |
| `max_len` (inference) | 50 | max tokens sinh ra |
| `batch_size` | 128 (A/B), 64 (C/D) | |
| Answer vocab threshold | 3 | |
| Grad clip | 5.0 | clip_grad_norm |
| Label smoothing | 0.1 | CrossEntropyLoss |
| LR Warmup | 3 epochs | LinearLR(start_factor=0.1) |
| Coverage λ | 1.0 | weight for coverage loss (C/D) |
| N-gram blocking | 3 | trigram blocking in beam search |
| Accum steps | 1 (default) | gradient accumulation |

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
- Ampere+ GPU (A100): BFloat16 tự động, không cần GradScaler
- GPU khác: Float16 + GradScaler

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

## 13. BUG TIỀM NĂNG CẦN CHÚ Ý SAU KHI CÓ DATA

1. **Field name mismatch**: VQA-E có thể dùng `multiple_choice_answer` thay vì `answer` — code đã handle fallback nhưng cần verify
2. **Explanation là list hay string?** Code xử lý: `exp_list[0] if exp_list else ''` — nếu explanation là string (không phải list) thì sẽ lấy ký tự đầu tiên → cần kiểm tra
3. **Vocab size nhỏ**: Nếu VQA-E train set nhỏ hơn dự kiến, answer vocab có thể rất nhỏ → điều chỉnh threshold xuống 2
