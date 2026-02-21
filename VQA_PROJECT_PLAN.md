# VQA Project Plan — LSTM-Decoder Architecture

## Mục tiêu

Xây dựng hệ thống Visual Question Answering sử dụng CNN (Image Encoder) + LSTM (Question Encoder) + LSTM-Decoder (Answer Generator).

**Input:** Ảnh + Câu hỏi  
**Output:** Câu trả lời được **sinh ra** bởi LSTM-Decoder (generative)

---

## 4 Model cần xây dựng

| Model | CNN Encoder        | Attention | Decoder       |
|-------|--------------------|-----------|---------------|
| **A** | Train from scratch | Không     | LSTM-Decoder  |
| **B** | Pretrained ResNet  | Không     | LSTM-Decoder  |
| **C** | Train from scratch | Có        | LSTM-Decoder  |
| **D** | Pretrained ResNet  | Có        | LSTM-Decoder  |

---

## Kiến trúc tổng quát

```
┌──────────────────────────────────────────────────────────────┐
│                    KIẾN TRÚC TỔNG QUÁT                       │
│                                                              │
│  ┌─────────┐     ┌──────────────┐     ┌──────────────────┐   │
│  │  IMAGE   │     │   QUESTION   │     │   LSTM-DECODER   │   │
│  │ ENCODER  │     │   ENCODER    │     │   (sinh answer)  │   │
│  │ (CNN)    │     │   (LSTM)     │     │                  │   │
│  └────┬─────┘     └──────┬───────┘     │  Input: fusion   │   │
│       │                  │             │  Output: tokens   │   │
│       └───────┬──────────┘             │  one by one       │   │
│               │                        │                  │   │
│          ┌────▼────┐                   │  <start> → "yes" │   │
│          │ FUSION  │──────────────────▶│  "yes"  → <end>  │   │
│          └─────────┘                   └──────────────────┘   │
│                                                              │
│  ⚡ Attention: CNN giữ spatial (196, 2048)                   │
│     Decoder attend vào từng vùng ảnh mỗi bước sinh          │
│                                                              │
│  ❌ No Attention: CNN mean pool → (2048)                     │
│     Fusion = concat/hadamard → init decoder                  │
└──────────────────────────────────────────────────────────────┘
```

### Model A & B — No Attention

```
CNN → (batch, 2048) → Linear → (batch, hidden)
LSTM-Encoder(question) → (batch, hidden)
Fusion = img * q → (batch, hidden)
LSTM-Decoder:
  - Initial hidden state = fusion
  - Mỗi bước: input = previous_word_embedding
  - Output = next token
  - Dừng khi sinh <end> hoặc max_len
```

### Model C & D — With Attention

```
CNN → (batch, 196, 2048) → Linear → (batch, 196, hidden)   # giữ spatial
LSTM-Encoder(question) → (batch, hidden)
LSTM-Decoder mỗi bước:
  1. Tính attention weights: α = softmax(decoder_hidden @ image_regions)
  2. Context vector: c = Σ(α * image_regions)
  3. Decoder input = [context; previous_word_embedding; question_feature]
  4. Sinh token tiếp theo
```

### Scratch vs Pretrained

- **Scratch:** CNN random init, `requires_grad = True`, train cùng toàn bộ model
- **Pretrained:** ResNet101 pretrained ImageNet, freeze hoặc fine-tune last layers

---

## Vấn đề với code hiện tại

Model hiện tại dùng **classifier** (discriminative — chọn 1 trong N đáp án), nhưng đề bài yêu cầu **LSTM-Decoder** (generative — sinh answer token-by-token). Cần **viết lại**:

1. **Extract features:** cần thêm bản giữ spatial `(N, 196, 2048)` cho Attention models
2. **Dataset:** answer phải trả về dạng **sequence tokens** `[<start>, token1, token2, ..., <end>]` thay vì 1 class index
3. **Model:** thay classifier bằng LSTM-Decoder
4. **Train:** thêm teacher forcing
5. **Evaluate:** thêm các metrics cho generative task

---

## Cấu trúc thư mục đề xuất

```
src/
├── scripts/
│   ├── 1_build_vocab.py                # Đã có — OK
│   ├── 2_extract_features.py           # SỬA: thêm mode spatial (196, 2048)
│   └── 3_preprocess_answers.py         # MỚI: answer → sequence tokens
├── models/
│   ├── encoder_cnn.py                  # MỚI: CNN scratch + pretrained
│   ├── encoder_question.py             # MỚI: LSTM encoder cho question
│   ├── decoder_lstm.py                 # MỚI: LSTM decoder (no attention)
│   ├── decoder_attention.py            # MỚI: LSTM decoder (with attention)
│   └── vqa_model.py                    # MỚI: Wrapper gộp encoder + decoder
├── dataset.py                          # SỬA: answer dạng sequence
├── vocab.py                            # Đã có — OK
├── train.py                            # SỬA: teacher forcing, 4 model configs
├── evaluate.py                         # MỚI: tính BLEU, CIDEr, VQA Accuracy
├── inference.py                        # MỚI: sinh câu trả lời từ ảnh + câu hỏi
└── compare.py                          # MỚI: so sánh 4 model, visualization
```

---

## Các bước thực hiện

### Phase 1: Chuẩn bị dữ liệu

| Bước | Việc cần làm | File | Trạng thái |
|------|-------------|------|------------|
| 1 | Sửa extract features — thêm mode spatial `(N, 196, 2048)` | `scripts/2_extract_features.py` | ⏳ Chưa làm (chỉ cần cho Model C, D) |
| 2 | Viết lại dataset — answer dạng sequence, load raw image | `dataset.py` | ✅ Hoàn thành |
| 3 | Tạo dummy data để test pipeline | `create_dummy_data.py` | ✅ Hoàn thành |

### Phase 2: Xây dựng Models

| Bước | Việc cần làm | File | Trạng thái |
|------|-------------|------|------------|
| 4 | CNN Encoder scratch | `models/encoder_cnn.py` | ✅ Hoàn thành — output `(batch, 1024)` |
| 5 | Question Encoder | `models/encoder_questions.py` | ✅ Hoàn thành — output `(batch, 1024)` |
| 6 | LSTM Decoder (No Attention) | `models/decoder_lstm.py` | ✅ Hoàn thành — teacher forcing mode |
| 7 | VQA Wrapper Model A | `models/vqa_models.py` | ✅ Hoàn thành — `VQAModelA` |
| 8 | LSTM Decoder (Attention) | `models/decoder_attention.py` | ⏳ Chưa làm (Model C, D) |

### Phase 3: Training

| Bước | Việc cần làm | File | Trạng thái |
|------|-------------|------|------------|
| 9 | Training loop Model A | `train.py` | 🔧 Gần xong — còn 1 bug nhỏ (xem devlog) |
| 10 | Thêm validation loop | `train.py` | ⏳ Chưa làm |

### Phase 4: Evaluation & So sánh

| Bước | Việc cần làm | File | Trạng thái |
|------|-------------|------|------------|
| 11 | Code evaluation | `evaluate.py` | ⏳ Chưa làm |
| 12 | Code inference (greedy/beam search) | `inference.py` | ⏳ Chưa làm |
| 13 | So sánh 4 model | `compare.py` | ⏳ Chưa làm |

---

## Độ đo đánh giá

| Độ đo | Mô tả | Lý do sử dụng |
|-------|-------|----------------|
| **BLEU** (1,2,3,4) | N-gram precision giữa predicted vs ground truth | Metric chuẩn cho text generation |
| **METEOR** | Xét synonyms + stemming + alignment | Bổ sung cho BLEU, xét ngữ nghĩa tốt hơn |
| **CIDEr** | TF-IDF weighted n-gram consensus | Đặc trưng cho image-text tasks |
| **VQA Accuracy** | `min(count(predicted_ans) / 3, 1)` theo VQA Challenge | **Metric chính** của VQA benchmark |
| **ROUGE-L** | Longest Common Subsequence F1 | Đánh giá bổ sung cho sequence |

---

## Bảng so sánh (template)

| Model | BLEU-1 | BLEU-4 | METEOR | CIDEr | VQA Acc | Training Time |
|-------|--------|--------|--------|-------|---------|---------------|
| A (Scratch, No Attn) | — | — | — | — | — | — |
| B (Pretrained, No Attn) | — | — | — | — | — | — |
| C (Scratch, Attn) | — | — | — | — | — | — |
| D (Pretrained, Attn) | — | — | — | — | — | — |

### Dự kiến kết quả

- **B > A**: Pretrained features chất lượng cao hơn scratch
- **D > C**: Tương tự, pretrained + attention mạnh nhất
- **C > A, D > B**: Attention giúp focus vào vùng ảnh liên quan đến câu hỏi
- **D** là model tốt nhất tổng thể

---

## Phân tích bổ sung (để đạt full điểm)

1. **Qualitative Analysis**: Hiển thị ảnh + câu hỏi + predicted answer vs ground truth (đúng & sai)
2. **Attention Heatmap**: Visualize vùng ảnh mà model C/D tập trung khi trả lời
3. **Error Analysis**: Phân loại lỗi theo loại câu hỏi (yes/no, counting, color, ...)
4. **Ablation Study**: Ảnh hưởng của hyperparameters (embed_size, hidden_size, num_layers)
5. **Training Curves**: Plot loss & accuracy theo epoch cho cả 4 model trên cùng 1 biểu đồ

---

## Lưu ý kỹ thuật

- **GPU MX330 không tương thích PyTorch CUDA mới** → Train trên CPU hoặc Google Colab
- **Teacher Forcing Ratio**: Bắt đầu 1.0, giảm dần về 0.5 theo epoch
- **Beam Search**: Dùng beam_size = 3-5 khi inference để cải thiện chất lượng
- **Gradient Clipping**: `clip_grad_norm_(model.parameters(), max_norm=5.0)` tránh exploding gradients
- Vocab answer cần `<start>`, `<end>`, `<pad>`, `<unk>` tokens (đã có trong `Vocabulary`)
