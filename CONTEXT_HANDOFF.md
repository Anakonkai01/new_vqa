# CONTEXT HANDOFF — VQA Project (Master State)

> Tài liệu này tóm tắt toàn bộ ngữ cảnh cực kỳ chi tiết của dự án VQA (Visual Question Answering) cho phiên bản hoàn chỉnh nhất (Model E).
> Cập nhật lần cuối: 2026-03-14 (Hoàn thiện Model E, RL SCST, LLM Evaluation).

---

## 1. ĐỀ BÀI VÀ MỤC TIÊU CUỐI CÙNG

Dự án VQA yêu cầu xây dựng một mô hình có khả năng nhận input là (Ảnh, Câu hỏi) và sinh ra Câu trả lời mang tính giải thích (Explanation).
- **Dataset:** VQA-E (VQA-Explanation) kết hợp với COCO Captions (Multi-task learning).
- **Mục tiêu:** Sinh ra một câu trả lời hoàn chỉnh, mạch lạc và có giải thích.
- **Model Tối Thượng (Model E):** Sử dụng các kiến trúc tinh hoa nhất gồm CLIP ViT-B/32, BiLSTM, FiLM Fusion và LSTM Decoder. Sau đó Train qua 4 Phase, bao gồm cả Reinforcement Learning (SCST) và đánh giá bằng LLM-as-a-judge (Gemini).

---

## 2. TRẠNG THÁI HIỆN TẠI VÀ THÀNH QUẢ

Dự án đã giải quyết toàn bộ các vấn đề kỹ thuật khó nhất và đang ở Phase Train cuối cùng trên máy local RTX 5070 Ti.

### Hoàn thiện Data Pipeline (Multi-Task Learning)
- Tích hợp thành công **COCO Captions** vào chung với VQA-E để giúp decoder học được cấu trúc ngữ pháp ngôn ngữ tự nhiên tốt hơn.
- Cấu trúc `<task_vqa>` và `<task_cap>` được đưa vào đầu câu để model phân biệt tác vụ.
- Vocab được rebuild sạch sẽ bằng `nltk.word_tokenize`, cho size hoàn hảo (~14.4k tokens cho Answer/Caption).

### Kiến Trúc Model E (Tốt nhất)
Model E (`VQAModelE`) tụ hội mọi tinh hoa:
1. **Vision:** `CLIPViTEncoder` trích xuất đặc trưng hình ảnh mạnh mẽ từ pretrained CLIP ViT-B/32 của OpenAI (dùng HuggingFace `transformers`).
2. **Language:** BiLSTM Question Encoder.
3. **Fusion:** `FiLMFusion` (Feature-wise Linear Modulation) thay thế cho Gated Fusion cũ, giúp image features được điều chỉnh trực tiếp bởi question features.
4. **Decoder:** LSTM Decoder với weight tying và hidden state initialization bằng `projection` từ text và image features. Tối đa chiều dài sinh từ lên `max_len=100`.

### Thuật Toán Training Đột Phá
- **SequentialLR Hack (Bug Fix):** PyTorch có bug với `SequentialLR` khi resume. Đã implement kỹ thuật "fast-forward" (tua nhanh `scheduler.step()`) trong `train.py` để Resume hoàn hảo không bị nổ Loss.
- **RL SCST (Self-Critical Sequence Training):** Cài đặt thành công Phase 4 với REINFORCE thuật toán. Model tự sinh câu (Sampled vs Greedy) và dùng phần thưởng BLEU-4 để tự cập nhật gradient (`train_rl.py`).
- **Grad Accumulation & BF16:** Chạy tối đa công suất RTX 5070 Ti (Batch size 256, Workers 12-16, Mixed Precision BFloat16).

---

## 3. CÁC PHASE TRAINING

Gồm 4 phase đào tạo chuyên sâu (File chạy chính: `train_model_e.ipynb`):
1. **Phase 1: Base Train (15 epochs)**
   - Đóng băng (Freeze) CLIP ViT. Chỉ train Decoder + Fusion + Q-Encoder.
   - LR = `1e-3`, Warmup 3 epochs, Batch size 256.
2. **Phase 2: Fine-Tuning Backbones (10 epochs)**
   - Rã đông dãy layer cuối của CLIP ViT để fine-tune.
   - LR = `1e-4` cho mạng chính, `1e-5` cho CLIP backbone.
3. **Phase 3: Scheduled Sampling (5 epochs)**
   - Giảm dần sự phụ thuộc vào Ground Truth (Teacher Forcing). Epsilon phân rã theo hàm lũy thừa để model học cách tự sửa lỗi sinh từ dự đoán trước đó của chính nó.
4. **Phase 4: RL SCST (3 epochs)**
   - Tối ưu hóa trực tiếp trên Non-differentiable metric (BLEU-4).
   - Sử dụng REINFORCE loss: $L(\theta) = - (r(w^s) - r(\hat{w})) \sum \log p(w^s_t | h_t)$.
   - Script: `src/train_rl.py`.

---

## 4. HỆ THỐNG ĐÁNH GIÁ (EVALUATION)

Có 2 cấp độ đánh giá:
1. **Traditional Metrics (`evaluate.py`)**:
   - Sử dụng BLEU-4, METEOR, và BERTScore.
   - Cung cấp điểm số định lượng về N-gram overlap và semantic similarity.
2. **LLM-as-a-Judge (`llm_eval.py`)**:
   - Sử dụng Google Gemini API.
   - Vượt qua sự cứng nhắc của BLEU/METEOR bằng cách cho mô hình ngôn ngữ lớn đọc câu hỏi, ảnh (nếu cần), ground truth và câu trả lời của Model E để chấm điểm (0-10) dựa trên độ chính xác ngữ nghĩa và lập luận. Có parse thành file CSV để làm Report.

---

## 5. KIẾN TRÚC MÃ NGUỒN (SOURCE CODE)

Thư mục `new_vqa/` hiện tại (Đã clean các file rác):

```
new_vqa/
├── CONTEXT_HANDOFF.md           # File này (Ngữ cảnh tổng quát)
├── DOCUMENTATION.md             # Đang chờ update
├── README.md                    
├── train_model_e.ipynb          # Notebook thực thi 4 Phase train cho Model E (Main Entry)
├── archive/                     # Chứa các notebook và proposal cũ (Model A, B, C, D)
├── checkpoints/                 # Lưu Best state, Resume state
├── data/
│   ├── raw/train2014 & val2014  # COCO images
│   ├── vqa_e/                   # VQA-E dataset json
│   └── captions/                # COCO Captions json (cho multi-task)
└── src/
    ├── vocab.py                 # NLTK tokenization + <task_vqa> <task_cap>
    ├── dataset.py               # Dataset class trộn VQA-E và Captions theo tỷ lệ
    ├── train.py                 # Main training script (Phase 1, 2, 3) + Fixed Resume Bug
    ├── train_rl.py              # SCST Reinforcement Learning script (Phase 4)
    ├── evaluate.py              # BLEU, METEOR, BERTScore + Greedy/Beam gen
    ├── llm_eval.py              # Gemini LLM Judge
    ├── inference.py             # Gen tool 
    └── models/
        ├── vqa_models.py        # VQAModelE (Main class)
        ├── encoder_cnn.py       # CLIPViTEncoder
        ├── encoder_question.py  # BiLSTM
        └── decoder_lstm.py / decoder_attention.py # Custom LSTM có method `sample` cho RL
```

---

## 6. LƯU Ý KỸ THUẬT SIÊU QUAN TRỌNG KHI MAINTAIN

1. **Max Length:** Parameter `max_len` trong `train_rl.py` đã set từ 20 lên 60. Trong `inference.py` và `evaluate.py`, `max_len` đã set lên 100. Lý do: VQA-E có câu explanation rất dài, nếu set ngắn model sẽ bị **Cắt Cụt Đuôi (Truncation)**, dẫn tới học sai và Loss tăng đột biến.
2. **Resume Bug PyTorch:** Tuyệt đối không xóa vòng lặp `for _ in range(start_epoch): scheduler.step()` ở dòng 405 của `train.py`. Nếu xóa, PyTorch sẽ đẩy LR lên 1 con số khổng lồ (vài e-3) khi đánh lại warmup, làm Loss nổ tung.
3. **Multi-task Mixing:** Lớp Dataset tự động mix VQA-E và Captions với cơ chế lấy 1 phần Captions bù vào mỗi epoch, nếu bỏ tính năng này model sẽ suy giảm khả năng ngữ pháp.

---
**HARDWARE CHẠY HIỆN TẠI:**
- OS: Linux
- Máy local: i7-14700KF, 64GB RAM.
- GPU: **NVIDIA RTX 5070 Ti (16GB VRAM)**. Cực kỳ mạnh mẽ. Chạy mượt batch size 256 (hoặc 128 ở Mode RL).

(Mọi thiết lập đã hoàn hảo, dự án có thể chạy từ `train_model_e.ipynb` ngay bất cứ lúc nào).
