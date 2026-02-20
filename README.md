# VQA From Scratch (Visual Question Answering)

> **Status:** Developing (Sprint 1)  
> **Author:** [Tên Của Bạn]  
> **Environment:** Omarchy (Arch Linux) | PyTorch | CUDA  

Dự án xây dựng hệ thống AI có khả năng trả lời câu hỏi dựa trên hình ảnh (VQA), phát triển từ con số 0 để phục vụ mục đích học tập chuyên sâu (Deep Learning, NLP, CV).

## Mục Tiêu (Goals)

1.  **Hiểu sâu bản chất:** Tự code các module cốt lõi (LSTM, Attention, CNN Pipeline) thay vì dùng thư viện ăn sẵn.
2.  **Kiến trúc linh hoạt:** Xây dựng hệ thống theo dạng Module để dễ dàng nâng cấp từ Simple Model lên Attention Model.
3.  **Generative VQA:** Model phải *sinh ra* câu trả lời (Open-ended generation) thay vì chỉ chọn từ tập đóng (Classification).

---

## Development Roadmap (Scrum Board)

### Phase 1: The Foundation (Data Pipeline)
*Mục tiêu: Dữ liệu chảy thông suốt từ Raw -> Tensor -> DataLoader.*
- [ ] **Task 1.1:** Setup project structure (Folder, Symlinks).
- [ ] **Task 1.2:** Viết Module Vocabulary (`src/vocab.py`) xử lý cả Question & Answer.
- [ ] **Task 1.3:** Viết Script Preprocessing (`scripts/1_build_vocab.py`).
- [ ] **Task 1.4:** Viết Script Feature Extraction (`scripts/2_extract_features.py`) - Lưu tensor 3D (14x14).
- [ ] **Task 1.5:** Viết Dataset Class (`src/dataset.py`) ghép nối tất cả.

### Phase 2: The Prototype (Simple LSTM)
*Mục tiêu: "Walking Skeleton" - Model chạy được, loss giảm, chưa cần thông minh.*
- [ ] **Task 2.1:** Viết Image Encoder (Flatten features).
- [ ] **Task 2.2:** Viết Question Encoder (LSTM).
- [ ] **Task 2.3:** Viết Decoder đơn giản (Concat Image + Question -> LSTM).
- [ ] **Task 2.4:** Training Loop v1 (Overfit trên 1 batch nhỏ để test code).

### Phase 3: The Intelligence (Attention Mechanism)
*Mục tiêu: Model biết "nhìn" vào đâu khi trả lời.*
- [ ] **Task 3.1:** Implement Soft Attention Module.
- [ ] **Task 3.2:** Nâng cấp Decoder để tích hợp Attention.
- [ ] **Task 3.3:** Training trên full dataset.
- [ ] **Task 3.4:** Evaluation (BLEU Score, Accuracy).

---

## 🛠️ Architecture Overview

### 1. Data Flow
`Raw Images` -> **ResNet101** -> `Visual Features (14x14x2048)`  
`Questions` -> **Tokenizer** -> `Indices Tensor`  
`Answers` -> **Tokenizer** -> `Indices Tensor`

### 2. Model Design (Attention Variant)
* **Image Encoder:** ResNet-101 (Pretrained, remove last FC).
* **Question Encoder:** Embedding + LSTM (2 layers).
* **Fusion:** Soft Attention (Bahdanau Style).
* **Decoder:** LSTM Generator (Word-by-word generation).

---

## Project Structure

```text
vqa_scratch/
├── data/
│   ├── raw/                # Symlinks tới COCO & VQA Dataset gốc
│   ├── processed/          # Chứa vocab.json, features.h5
├── src/                    # Source code chính
│   ├── __init__.py
│   ├── vocab.py            # Xử lý ngôn ngữ
│   ├── dataset.py          # PyTorch Dataset
│   ├── model.py            # Định nghĩa kiến trúc Neural Net
├── scripts/                # Scripts chạy 1 lần (Data prep)
│   ├── 1_build_vocab.py
│   ├── 2_extract_features.py
├── checkpoints/            # Lưu model weights
└── DEV_LOG.md              # Nhật ký phát triển chi tiết
