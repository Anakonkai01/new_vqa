# Dev Log — VQA Project

## Ngữ cảnh cho chat mới

**Repo:** https://github.com/Anakonkai01/new_vqa  
**Branch hiện tại:** `experiment/model-a`  
**Mục tiêu:** Xây dựng 4 kiến trúc VQA (CNN + LSTM-Decoder), xem chi tiết trong [VQA_PROJECT_PLAN.md](VQA_PROJECT_PLAN.md)

---

## Môi trường

- **OS:** Linux
- **Python:** 3.9 (conda env `d2l`)
- **GPU:** NVIDIA GeForce MX330 — **KHÔNG tương thích PyTorch CUDA mới** (cuda capability 6.1, PyTorch yêu cầu 7.0+)
- **→ Phải train trên CPU:** `DEVICE = torch.device('cpu')`
- **Chạy script:** `python src/train.py` (từ thư mục gốc `/home/anakonkai/Work/Projects/vqa_new`)

---

## Cấu trúc thư mục hiện tại

```
src/
├── models/
│   ├── encoder_cnn.py          ✅ SimpleCNN — output (batch, 1024)
│   ├── encoder_questions.py    ✅ QuestionEncoder — output (batch, 1024)
│   ├── decoder_lstm.py         ✅ LSTMDecoder — teacher forcing mode
│   └── vqa_models.py           ✅ VQAModelA — wrapper gộp 3 thành phần
├── scripts/
│   ├── 1_build_vocab.py        ✅ build vocab_questions.json + vocab_answers.json
│   └── 2_extract_features.py   ✅ extract ResNet101 features → h5 (chỉ dùng cho Model B, D)
├── dataset.py                  ✅ VQADatasetA — load raw image, answer dạng sequence
├── vocab.py                    ✅ Vocabulary class với <pad>=0, <start>=1, <end>=2, <unk>=3
└── train.py                    🔧 Gần xong — xem bug bên dưới

create_dummy_data.py            ✅ Tạo dummy data để test pipeline (100 samples)
VQA_PROJECT_PLAN.md             ✅ Full roadmap 4 models
```

---

## Kiến trúc Model A (đã implement)

```
Input: ảnh (batch, 3, 224, 224) + câu hỏi (batch, max_q_len)

SimpleCNN:
  5x conv_block (Conv→BN→ReLU→MaxPool)
  3→64→128→256→512→1024 channels
  AdaptiveAvgPool2d(1) → flatten → Linear(1024, hidden=1024)
  Output: (batch, 1024)

QuestionEncoder:
  Embedding(vocab_q_size, 512) + LSTM(512→1024, layers=2)
  Output: hidden[-1] → (batch, 1024)

Fusion: img_feature * q_feature (Hadamard) → (batch, 1024)

LSTMDecoder (Teacher Forcing):
  h_0 = fusion.unsqueeze(0).repeat(2, 1, 1)  # (2, batch, 1024)
  c_0 = zeros_like(h_0)
  Input: answer[:, :-1] = [<start>, w1, w2]
  Target: answer[:, 1:]  = [w1, w2, <end>]
  Output: logits (batch, seq_len, vocab_a_size)

Loss: CrossEntropyLoss(ignore_index=0)
      logits.view(-1, vocab_size) vs decoder_target.contiguous().view(-1)

Optimizer: Adam lr=1e-3
Gradient clipping: max_norm=5.0
```

---

## Bug cần fix ngay khi mở chat mới

### Bug 1 — DEVICE sai (QUAN TRỌNG NHẤT)
```python
# train.py dòng ~40
# Hiện tại: ❌
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Sửa thành: ✅
DEVICE = torch.device('cpu')
```
**Lý do:** GPU MX330 detect được nhưng không chạy được → CUDA out of memory crash.

### Bug 2 — vocab_a load sai path
```python
# train.py dòng ~61
# Hiện tại: ❌ (load vocab_q cho cả vocab_a)
vocab_a.load(VOCAB_Q_PATH)

# Sửa thành: ✅
vocab_a.load(VOCAB_A_PATH)
```

### Bug 3 — encoder_questions.py tên file có 's' thừa
```python
# vqa_models.py import:
from models.encoder_questions import QuestionEncoder  # 's' ở cuối
# File thực tế tên là: encoder_questions.py — OK, khớp rồi
```

---

## Việc cần làm tiếp theo (theo thứ tự)

### Ngay lập tức
1. Fix Bug 1 + Bug 2 trong `train.py`
2. Chạy `python create_dummy_data.py` để tạo dummy data
3. Chạy `python src/scripts/1_build_vocab.py` để build vocab (nếu chưa có `data/processed/vocab_*.json`)
4. Chạy `python src/train.py` → verify pipeline chạy được trên dummy data

### Sau khi train chạy được
5. Viết `src/evaluate.py` — tính BLEU, VQA Accuracy
6. Viết `src/inference.py` — greedy decode để sinh answer từ ảnh + câu hỏi
7. Implement Model B (Pretrained ResNet, No Attention) — thêm class vào `vqa_models.py`
8. Implement Model C (Scratch CNN + Attention)
9. Implement Model D (Pretrained + Attention)
10. Viết `src/compare.py` — so sánh 4 model

---

## Dữ liệu

**Dummy data** (để test pipeline):
```bash
python create_dummy_data.py
# Tạo: data/raw/images/train2014/ (100 ảnh 224x224 random)
#       data/raw/vqa_json/v2_OpenEnded_mscoco_train2014_questions.json
#       data/raw/vqa_json/v2_mscoco_train2014_annotations.json
#       data/processed/train_features.h5
```

**Real data** (cần download, ~13GB):
```bash
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip -d data/raw/images/
```

---

## Những điều đã học (để giải thích lại nếu cần)

- **padding=1 + kernel=3:** Giữ nguyên spatial size sau Conv
- **AdaptiveAvgPool2d(1):** Squeeze spatial 7×7 → 1×1 để flatten thành vector
- **Teacher Forcing:** Dùng ground truth token làm input bước tiếp thay vì predict của bước trước
- **`target[:, :-1]` vs `target[:, 1:]`:** Shift 1 bước — input và label offset nhau 1 token
- **`contiguous().view(-1)`:** Cần thiết sau slice để reshape an toàn
- **`c_0 = zeros`:** Cell state khởi tạo trắng, chỉ h_0 mang context ảnh + câu hỏi
- **2 vocab riêng (vocab_q, vocab_a):** vocab_a nhỏ hơn, chỉ chứa từ trong answers, decoder hiệu quả hơn
- **`ignore_index=0`:** Không tính loss trên `<pad>` token
- **gradient clipping `max_norm=5.0`:** LSTM hay bị exploding gradient

---

## Ghi chú kỹ thuật

- File `encoder_questions.py` (có chữ 's') — đặt tên hơi khác convention nhưng vẫn chạy được
- `VQAModelA` trong `vqa_models.py` — tên class có chữ 'A' để phân biệt với Model B, C, D sau này
- Vocab đã có `<start>=1`, `<end>=2` → `numericalize()` tự thêm vào cả question lẫn answer
- CNN scratch chỉ có 2GB VRAM trên MX330 → **bắt buộc dùng CPU**
