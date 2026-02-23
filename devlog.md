# Dev Log — VQA Project

## Prompt cho chat mới

```
Đọc kỹ devlog.md và VQA_PROJECT_PLAN.md trước khi làm gì.
Sau khi đọc xong, tóm tắt lại trạng thái hiện tại để tôi xác nhận bạn đã nẵm đúng ngữ cảnh.

Project VQA PyTorch — 4 kiến trúc (A/B/C/D), CNN+LSTM encoder, LSTM decoder.
TOÀN BỘ CODE ĐÃ HOÀN CHỈNH. Việc còn lại là TRAIN trên Kaggle rồi về evaluate.

Kaggle: GPU tự động detect qua torch.cuda.is_available().

Phong cách: thầy hướng dẫn — giải thích kỹ, để tôi tự code trừ khi tôi nói "hãy thực hiện".
```

---

## Cập nhật lần cuối: 2026-02-21 — CODE HOÀN CHỈNH, SẴN SÀNG PUSH

---

## Môi trường

- **OS:** Linux
- **Python:** 3.9 (conda env `d2l`) — local dùng env này
- **GPU local:** NVIDIA GeForce MX330 — KHÔNG dùng được CUDA (sm_61, yêu cầu sm_70+)
- **DEVICE:** `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` — tự detect
- **Working dir:** `/home/anakonkai/Work/Projects/vqa_new`
- **Branch:** `experiment/model-a`
- **Kaggle:** clone repo → train → download checkpoints/ về local → evaluate/visualize

---

## Trạng thái code — TẤT CẢ HOÀN CHỈNH ✅

```
src/
├── models/
│   ├── encoder_cnn.py        ✅ SimpleCNN, SimpleCNNSpatial, ResNetEncoder, ResNetSpatialEncoder
│   ├── encoder_question.py   ✅ QuestionEncoder — LSTM, output (batch, 1024)
│   ├── decoder_lstm.py       ✅ LSTMDecoder — teacher forcing
│   ├── decoder_attention.py  ✅ BahdanauAttention + LSTMDecoderWithAttention
│   └── vqa_models.py         ✅ VQAmodelA, VQAModelB, VQAModelC, VQAModelD
├── dataset.py                ✅ VQADatasetA + vqa_collate_fn
├── vocab.py                  ✅ Vocabulary class
├── train.py                  ✅ MODEL_TYPE config, factory fn, history JSON, auto DEVICE
├── inference.py              ✅ greedy_decode (A/B) + greedy_decode_with_attention (C/D)
├── evaluate.py               ✅ --model_type, Exact Match + BLEU-1/2/3/4 + METEOR
├── compare.py                ✅ so sánh 4 model, bảng kết quả
├── plot_curves.py            ✅ vẽ training curves từ history JSON
└── visualize.py              ✅ attention heatmap cho C/D
```

---

## Kiến trúc 4 Models

| Model | CNN Encoder | Decoder | CNN Output | Test |
|-------|-------------|---------|------------|------|
| A | SimpleCNN (scratch) | LSTMDecoder (no attn) | (batch, 1024) | ✅ (2,9,50) |
| B | ResNetEncoder (pretrained, frozen) | LSTMDecoder (no attn) | (batch, 1024) | logic OK |
| C | SimpleCNNSpatial (scratch) | LSTMDecoderWithAttention | (batch, 49, 1024) | ✅ (2,9,50) |
| D | ResNetSpatialEncoder (pretrained, frozen) | LSTMDecoderWithAttention | (batch, 49, 1024) | logic OK |

B và D cần ResNet101 weights (~170MB, cache `~/.cache/torch`) — download lần đầu chạy.

---

## Chi tiết quan trọng

### train.py
- `MODEL_TYPE = 'A'` ở config — đổi thành B/C/D để train model khác
- `DEVICE` tự detect GPU/CPU — không cần sửa khi chạy Kaggle
- Lưu checkpoint: `checkpoints/model_a_epoch1.pth`, ..., `model_a_epoch10.pth`
- Lưu history: `checkpoints/history_model_a.json` — cập nhật sau **mỗi epoch**

### inference.py
- A/B: `greedy_decode(model, img_tensor, q_tensor, vocab_a)`
- C/D: `greedy_decode_with_attention(model, img_tensor, q_tensor, vocab_a)`
- `get_model(type, q_size, a_size)` — factory function, dùng chung với evaluate/compare

### evaluate.py
```bash
python src/evaluate.py --model_type A
python src/evaluate.py --model_type C --checkpoint checkpoints/model_c_epoch5.pth
python src/evaluate.py --model_type B --num_samples 100
```
Output: Exact Match, BLEU-1/2/3/4, METEOR

### compare.py
```bash
python src/compare.py                     # cả 4 model epoch 10
python src/compare.py --epoch 5
python src/compare.py --models A,C --num_samples 50
```
Model nào thiếu checkpoint → tự động SKIP thay vì crash.

### plot_curves.py
```bash
python src/plot_curves.py                 # vẽ cả 4 model
python src/plot_curves.py --models A,C --output results/curves.png
```
Đọc `checkpoints/history_model_*.json` → lưu `checkpoints/training_curves.png`

### visualize.py (chỉ C và D)
```bash
python src/visualize.py --model_type C
python src/visualize.py --model_type D --epoch 5 --sample_idx 3
```
Attention heatmap (jet) chồng lên ảnh cho từng token → lưu `checkpoints/attn_model_c.png`

---

## Kỹ thuật quan trọng

- **Class tên:** `VQAmodelA` (chữ m thường) — tên cũ, giữ nguyên không đổi
- **File:** `encoder_question.py` (không có 's') — tên file thực tế
- **ResNet weights:** `models.ResNet101_Weights.DEFAULT` — KHÔNG phải string `"Default"`
- **Vocab:** `<pad>=0, <start>=1, <end>=2, <unk>=3`
- **Train/val split:** 90/10, `manual_seed=42` — phải nhất quán train.py ↔ evaluate.py
- **Teacher forcing:** `answer[:, :-1]` input, `answer[:, 1:]` target
- **Attention input:** lstm `input_size = embed_size + hidden_size` (ghép embed + context)
- **ignore_index=0:** CrossEntropyLoss không tính loss trên `<pad>`
- **`[:-1]` vs `[:-2]`:** ResNet `[:-1]` giữ avgpool (1×1), `[:-2]` giữ spatial (7×7)
- **contiguous().view(-1):** cần sau tensor slice để reshape an toàn

---

## Workflow Kaggle

```bash
# 1. Clone
git clone https://github.com/Anakonkai01/new_vqa && cd new_vqa

# 2. Cài dependencies
pip install torch torchvision nltk matplotlib pillow tqdm

# 3. Build data (hoặc dùng real COCO data)
python create_dummy_data.py
python src/scripts/1_build_vocab.py

# 4. Train — đổi MODEL_TYPE trong train.py rồi chạy
# MODEL_TYPE = 'A'
python src/train.py
# MODEL_TYPE = 'B'
python src/train.py
# MODEL_TYPE = 'C'
python src/train.py
# MODEL_TYPE = 'D'
python src/train.py

# 5. Download toàn bộ thư mục checkpoints/ về local

# 6. Evaluate và visualize (local)
python src/compare.py
python src/plot_curves.py
python src/visualize.py --model_type C
python src/visualize.py --model_type D
```

---

## Dữ liệu thực

```bash
# Download COCO train2014 (~13GB)
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip -d data/raw/images/

# Build vocab và features
python src/scripts/1_build_vocab.py
```

---

## Những điều đã học

- Teacher Forcing: `answer[:, :-1]` input, `answer[:, 1:]` target — shift 1 bước
- `random_split` với `manual_seed(42)` — train.py và evaluate.py PHẢI dùng cùng seed
- `model.eval()` + `torch.no_grad()` — bắt buộc cho validation/inference
- Bahdanau Attention: query=hidden_state, key=value=image_regions → context=weighted sum
- `[:-1]` vs `[:-2]` ResNet: giữ avgpool (1×1) vs giữ spatial feature map (7×7)
- `Conv2d(kernel=1)` = Linear áp độc lập lên từng vùng spatial
- `squeeze(1)` không phải `squeeze()` — tránh squeeze nhầm batch dim
- `contiguous().view(-1)` — cần sau tensor slice để reshape an toàn
- `matplotlib.use('Agg')` — headless server/Kaggle không cần display
- History JSON ghi sau mỗi epoch — data không mất nếu training bị ngắt
