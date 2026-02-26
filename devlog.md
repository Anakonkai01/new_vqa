# Development Log — VQA Project

---

## Phase 0: Khởi tạo dự án

### Thiết lập codebase

- Tạo repository `new_vqa` trên GitHub, branch `main`
- Cấu trúc modular: `src/models/`, `src/scripts/`, `data/`, `checkpoints/`
- Thiết kế theo nguyên tắc **CLI-driven** — tất cả hyperparameters qua command-line arguments, không cần sửa source code

### Xây dựng Data Pipeline

- **VQA v2.0 dataset**: COCO train2014 (~82K ảnh, ~443K QA pairs) + val2014 (~40K ảnh, ~214K QA pairs)
- Tải data từ Kaggle (train images, val images, JSON files)
- `Vocabulary` class: `word2idx`, `idx2word`, `numericalize()`, save/load JSON
- `1_build_vocab.py`: build question vocab (min_freq=3) + answer vocab (min_freq=5)
- Sử dụng official train/val split (không random_split) — đảm bảo reproducibility

### VQADataset & Collate

- Load ảnh on-the-fly, resize 224×224, ImageNet normalize
- `vqa_collate_fn`: `torch.stack` images + `pad_sequence` questions/answers
- `split` parameter phân biệt `COCO_train2014_` vs `COCO_val2014_` filename prefix
- Thêm `max_samples` để test pipeline nhanh với dummy data

---

## Phase 1: Xây dựng 4 kiến trúc

### Image Encoders

**Model A — SimpleCNN:**
- 5 convolutional blocks: Conv2d → BatchNorm → ReLU → MaxPool
- AdaptiveAvgPool2d(1) → Linear → output (B, 1024)
- Train from scratch

**Model B — ResNetEncoder:**
- ResNet101(pretrained=True), bỏ FC layer: `nn.Sequential(*list(resnet.children())[:-1])`
- Linear(2048 → 1024) projection
- `freeze=True` ban đầu, `unfreeze_top_layers()` mở layer3+layer4 cho Phase 2
- `backbone_params()` trả về params cho differential LR

**Model C — SimpleCNNSpatial:**
- 5 conv blocks giống SimpleCNN, nhưng không global pool
- `Conv2d(kernel=1)` projection → 7×7 = 49 spatial regions → output (B, 49, 1024)

**Model D — ResNetSpatialEncoder:**
- ResNet101 bỏ avgpool+fc: `nn.Sequential(*list(resnet.children())[:-2])`
- `Conv2d(kernel=1)` từ 2048 → 1024 channels, flatten → 49 regions
- unfreeze/backbone_params tương tự Model B

### Question Encoder

- `QuestionEncoder`: Embedding(padding_idx=0) → LSTM → lấy `h_n[-1]`
- Shared bởi tất cả 4 models, output (B, 1024)

### Decoders

**LSTMDecoder (Model A, B):**
- Embedding → Dropout(0.5) → LSTM(num_layers=2, dropout=0.5) → Linear(hidden→vocab)
- `forward(encoder_hidden, target_seq)`: teacher forcing, trả về full logits

**LSTMDecoderWithAttention (Model C, D):**
- BahdanauAttention: additive attention over 49 image regions
- LSTM input = concat(embedding, attention_context) → richer input tại mỗi step
- `decode_step(token, hidden, img_features)`: 1-step decode, trả về (logit, hidden, alpha)
- Alpha weights dùng cho attention heatmap visualization

### Model Wrappers

- `VQAModelA/B/C/D` trong `vqa_models.py`
- L2 normalize image features trước fusion: `F.normalize(feat, p=2, dim=...)`
- Hadamard fusion: `fusion = img_feat * q_feat`
- Initial hidden state: `h_0 = fusion.unsqueeze(0).repeat(num_layers, 1, 1)`

---

## Phase 2: Training Pipeline

### train.py — Core Training Loop

- CLI-based: `--model A/B/C/D --epochs N --lr FLOAT --batch_size N ...`
- Teacher forcing: `decoder_input = answer[:, :-1]`, `decoder_target = answer[:, 1:]`
- CrossEntropyLoss với `ignore_index=0` (bỏ qua `<pad>` tokens)
- Validation loss mỗi epoch
- History JSON lưu train/val loss (saved mỗi epoch, safe against disconnect)

### Mixed Precision (AMP)

- Auto-detect GPU capability:
  - Ampere+ (A100, H100): BFloat16 — wider dynamic range, không cần GradScaler
  - Older GPUs: Float16 + GradScaler
- `autocast(dtype=amp_dtype)` wrap forward + loss computation
- `scaler.scale(loss).backward()` + `scaler.step(optimizer)` + `scaler.update()`

### LR Scheduling

- `ReduceLROnPlateau(mode='min', factor=0.5, patience=2)`
- Step theo val loss — tự động giảm LR khi val loss plateau

### Gradient Clipping

- `clip_grad_norm_(max_norm=5.0)` — prevent exploding gradients

---

## Phase 3: Scheduled Sampling

### Cơ chế

- `ss_forward()` function trong `train.py`
- Bypass `model.forward()`, gọi trực tiếp encoders + decoder step-by-step
- Mỗi step: xác suất ε dùng GT token, (1-ε) dùng `argmax(logit).detach()`
- Epsilon decay: inverse-sigmoid `ε = k / (k + exp(epoch/k))`, k=5

### Implementation Notes

- Model A/B: gọi `model.decoder.dropout(model.decoder.embedding(tok))` → `lstm()` → `fc()`
- Model C/D: gọi `model.decoder.decode_step(tok, hidden, img_features)`
- `--scheduled_sampling` flag + `--ss_k 5.0` CLI args

---

## Phase 4: Anti-Overfitting

### Vấn đề

- Phase 1 training cho thấy val loss bắt đầu tăng sau ~epoch 11
- Model overfit do dataset lớn nhưng vocab hữu hạn

### Giải pháp — 5 kỹ thuật

1. **Data Augmentation** (`--augment`):
   - `RandomHorizontalFlip(0.5)` + `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)`
   - Chỉ áp dụng cho train, không cho val

2. **Weight Decay** (`--weight_decay 1e-5`):
   - L2 regularization trực tiếp trong Adam optimizer

3. **Early Stopping** (`--early_stopping 3`):
   - Patience = 3 epochs
   - Khi trigger: copy `model_best.pth` → `model_X_epoch{target}.pth` cho compare.py

4. **Embedding Dropout** (built-in):
   - `self.dropout = nn.Dropout(0.5)` trong LSTMDecoder
   - Áp dụng sau embedding layer: `embeds = self.dropout(self.embedding(target_seq))`

5. **LSTM Inter-layer Dropout** (built-in):
   - `nn.LSTM(dropout=0.5)` khi num_layers > 1

---

## Phase 5: Checkpoint & Storage

### Vấn đề

- Google Drive miễn phí chỉ có 15GB
- Per-epoch checkpoints cho 4 models tràn Drive nhanh
- Colab disconnect → mất tiến trình training

### Giải pháp — Milestone-only Strategy

| Checkpoint | Tần suất | Ghi đè |
|-----------|---------|--------|
| `model_X_resume.pth` | Mỗi epoch | Có — 1 file/model |
| `model_X_best.pth` | Khi val loss giảm | Có — 1 file/model |
| `model_X_epoch{10,15,20}.pth` | Milestone only | Không |

- Resume checkpoint chứa: model + optimizer + scheduler + scaler + epoch + best_val_loss + history
- Total storage: ~3 files/model thay vì 20 files/model

### Google Drive Integration

- `sync_to_drive(src, category)`: copy file từ Colab local → Drive
- `restore_from_drive(category, local_dir)`: restore khi runtime restart
- 3 thư mục Drive: `checkpoints/`, `vocab/`, `outputs/`

---

## Phase 6: Evaluation & Comparison

### Metrics Implementation

- **VQA Accuracy**: official metric — `min(matching_annotations / 3, 1.0)` với 10 human annotations/question
- **Exact Match**: strict string equality
- **BLEU-1/2/3/4**: n-gram precision với smoothing (nltk)
- **METEOR**: synonym-aware matching (nltk)

### evaluate.py

- Evaluate single model với configurable decode mode (greedy/beam)
- Load full val dataset, batch decode, compute all 7 metrics

### compare.py

- Side-by-side comparison table cho multiple models tại cùng epoch
- **Fallback logic**: nếu `model_X_epoch{N}.pth` không tồn tại (early stopping), dùng `model_X_best.pth`
- Output formatted table: Model | VQA Acc | EM | BLEU-1/2/3/4 | METEOR | Checkpoint

---

## Phase 7: Inference & Visualization

### Inference Modes

- **Greedy decode**: argmax tại mỗi step, nhanh
- **Beam search**: giữ top-k candidates, length-normalized log probability
- Separate functions cho attention (C/D) vs non-attention (A/B) models
- Batch wrappers cho evaluation pipeline

### Visualization

- `plot_curves.py`: training/val loss curves cho 4 models trên cùng đồ thị
- `visualize.py`: attention heatmap overlay (Model C/D) — reshape alpha (49,) → (7,7) → upsample

### Notebook Analysis Cells

- Metrics explanation — giải thích ý nghĩa từng metric
- Qualitative analysis — ví dụ predictions đúng/sai
- Error analysis by question type — accuracy breakdown
- Discussion — so sánh 4 architectures, kết luận

---

## Phase 8: GPU Optimizations

### Tự động detect & áp dụng

- `cudnn.benchmark = True`: auto-tune conv algorithms
- TF32 matmul + conv: Ampere+ only, ~2× faster, near-FP32 accuracy
- BFloat16 AMP: `compute_capability >= 8.0`, no GradScaler needed
- Float16 AMP + GradScaler: fallback cho older GPUs

### Fused Adam

- `_fused_adam_available()` guard: check CUDA + PyTorch 2.0+
- `fused=True` trong cả 2 optimizer calls (regular + differential LR)
- ~10-20% faster optimizer step, kernel launch reduction

### DataLoader Optimization

- `pin_memory=True`: faster CPU→GPU transfer
- `persistent_workers=True`: avoid worker respawn overhead
- `prefetch_factor=4`: pre-load next batches

---

## Bugs đã fix

### Bug 1: Optimizer Resume Crash (Phase 1→2)

**Triệu chứng:** `ValueError: loaded state dict has a different number of parameter groups`

**Nguyên nhân:** Phase 1 optimizer có 1 param group (ResNet frozen). Phase 2 với `--finetune_cnn` thêm param group cho backbone → 2 groups. Resume cố load 1-group state vào 2-group optimizer.

**Fix:** So sánh `saved_groups` vs `current_groups`. Nếu khác → skip optimizer/scheduler restore, dùng fresh optimizer với LR mới từ CLI.

### Bug 2: ss_forward Missing Dropout

**Triệu chứng:** `AttributeError: 'LSTMDecoder' object has no attribute 'dropout'`

**Nguyên nhân:** `ss_forward()` gọi `model.decoder.dropout(...)` nhưng LSTMDecoder chưa có attribute này.

**Fix:** Thêm `self.dropout = nn.Dropout(dropout)` vào LSTMDecoder.__init__() và dùng trong forward().

### Bug 3: Early Stopping → compare.py SKIP

**Triệu chứng:** compare.py in `[SKIP] model_X_epoch15.pth not found` dù model đã train xong Phase 2.

**Nguyên nhân:** Early stopping dừng training trước milestone epoch → milestone checkpoint chưa được tạo.

**Fix (2 nơi):**
1. `train.py`: khi early stopping trigger, copy `model_best.pth` → `model_X_epoch{target}.pth`
2. `compare.py`: thêm fallback — nếu epoch-specific không tồn tại → dùng `model_best.pth`

### Bug 4: Drive Storage Overflow

**Triệu chứng:** Google Drive tràn 15GB sau training Phase 1 cho 4 models.

**Nguyên nhân:** Save checkpoint mỗi epoch × 4 models × ~500MB = ~20GB/phase.

**Fix:** Milestone-only saving — chỉ lưu epochs {10, 15, 20}. Resume + best checkpoint ghi đè.

### Bug 5: Batch Size Không Công Bằng

**Triệu chứng:** So sánh 4 models không fair — batch size khác nhau.

**Fix:** Thống nhất `--batch_size 256` cho tất cả 12 training cells (4 models × 3 phases).

---

## Hardware sử dụng

| Giai đoạn | GPU | VRAM | Ghi chú |
|-----------|-----|------|---------|
| Development & testing | Local (MX330) | 2GB | Chỉ test pipeline, không train |
| Phase 1 initial | Colab RTX PRO 6000 Blackwell | 102GB | ~200s/epoch |
| Phase 1→3 retrain | Colab A100 | 40GB | ~600-700s/epoch |

**Lưu ý:** Tốc độ training phụ thuộc GPU Colab cấp phát — cùng code nhưng khác GPU cho thời gian epoch khác nhau. Đây không phải bug code.

---

## Trạng thái cuối cùng

- ✅ Tất cả code hoàn chỉnh, 0 errors
- ✅ 4 models implemented và tested
- ✅ 3-phase training pipeline
- ✅ Anti-overfitting (5 kỹ thuật)
- ✅ GPU optimizations (auto-detect)
- ✅ Evaluation (7 metrics) + Comparison
- ✅ Google Drive integration
- ✅ Notebook pipeline (vqa_colab.ipynb)
- ✅ Documentation đầy đủ
