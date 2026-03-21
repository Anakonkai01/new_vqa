Bạn là senior AI engineer tiếp tục dự án Generative VQA (Model G).
Đọc memory trước, sau đó đọc các file sau theo thứ tự:

1. research/Model_G_Architecture_Specification_v2.md
2. research/Model_G_Data_Strategy.md
3. reports/model_g_data_phase.md

CONTEXT HIỆN TẠI:

DATA PHASE: HOÀN THÀNH
- 184,954 filtered samples tại data/processed/merged_train_filtered.json
- Vocab: 9,937 question / 11,271 answer tokens tại data/processed/vocab_{questions,answers}.json
- 6 scripts đã viết tại src/scripts/
- Chi tiết đầy đủ + debug history: reports/model_g_data_phase.md

CODEBASE AUDIT: HOÀN THÀNH
4 bugs confirmed + structural problems → Selective Targeted Rewrite.

REFACTOR TIẾN ĐỘ: Step A ✅ Step B ✅ Step C ⬜ Step D ⬜

═══════════════════════════════════════════════════
STEP A: HOÀN THÀNH ✅ (2026-03-20)
═══════════════════════════════════════════════════
Files tạo mới (không đụng file cũ nào):
  src/config/__init__.py
  src/config/model_config.py    — ModelConfig, EncoderConfig, DecoderConfig, FusionConfig
  src/config/train_config.py    — TrainConfig, DataConfig, OptimizerConfig, PhaseConfig
  src/models/base.py            — VQAOutput, VQAModelProtocol, DecoderProtocol
  src/models/registry.py        — build_model(), build_encoder(), build_decoder(), build_fusion()

Key design:
- ModelConfig.from_args(args)    → bridge từ argparse (train.py vẫn hoạt động)
- ModelConfig.model_g_full()     → preset G1-G5 all enabled
- TrainConfig.model_g_default()  → 4-phase preset (15+10+7+3 epochs)
- MODEL_REGISTRY dict + @register_model decorator → Step C sẽ đăng ký 'G'
- VQAOutput dataclass: logits + attention_img + attention_q + pgn_weights + infonce_z + coverage
- build_model() dispatches to legacy VQAModelA/B/C/D/E/F via _build_legacy() (lazy import)

Verification: conda run -n d2l python -c "from config.model_config import ModelConfig; ..."
→ All 6 tests passed.

═══════════════════════════════════════════════════
STEP B: HOÀN THÀNH ✅ (2026-03-20)
═══════════════════════════════════════════════════
Files tạo mới (không đụng file cũ nào):
  src/data/__init__.py
  src/data/dataset.py     — VQAGenerativeDataset (1 class, thay 3-class inheritance)
  src/data/collate.py     — VQABatch dataclass + image_collate_fn + butd_collate_fn
  src/data/samplers.py    — build_mixed_sampler (N-source) + build_replay_sampler

Key design:
- Composition not inheritance: feature_loader Callable thay BUTDDataset(VQAEDataset)
- make_image_loader(image_dir, split, augment)  → f(img_id) → Tensor(3,224,224)
  - Auto-fallback: thử train2014/ trước, val2014/ sau (fix A-OKVQA COCO-2017 images)
- make_butd_loader(feat_dir) → f(img_id) → (feat_tensor, label_names_or_None)
- VQAGenerativeDataset.from_merged_json()  — load merged_train_filtered.json
- VQAGenerativeDataset.from_vqa_v2()       — load VQA v2.0 JSONs
- VQAGenerativeDataset.from_legacy_vqae_json() — bridge cho code cũ

BUG FIX QUAN TRỌNG (G5):
  length_bin được recompute từ FULL sequence (answer + because + explanation)
  Kết quả: 20.0% LONG (thay vì 9.1% từ explanation-only trong JSON)
  Code: _full_seq_length_bin(answer, explanation) → 0/1/2

VQABatch dataclass fields:
  feats        (B,3,H,W) hoặc (B,max_k,D)
  questions    (B,max_q)
  targets      (B,max_t)
  img_mask     (B,max_k) bool — None cho image batches
  length_bins  (B,) int64     — G5, None nếu không dùng
  label_tokens (B,max_k,max_toks) int64 — G2, None nếu không có label metadata
  sources      List[str]

build_mixed_sampler([ds_e, ds_x, ds_a], [0.4, 0.3, 0.3]) — Phase 1 mix
build_replay_sampler(expl_ds, vqa_v2_ds, replay_fraction=0.2) — Phase 2/3

Verification: 10 tests passed (length_bin, VQABatch, vocab load, __getitem__,
              DataLoader, sources_filter, 3-source mixed sampler, always_long,
              replay sampler, aokvqa val2014 fallback).

═══════════════════════════════════════════════════
STEP C: CHƯA LÀM ⬜ (tiếp theo)
═══════════════════════════════════════════════════
Mục tiêu: Reorganize src/models/ + 1 VQAModel class thay 6 + backward compat A-F checkpoints

Files cần tạo/sửa:
  src/models/encoders/cnn.py        — SimpleCNN, SimpleCNNSpatial
  src/models/encoders/resnet.py     — ResNetEncoder, ResNetSpatialEncoder
  src/models/encoders/convnext.py   — ConvNeXtSpatialEncoder
  src/models/encoders/butd.py       — BUTDFeatureEncoder
  src/models/encoders/question.py   — QuestionEncoder (move từ encoder_question.py)
  src/models/decoders/base.py       — DecoderProtocol (move từ models/base.py)
  src/models/decoders/lstm.py       — LSTMDecoder
  src/models/decoders/attention.py  — LSTMDecoderWithAttention
  src/models/fusion/gated.py        — GatedFusion
  src/models/fusion/mutan.py        — MUTANFusion
  src/models/vqa_model.py           — REWRITE: 1 VQAModel class (thay VQAModelA/B/C/D/E/F)

Yêu cầu backward compat:
- load_checkpoint(path) → detect legacy model type từ state_dict keys
- VQAModel.from_config(ModelConfig) → factory
- @register_model('G') sẽ được gọi sau khi VQAModel được viết

═══════════════════════════════════════════════════
STEP D: CHƯA LÀM ⬜ (sau Step C)
═══════════════════════════════════════════════════
Mục tiêu: Implement G1–G5 per Architecture_Specification_v2.md

G1: BUTDFeatureEncoder: geo_dim 5→7 (thêm w/W và h/H vào spatial vector)
G2: LSTMDecoderWithAttention: 3-way PGN với label_tokens từ VQABatch
G3: InfoNCE projection heads (training only, discarded at inference)
G4: OHP reward trong scst.py
G5: LengthEmbedding(3, 64) concat vào decoder input + min_decode_len beam constraint

═══════════════════════════════════════════════════
FILE STRUCTURE (sau Steps A+B)
═══════════════════════════════════════════════════
src/
├── config/                  ← NEW (Step A)
│   ├── __init__.py
│   ├── model_config.py
│   └── train_config.py
├── models/
│   ├── base.py              ← NEW (Step A): VQAOutput, protocols
│   ├── registry.py          ← NEW (Step A): build_model()
│   ├── vqa_models.py        ← UNTOUCHED (legacy A-F)
│   ├── encoder_cnn.py       ← UNTOUCHED
│   ├── encoder_question.py  ← UNTOUCHED
│   ├── decoder_lstm.py      ← UNTOUCHED
│   ├── decoder_attention.py ← UNTOUCHED
│   └── pointer_generator.py ← UNTOUCHED
├── data/                    ← NEW (Step B)
│   ├── __init__.py
│   ├── dataset.py
│   ├── collate.py
│   └── samplers.py
├── dataset.py               ← UNTOUCHED (legacy, train.py still imports here)
├── train.py                 ← UNTOUCHED (still works)
└── ...

MÔIWTRƯỜNG: conda env d2l, RTX 5070 Ti 16GB, Python 3.11, PyTorch 2.x
CONSTRAINT BẤT BIẾN: LSTM core, NO transformers, cross-attention legal, self-attention FORBIDDEN
