# VQA Expansion Plan — Implementation Reference
# Last updated: 2026-03-17
# Sources: codebase analysis + Gemini research v1 + Gemini research v2-update
#
# CONSTRAINTS (hard):
#   1. LSTM must remain the core temporal sequence processor
#   2. NO transformers anywhere (no ViT, no BERT, no self-attention stacks)
#
# HOW TO USE THIS FILE:
#   - Each tier has a [ ] checkbox. Mark [x] when complete, [~] when in-progress.
#   - "EXACT CHANGE" blocks show precisely what to add/replace in each file.
#   - Dependencies listed so tiers are done in safe order.

---

## PROGRESS TRACKER

### Architecture Tiers
- [x] Tier 0   — Free wins (length penalty α=0.7, min_len=5 in beam search)
- [x] Tier D1  — Data bug fixes (flip guard, RandAugment, RandomResizedCrop, RandomErasing, all annotations)
- [x] Tier 1   — LSTM fortification (LayerNormLSTMStack, WeightDrop, HighwayLayer — flags: --layer_norm --dropconnect)
- [x] Tier 2   — Dense Co-Attention DCAN (replaces BahdanauAttention — flag: --dcan)
- [ ] Tier 3A  — ConvNeXt-Base encoder (replaces ResNetSpatialEncoder for Model E)
- [ ] Tier 4   — MUTAN Tucker Fusion (replaces GatedFusion for Model E)
- [ ] Tier 5   — Pointer-Generator decoder (replaces output head)
- [ ] Tier 6   — Abstention-Aware CSS (counterfactual robustness)
- [ ] Tier 7   — Deep BiLSTM + GloVe 840B + Char-CNN question encoder
- [ ] Tier 8   — SCST Reinforcement Learning (Phase 4 training)
- [ ] Tier 3B  — Faster R-CNN BUTD encoder (Model F, highest ceiling)
- [ ] Tier 9   — ConceptNet + GNN knowledge integration (optional)

### Data Tiers
- [ ] Tier D2  — Dataset expansion (VQA v2.0 full + Visual Genome pre-train + GQA)
- [ ] Tier D3  — Answer distribution balancing (freq weighting + soft labels)
- [ ] Tier D4  — Task Progressive Curriculum Learning (TPCL)
- [ ] Tier D5  — LLM synthetic data + hallucination noise filter

---

## CURRENT CODEBASE SNAPSHOT
*(as of 2026-03-17 — read this before editing any file)*

### File Map
```
src/
├── models/
│   ├── encoder_cnn.py          # SimpleCNN, SimpleCNNSpatial, ResNetEncoder, ResNetSpatialEncoder
│   ├── encoder_question.py     # QuestionEncoder (BiLSTM + attention pooling)
│   ├── decoder_lstm.py         # LSTMDecoder (Models A/B, no attention)
│   ├── decoder_attention.py    # BahdanauAttention, LSTMDecoderWithAttention (Models C/D)
│   └── vqa_models.py           # GatedFusion, VQAModelA/B/C/D
├── train.py                    # Main entry, 3-phase training, argparse
├── inference.py                # Greedy + beam search decode (single + batch, attn variants)
├── evaluate.py                 # 7-metric evaluation
├── dataset.py                  # VQAEDataset, VQADataset, vqa_collate_fn
├── vocab.py                    # Vocabulary (4 special tokens: pad=0, start=1, end=2, unk=3)
├── compare.py                  # Multi-model comparison
├── plot_curves.py              # Training curve plotting
├── visualize.py                # Attention heatmaps (C/D only)
├── scripts/
│   └── 1_build_vocab.py        # Builds vocab_questions.json + vocab_answers.json
└── training/                   # (empty — new training utilities go here)
```

### Exact Current Class Signatures

**encoder_cnn.py**
```python
class SimpleCNN(nn.Module):
    def __init__(self, output_size=1024)
    def forward(self, x) -> (B, 1024)           # x: (B,3,224,224)

class SimpleCNNSpatial(nn.Module):
    def __init__(self, output_size=1024)
    def forward(self, x) -> (B, 49, 1024)       # keeps 7×7 spatial grid

class ResNetEncoder(nn.Module):
    def __init__(self, output_size=1024, freeze=True)
    def forward(self, x) -> (B, 1024)
    def unfreeze_top_layers(self)               # unfreezes layer3, layer4, fc
    def backbone_params(self)                   # returns trainable backbone params

class ResNetSpatialEncoder(nn.Module):
    def __init__(self, output_size=1024, freeze=True)
    def forward(self, x) -> (B, 49, 1024)       # Conv2d(2048→1024) projection
    def unfreeze_top_layers(self)               # unfreezes layer3, layer4, proj
    def backbone_params(self)
```

**encoder_question.py**
```python
class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers,
                 dropout=0.5, pretrained_embeddings=None)
    def forward(self, questions) -> (q_feature, q_hidden_states)
        # q_feature:       (B, hidden_size)       — concat fwd+bwd last hidden
        # q_hidden_states: (B, max_len, hidden_size) — all timesteps
    # BiLSTM: hidden_size//2 per direction → concat → hidden_size
```

**decoder_lstm.py**
```python
class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers,
                 dropout=0.5, pretrained_embeddings=None)
    def forward(self, encoder_hidden, target_seq) -> logits (B, max_len, vocab_size)
    # Weight tying: fc.weight = embedding.weight
    # out_proj: Linear(hidden_size → embed_size)
    # fc:       Linear(embed_size → vocab_size, bias=False)
```

**decoder_attention.py**
```python
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, attn_dim=512, use_coverage=False)
    def forward(self, hidden, img_features, coverage=None) -> (context, alpha)
        # hidden:       (B, hidden_size)
        # img_features: (B, num_regions, hidden_size)
        # context:      (B, hidden_size)
        # alpha:        (B, num_regions)
    # W_h:   Linear(hidden_size → attn_dim)
    # W_img: Linear(hidden_size → attn_dim)
    # W_cov: Linear(1 → attn_dim, bias=False)  [if use_coverage]
    # v:     Linear(attn_dim → 1, bias=False)

class LSTMDecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers,
                 attn_dim=512, dropout=0.5, pretrained_embeddings=None,
                 use_coverage=False)
    def forward(self, encoder_hidden, img_features, q_hidden_states, target_seq)
        # encoder_hidden:  (B, hidden_size)   — from GatedFusion, inits h_0 layer 0
        # img_features:    (B, 49, hidden_size)
        # q_hidden_states: (B, q_len, hidden_size)
        # target_seq:      (B, max_a_len)
        # returns: (logits: B×max_a_len×vocab_size, coverage_loss: scalar)
    def decode_step(self, token, hidden, img_features, q_hidden_states, coverage=None)
        # returns: (logit: B×vocab_size, hidden, img_alpha: B×num_regions, coverage)
    # LSTM input_size = embed_size + hidden_size * 2
    # img_attention: BahdanauAttention
    # q_attention:   BahdanauAttention
```

**vqa_models.py**
```python
class GatedFusion(nn.Module):
    def __init__(self, hidden_size)
    def forward(self, img_feature, q_feature) -> (B, hidden_size)
    # gate = σ(Linear(2H→H)([img; q]))
    # out  = gate * tanh(W_img(img)) + (1-gate) * tanh(W_q(q))
    # LayerNorm applied to output

class VQAModelA(nn.Module):
    # SimpleCNN + QuestionEncoder + GatedFusion + LSTMDecoder
    def __init__(self, vocab_size, answer_vocab_size, embed_size=512,
                 hidden_size=1024, num_layers=2, dropout=0.5,
                 pretrained_q_emb=None, pretrained_a_emb=None)
    def forward(self, images, questions, target_seq) -> logits (B, max_a_len, answer_vocab_size)

class VQAModelB(nn.Module):
    # ResNetEncoder + QuestionEncoder + GatedFusion + LSTMDecoder
    def __init__(self, vocab_size, answer_vocab_size, embed_size=512,
                 hidden_size=1024, num_layers=2, freeze=True, dropout=0.5,
                 pretrained_q_emb=None, pretrained_a_emb=None)
    def forward(...) -> logits

class VQAModelC(nn.Module):
    # SimpleCNNSpatial + QuestionEncoder + GatedFusion + LSTMDecoderWithAttention
    def __init__(self, vocab_size, answer_vocab_size, embed_size=512,
                 hidden_size=1024, num_layers=2, attn_dim=512,
                 dropout=0.5, use_coverage=False,
                 pretrained_q_emb=None, pretrained_a_emb=None)
    def forward(images, questions, target_seq) -> (logits, coverage_loss)

class VQAModelD(nn.Module):
    # ResNetSpatialEncoder + QuestionEncoder + GatedFusion + LSTMDecoderWithAttention
    def __init__(self, vocab_size, answer_vocab_size, embed_size=512,
                 hidden_size=1024, num_layers=2, attn_dim=512,
                 freeze_cnn=True, dropout=0.5, use_coverage=False,
                 pretrained_q_emb=None, pretrained_a_emb=None)
    def forward(...) -> (logits, coverage_loss)
```

**train.py key args** (existing):
```
--model         A/B/C/D
--epochs        int
--lr            float (default 1e-3)
--batch_size    int (default 128)
--resume        path
--scheduled_sampling  bool
--ss_k          float (default 5.0)
--finetune_cnn  bool  (B/D only)
--cnn_lr_factor float (default 0.1)
--augment       bool
--glove         bool
--glove_dim     int (default 300)
--coverage      bool (C/D only)
--coverage_lambda float (default 1.0)
--accum_steps   int (default 1)
--warmup_epochs int (default 3)
--dropout       float (default 0.5)
--grad_clip     float (default 5.0)
--label_smoothing float (default 0.1)
--early_stopping int (default 0)
--weight_decay  float (default 1e-5)
--num_workers   int (default 4)
--no_compile    bool
```

**Data paths** (hardcoded in train.py):
```
TRAIN_IMAGE_DIR = "data/raw/images/train2014"
TRAIN_VQA_E_JSON = "data/raw/vqa_e_json/VQA-E_train_set.json"
VAL_IMAGE_DIR = "data/raw/images/val2014"
VAL_VQA_E_JSON = "data/raw/vqa_e_json/VQA-E_val_set.json"
VOCAB_Q_PATH = "data/processed/vocab_questions.json"
VOCAB_A_PATH = "data/processed/vocab_answers.json"
```

**Default hyperparameters** (consistent across all models):
```
embed_size  = 512
hidden_size = 1024
num_layers  = 2
attn_dim    = 512
dropout     = 0.5
```

**Current best results** (Model D, epoch ~20):
```
BLEU-1: ~38%    BLEU-4: 11.59%
METEOR: 35.94%  ROUGE-L: 42.70%
Exact Match: ~8%
```

---

## MODEL NAMING CONVENTION

| Model | Encoder | Attention | Fusion | Decoder | Status |
|---|---|---|---|---|---|
| A | SimpleCNN | None | GatedFusion | LSTMDecoder | Exists |
| B | ResNet101 | None | GatedFusion | LSTMDecoder | Exists |
| C | SimpleCNNSpatial | Bahdanau | GatedFusion | LSTMDecoderWithAttention | Exists |
| D | ResNetSpatialEncoder | Bahdanau | GatedFusion | LSTMDecoderWithAttention | Exists (best) |
| **E** | **ConvNeXt-Base** | **DCAN** | **MUTAN** | **LSTM+Attn+PGN** | To build |
| **F** | **FasterRCNN BUTD** | **DCAN** | **MUTAN** | **LSTM+Attn+PGN** | To build (after E) |

---

---

# PART I — ARCHITECTURE TIERS

---

## Tier 0 — Free Wins
**Status: [ ] TODO**
**Effort: 2–4 hours**
**Dependencies: none**
**Files: `src/train.py`, `src/inference.py`**

These are already coded in the codebase but not fully activated.

### 0A — GloVe embeddings
Already supported via `--glove` flag in train.py and `pretrained_embeddings` param in all encoders/decoders. Just use the flag:
```bash
python src/train.py --model D --glove --glove_dim 300 ...
```
Download: `wget http://nlp.stanford.edu/data/glove.840B.300d.zip` → unzip to `data/raw/glove/`

### 0B — Length penalty in beam search
**File: `src/inference.py`**

Current length normalization (line ~220 in beam_search_decode):
```python
score / max(len(sequence) - 1, 1)
```
Replace with:
```python
score / (max(len(sequence) - 1, 1) ** 0.7)   # alpha=0.7 length penalty
```
Apply to ALL beam_search functions: `beam_search_decode`, `beam_search_decode_with_attention`,
`batch_beam_search_decode`, `batch_beam_search_decode_with_attention`.

### 0C — Minimum decode length
**File: `src/inference.py`**

In each beam search decode loop, after computing next token log_probs:
```python
MIN_LEN = 5
if step < MIN_LEN:
    log_probs[:, vocab_a.word2idx['<end>']] = float('-inf')
```
Apply to all 4 beam search variants.

### 0D — No-repeat n-gram already implemented
`no_repeat_ngram_size=3` is already in beam search. Ensure it's used in evaluate.py calls.

---

## Tier D1 — Data Bug Fixes + Augmentation
**Status: [ ] TODO**
**Effort: 2–3 hours**
**Dependencies: none**
**Files: `src/dataset.py`**

### D1A — Horizontal flip guard (BUG FIX)
**File: `src/dataset.py`**

Current code (both VQAEDataset and VQADataset `__init__` augment block):
```python
transforms.RandomHorizontalFlip(p=0.5),
```

This is WRONG for spatial questions. Replace with a custom transform:

```python
# Add this class at top of dataset.py (after imports):
SPATIAL_KEYWORDS = {'left', 'right', 'east', 'west', 'beside', 'next to',
                    'leftmost', 'rightmost', 'lefthand', 'righthand'}

class SpatialAwareHorizontalFlip:
    """Skip horizontal flip if question contains spatial direction words."""
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img_and_question):
        img, question = img_and_question
        words = set(question.lower().split())
        if words & SPATIAL_KEYWORDS:
            return img     # skip flip
        if torch.rand(1).item() < self.p:
            return TF.hflip(img)
        return img
```

**Important:** This requires passing question text into the transform. Simpler alternative (recommended for minimal refactor):

```python
# In VQAEDataset.__getitem__ and VQADataset.__getitem__, AFTER loading image:
import torchvision.transforms.functional as TF
q_words = set(q_text.lower().split())
has_spatial = bool(q_words & SPATIAL_KEYWORDS)
# Only apply flip augmentation if augment=True and no spatial keywords
if self.augment and not has_spatial and torch.rand(1).item() < 0.5:
    image = TF.hflip(image)
# Then apply the rest of transforms (without RandomHorizontalFlip):
img_tensor = self.transform_no_flip(image)
```

Create two transforms in `__init__`:
- `self.transform_base` — Resize + ColorJitter + ToTensor + Normalize (no flip)
- Flip applied manually in `__getitem__` with spatial check

### D1B — RandomResizedCrop + RandAugment
**File: `src/dataset.py`**

Replace the augment transform block in both `VQAEDataset.__init__` and `VQADataset.__init__`:

```python
# BEFORE (current):
if augment:
    self.transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# AFTER:
if augment:
    self.transform_base = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    ])
else:
    self.transform_base = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
self.augment = augment
```

### D1C — All 10 annotations (VQADataset)
**File: `src/dataset.py` — `VQADataset.__init__` and `__getitem__`**

```python
# In __init__, change:
self.qid2ans = {ann['question_id']: ann['multiple_choice_answer'] for ann in annotations}
# TO:
self.qid2answers = {ann['question_id']: [a['answer'] for a in ann['answers']]
                    for ann in annotations}

# In __getitem__, change:
a_text = self.qid2ans.get(q_id, "")
# TO:
answers = self.qid2answers.get(q_id, [""])
a_text = random.choice(answers)   # import random at top
```

### D1D — All VQA-E explanations (VQAEDataset)
**File: `src/dataset.py` — `VQAEDataset.__getitem__`**

```python
# BEFORE:
explanation = exp_list[0] if exp_list and isinstance(exp_list[0], str) else ''
# AFTER:
valid_exps = [e for e in exp_list if isinstance(e, str) and e.strip()]
explanation = random.choice(valid_exps) if valid_exps else ''
```

---

## Tier 1 — LSTM Structural Fortification
**Status: [ ] TODO**
**Effort: 2–3 days**
**Dependencies: none (foundational — do before Tiers 2–8)**
**Files: `src/models/decoder_lstm.py`, `src/models/decoder_attention.py`**

### 1A — LayerNormLSTMCell
Add this new class to `src/models/decoder_lstm.py` (before `LSTMDecoder`):

```python
class LayerNormLSTMCell(nn.Module):
    """LSTM cell with LayerNorm on each gate's pre-activation.
    Replaces nn.LSTMCell. Stabilizes training on variable-length sequences."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size  = input_size
        # Single combined linear: gates for i, f, g, o
        self.linear = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        # Separate LayerNorm for each gate
        self.ln_i = nn.LayerNorm(hidden_size)
        self.ln_f = nn.LayerNorm(hidden_size)
        self.ln_g = nn.LayerNorm(hidden_size)
        self.ln_o = nn.LayerNorm(hidden_size)
        self.ln_c = nn.LayerNorm(hidden_size)  # cell state norm

    def forward(self, x, hx):
        # x:  (B, input_size)
        # hx: tuple(h: B×H, c: B×H)
        h, c = hx
        gates = self.linear(torch.cat([x, h], dim=-1))   # (B, 4H)
        i_raw, f_raw, g_raw, o_raw = gates.chunk(4, dim=-1)
        i = torch.sigmoid(self.ln_i(i_raw))
        f = torch.sigmoid(self.ln_f(f_raw))
        g = torch.tanh(self.ln_g(g_raw))
        o = torch.sigmoid(self.ln_o(o_raw))
        c_new = f * c + i * g
        h_new = o * torch.tanh(self.ln_c(c_new))
        return h_new, c_new
```

### 1B — DropConnect (WeightDrop)
Add this class to `src/models/decoder_lstm.py`:

```python
class WeightDrop(nn.Module):
    """AWD-LSTM DropConnect: zeros weights in hidden-to-hidden matrices.
    Regularizes without breaking LSTM temporal memory flow."""
    def __init__(self, module, weights, dropout=0.5):
        super().__init__()
        self.module   = module
        self.weights  = weights  # e.g. ['weight_hh_l0']
        self.dropout  = dropout
        self._setup()

    def _setup(self):
        for w_name in self.weights:
            raw = getattr(self.module, w_name)
            self.module.register_parameter(w_name + '_raw', nn.Parameter(raw.data))
            delattr(self.module, w_name)
            setattr(self.module, w_name, raw.data.clone())

    def _setweights(self):
        for w_name in self.weights:
            raw = getattr(self.module, w_name + '_raw')
            if self.training:
                w = F.dropout(raw, p=self.dropout, training=True)
            else:
                w = raw
            setattr(self.module, w_name, w)

    def forward(self, *args, **kwargs):
        self._setweights()
        return self.module(*args, **kwargs)
```

Usage in `LSTMDecoder.__init__`:
```python
# Replace: self.lstm = nn.LSTM(...)
# With:
_lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True, dropout=dropout)
self.lstm = WeightDrop(_lstm, ['weight_hh_l0'], dropout=0.3)
```

### 1C — Highway Connections
Add this class to `src/models/decoder_lstm.py`:

```python
class HighwayLayer(nn.Module):
    """Gated highway skip connection between LSTM layers.
    h_out = gate * lstm_out + (1 - gate) * h_in"""
    def __init__(self, hidden_size):
        super().__init__()
        self.gate = nn.Linear(hidden_size, hidden_size)

    def forward(self, lstm_out, h_in):
        g = torch.sigmoid(self.gate(h_in))
        return g * lstm_out + (1 - g) * h_in
```

For multi-layer LSTM in `LSTMDecoderWithAttention`, apply after each layer's output during the manual unroll (needed when using `LayerNormLSTMCell` which replaces the monolithic `nn.LSTM`).

**Note:** Full integration of 1A+1B+1C requires replacing `nn.LSTM` with a manual layer-by-layer unroll using `LayerNormLSTMCell`. Implementation skeleton:

```python
# In LSTMDecoder.forward(), replace nn.LSTM call with:
h_list, c_list = list(zip(*hidden_states))   # separate per-layer
outputs = []
for t in range(seq_len):
    x_t = embedded[:, t, :]
    new_h_list, new_c_list = [], []
    for layer_idx, cell in enumerate(self.cells):   # self.cells = nn.ModuleList of LayerNormLSTMCell
        h_new, c_new = cell(x_t if layer_idx == 0 else h_out,
                            (h_list[layer_idx], c_list[layer_idx]))
        if layer_idx > 0:
            h_new = self.highways[layer_idx-1](h_new, x_t)   # highway
        h_out = self.dropout_layer(h_new)
        new_h_list.append(h_new); new_c_list.append(c_new)
    outputs.append(h_out)
```

---

## Tier 2 — Dense Co-Attention (DCAN)
**Status: [ ] TODO**
**Effort: 2–3 days**
**Dependencies: Tier 1 preferred first, but can be done independently**
**Files: `src/models/decoder_attention.py`**

### New class: DenseCoAttention
Add to `src/models/decoder_attention.py` (alongside BahdanauAttention):

```python
class DenseCoAttention(nn.Module):
    """
    Dense Co-Attention Network (DCAN).
    Replaces the two BahdanauAttention modules in LSTMDecoderWithAttention.

    Steps:
      1. Intra-modal self-attention on image regions (region↔region)
      2. Intra-modal self-attention on question words (word↔word)
      3. Inter-modal guided attention: image features guide question attention
      4. Inter-modal guided attention: question features guide image attention
    """
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads   = num_heads
        self.d_head      = hidden_size // num_heads

        # Intra-modal image self-attention projections
        self.img_q = nn.Linear(hidden_size, hidden_size)
        self.img_k = nn.Linear(hidden_size, hidden_size)
        self.img_v = nn.Linear(hidden_size, hidden_size)

        # Intra-modal question self-attention projections
        self.q_q  = nn.Linear(hidden_size, hidden_size)
        self.q_k  = nn.Linear(hidden_size, hidden_size)
        self.q_v  = nn.Linear(hidden_size, hidden_size)

        # Inter-modal: text → image (question guides image attention)
        self.cross_qi_q = nn.Linear(hidden_size, hidden_size)  # query from question
        self.cross_qi_k = nn.Linear(hidden_size, hidden_size)  # key from image
        self.cross_qi_v = nn.Linear(hidden_size, hidden_size)  # value from image

        # Inter-modal: image → text (image guides question attention)
        self.cross_iq_q = nn.Linear(hidden_size, hidden_size)  # query from image
        self.cross_iq_k = nn.Linear(hidden_size, hidden_size)  # key from question
        self.cross_iq_v = nn.Linear(hidden_size, hidden_size)  # value from question

        # Output projections
        self.img_out = nn.Linear(hidden_size, hidden_size)
        self.q_out   = nn.Linear(hidden_size, hidden_size)

        # Layer norms
        self.ln_img = nn.LayerNorm(hidden_size)
        self.ln_q   = nn.LayerNorm(hidden_size)

        self.scale = self.d_head ** -0.5

    def _attention(self, Q, K, V, mask=None):
        """Scaled dot-product attention. All inputs: (B, num_heads, seq, d_head)"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale   # (B, H, Sq, Sk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        alpha  = F.softmax(scores, dim=-1)
        out    = torch.matmul(alpha, V)                               # (B, H, Sq, d_head)
        return out, alpha

    def _reshape_heads(self, x):
        """(B, seq, H) → (B, num_heads, seq, d_head)"""
        B, S, _ = x.shape
        return x.view(B, S, self.num_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x):
        """(B, num_heads, seq, d_head) → (B, seq, H)"""
        B, _, S, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, self.hidden_size)

    def forward(self, img_feats, q_hidden, lstm_h, q_mask=None):
        """
        img_feats: (B, num_regions, hidden_size)  e.g. (B, 49, 1024)
        q_hidden:  (B, q_len, hidden_size)
        lstm_h:    (B, hidden_size)   — current LSTM state (for conditioning, optional)
        q_mask:    (B, q_len) bool mask for padded positions

        returns:
          img_context: (B, hidden_size)  — attended image representation
          q_context:   (B, hidden_size)  — attended question representation
          img_alpha:   (B, num_regions)  — image attention weights (for visualization)
          q_alpha:     (B, q_len)        — question attention weights (for visualization)
        """
        # --- Step 1: Intra-modal image self-attention ---
        iQ = self._reshape_heads(self.img_q(img_feats))
        iK = self._reshape_heads(self.img_k(img_feats))
        iV = self._reshape_heads(self.img_v(img_feats))
        img_self_out, _ = self._attention(iQ, iK, iV)
        img_self = self.ln_img(img_feats + self.img_out(self._merge_heads(img_self_out)))

        # --- Step 2: Intra-modal question self-attention ---
        qQ = self._reshape_heads(self.q_q(q_hidden))
        qK = self._reshape_heads(self.q_k(q_hidden))
        qV = self._reshape_heads(self.q_v(q_hidden))
        # Build mask for padding if provided
        if q_mask is not None:
            q_attn_mask = q_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,q_len)
        else:
            q_attn_mask = None
        q_self_out, _ = self._attention(qQ, qK, qV, mask=q_attn_mask)
        q_self = self.ln_q(q_hidden + self.q_out(self._merge_heads(q_self_out)))

        # --- Step 3: Inter-modal image→question (image guides question) ---
        # Query: mean-pooled image, Key/Value: question words
        img_mean = img_self.mean(dim=1, keepdim=True)  # (B, 1, H)
        cQ = self._reshape_heads(self.cross_iq_q(img_mean))   # (B, nh, 1, d_head)
        cK = self._reshape_heads(self.cross_iq_k(q_self))
        cV = self._reshape_heads(self.cross_iq_v(q_self))
        q_cross_out, q_alpha_h = self._attention(cQ, cK, cV, mask=q_attn_mask)
        q_context = self._merge_heads(q_cross_out).squeeze(1)  # (B, H)
        # q_alpha: average over heads, squeeze query dim
        q_alpha = q_alpha_h.mean(dim=1).squeeze(1)   # (B, q_len)

        # --- Step 4: Inter-modal question→image (question guides image) ---
        # Query: q_context, Key/Value: image regions
        q_ctx_expand = q_context.unsqueeze(1)   # (B, 1, H)
        dQ = self._reshape_heads(self.cross_qi_q(q_ctx_expand))
        dK = self._reshape_heads(self.cross_qi_k(img_self))
        dV = self._reshape_heads(self.cross_qi_v(img_self))
        img_cross_out, img_alpha_h = self._attention(dQ, dK, dV)
        img_context = self._merge_heads(img_cross_out).squeeze(1)  # (B, H)
        img_alpha = img_alpha_h.mean(dim=1).squeeze(1)  # (B, num_regions)

        return img_context, q_context, img_alpha, q_alpha
```

### Modify LSTMDecoderWithAttention to use DCAN
**File: `src/models/decoder_attention.py`**

In `LSTMDecoderWithAttention.__init__`, add parameter `use_dcan=False`:

```python
def __init__(self, vocab_size, embed_size, hidden_size, num_layers,
             attn_dim=512, dropout=0.5, pretrained_embeddings=None,
             use_coverage=False, use_dcan=False, num_heads=4):
    ...
    self.use_dcan = use_dcan
    if use_dcan:
        self.attention = DenseCoAttention(hidden_size, num_heads=num_heads)
        # img_attention and q_attention are no longer needed
    else:
        self.img_attention = BahdanauAttention(hidden_size, attn_dim, use_coverage)
        self.q_attention   = BahdanauAttention(hidden_size, attn_dim, use_coverage=False)
```

In `decode_step`, add branch:
```python
def decode_step(self, token, hidden, img_features, q_hidden_states, coverage=None, q_mask=None):
    ...
    if self.use_dcan:
        h_top = hidden[0][-1]   # top layer hidden: (B, H)
        img_context, q_context, img_alpha, q_alpha = \
            self.attention(img_features, q_hidden_states, h_top, q_mask)
        coverage_loss = torch.tensor(0.0)
    else:
        # existing BahdanauAttention code
        ...
    # LSTM input stays same shape: [embed; img_context; q_context] → embed_size + H*2
```

---

## Tier 3A — ConvNeXt-Base Spatial Encoder
**Status: [ ] TODO**
**Effort: 3–4 days**
**Dependencies: none**
**Files: `src/models/encoder_cnn.py`, `src/models/vqa_models.py`, `src/train.py`**

### New class: ConvNeXtSpatialEncoder
Add to `src/models/encoder_cnn.py`:

```python
class ConvNeXtSpatialEncoder(nn.Module):
    """
    ConvNeXt-Base spatial encoder.
    Pure CNN (no transformers). Rivals ViT accuracy.
    Output: (B, 49, output_size) — same shape as ResNetSpatialEncoder.

    ConvNeXt-Base channel progression: 128→256→512→1024
    Final feature map: (B, 1024, 7, 7) from stage 3 → reshape to (B, 49, 1024)
    No projection needed if output_size=1024 (ConvNeXt-Base already outputs 1024).
    """
    def __init__(self, output_size=1024, freeze=True):
        super().__init__()
        from torchvision.models import convnext_base, ConvNeXt_Base_Weights
        backbone = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        # features[0..7]: stem + 4 stages + layernorm
        # features[0]: stem (stride 4)  → (B, 128, 56, 56)
        # features[1]: stage 0 (3 blks) → (B, 128, 56, 56)
        # features[2]: downsample       → (B, 256, 28, 28)
        # features[3]: stage 1 (3 blks) → (B, 256, 28, 28)
        # features[4]: downsample       → (B, 512, 14, 14)
        # features[5]: stage 2 (27 blks)→ (B, 512, 14, 14)
        # features[6]: downsample       → (B, 1024, 7, 7)
        # features[7]: stage 3 (3 blks) → (B, 1024, 7, 7)  ← WE STOP HERE
        self.features = backbone.features   # all 8 feature blocks
        self.proj = nn.Identity() if output_size == 1024 else \
                    nn.Conv2d(1024, output_size, kernel_size=1)
        self.output_size = output_size

        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False
            if output_size != 1024:
                # proj is always trainable
                pass

    def forward(self, x):
        # x: (B, 3, 224, 224)
        feat = self.features(x)       # (B, 1024, 7, 7)
        feat = self.proj(feat)        # (B, output_size, 7, 7)
        B, C, H, W = feat.shape
        feat = feat.permute(0, 2, 3, 1).contiguous()   # (B, 7, 7, C)
        feat = feat.view(B, H * W, C)                  # (B, 49, output_size)
        return feat

    def unfreeze_top_layers(self):
        """Unfreeze last 2 stages: features[5] (stage 2) and features[6-7] (downsample + stage 3)."""
        for block in [self.features[5], self.features[6], self.features[7]]:
            for p in block.parameters():
                p.requires_grad = True
        if self.output_size != 1024:
            for p in self.proj.parameters():
                p.requires_grad = True

    def backbone_params(self):
        return [p for p in self.features.parameters() if p.requires_grad]
```

### New model: VQAModelE
Add to `src/models/vqa_models.py`:

```python
class VQAModelE(nn.Module):
    """
    Model E: ConvNeXt-Base + DCAN + MUTAN + LSTMDecoderWithAttention + PointerGenerator
    Build tiers in order: start with ConvNeXt+DCAN, add MUTAN (Tier4), add PGN (Tier5)
    """
    def __init__(self, vocab_size, answer_vocab_size, embed_size=512,
                 hidden_size=1024, num_layers=2, attn_dim=512,
                 freeze_cnn=True, dropout=0.5, use_coverage=False,
                 use_mutan=False, use_pgn=False,
                 pretrained_q_emb=None, pretrained_a_emb=None):
        super().__init__()
        self.model_type = 'E'

        self.encoder     = ConvNeXtSpatialEncoder(output_size=hidden_size, freeze=freeze_cnn)
        self.q_encoder   = QuestionEncoder(vocab_size, embed_size, hidden_size,
                                           num_layers, dropout, pretrained_q_emb)
        if use_mutan:
            self.fusion  = MUTANFusion(hidden_size, hidden_size, hidden_size)
        else:
            self.fusion  = GatedFusion(hidden_size)

        self.decoder     = LSTMDecoderWithAttention(
            answer_vocab_size, embed_size, hidden_size, num_layers,
            attn_dim=attn_dim, dropout=dropout,
            pretrained_embeddings=pretrained_a_emb,
            use_coverage=use_coverage,
            use_dcan=True,       # DCAN replaces Bahdanau
            use_pgn=use_pgn,
        )

    def forward(self, images, questions, target_seq):
        img_feats = self.encoder(images)              # (B, 49, H)
        q_feat, q_hidden = self.q_encoder(questions)  # (B,H), (B,L,H)
        # Global image feature for fusion: mean-pool over regions
        img_global = img_feats.mean(dim=1)            # (B, H)
        fused      = self.fusion(img_global, q_feat)  # (B, H)
        logits, cov_loss = self.decoder(fused, img_feats, q_hidden, target_seq)
        return logits, cov_loss

    def unfreeze_cnn(self):
        self.encoder.unfreeze_top_layers()

    def cnn_backbone_params(self):
        return self.encoder.backbone_params()
```

### train.py changes for Model E
Add to `get_model()` factory function:

```python
elif model_type == 'E':
    from models.encoder_cnn import ConvNeXtSpatialEncoder
    from models.vqa_models import VQAModelE
    model = VQAModelE(
        vocab_size=vocab_q_size,
        answer_vocab_size=vocab_a_size,
        embed_size=512,
        hidden_size=1024,
        num_layers=2,
        attn_dim=512,
        freeze_cnn=True,
        dropout=dropout,
        use_coverage=use_coverage,
        use_mutan=use_mutan,      # new arg
        use_pgn=use_pgn,          # new arg
        pretrained_q_emb=pretrained_q_emb,
        pretrained_a_emb=pretrained_a_emb,
    )
```

Add new args to argparse:
```python
parser.add_argument('--model', type=str, default='A', choices=['A','B','C','D','E','F'])
parser.add_argument('--use_mutan', action='store_true')
parser.add_argument('--use_pgn', action='store_true')
```

---

## Tier 4 — MUTAN Tucker Fusion
**Status: [ ] TODO**
**Effort: 1–2 days**
**Dependencies: none (can test standalone)**
**Files: `src/models/vqa_models.py`**

Add to `src/models/vqa_models.py`:

```python
class MUTANFusion(nn.Module):
    """
    Multimodal Tucker Fusion (MUTAN).
    Replaces GatedFusion for Model E/F.
    Captures multiplicative cross-modal interactions via Tucker decomposition.

    Formula: y = T_c ×₁ (q W_q) ×₂ (v W_v)  then Linear → output
    where T_c ∈ R^{t_q × t_v × d_out} is learnable core tensor.
    """
    def __init__(self, d_q, d_v, d_out, t_q=360, t_v=360):
        super().__init__()
        self.t_q   = t_q
        self.t_v   = t_v
        self.d_out = d_out

        self.W_q   = nn.Linear(d_q, t_q, bias=False)
        self.W_v   = nn.Linear(d_v, t_v, bias=False)
        # Core tensor: learnable, initialized small to avoid exploding values
        self.T_c   = nn.Parameter(0.01 * torch.randn(t_q, t_v, d_out))
        self.bn    = nn.BatchNorm1d(d_out)
        self.drop  = nn.Dropout(0.5)

    def forward(self, q, v):
        """
        q: (B, d_q) — question feature (from QuestionEncoder)
        v: (B, d_v) — image feature (global, from mean-pool or direct)
        returns: (B, d_out)
        """
        q_proj = self.drop(torch.tanh(self.W_q(q)))   # (B, t_q)
        v_proj = self.drop(torch.tanh(self.W_v(v)))   # (B, t_v)
        # Tucker product: einsum over batch
        # q_proj: (B, t_q), T_c: (t_q, t_v, d_out)
        # intermediate: (B, t_v, d_out)
        inter = torch.einsum('bi,ijk->bjk', q_proj, self.T_c)  # (B, t_v, d_out)
        # Contract with v_proj: (B, t_v) × (B, t_v, d_out) → (B, d_out)
        out = torch.einsum('bj,bjk->bk', v_proj, inter)         # (B, d_out)
        out = self.bn(out)
        return out
```

**Usage in VQAModelE**: pass `use_mutan=True` to constructor.

---

## Tier 5 — Pointer-Generator Network
**Status: [ ] TODO**
**Effort: 3–4 days**
**Dependencies: Tier 2 (DCAN provides q_alpha needed for copy distribution)**
**Files: `src/models/pointer_generator.py` (new), `src/models/decoder_attention.py`**

### New file: `src/models/pointer_generator.py`

```python
"""
Pointer-Generator head for LSTMDecoderWithAttention.

At each decode step:
  p_gen = sigmoid(W_c * ctx + W_h * h_t + W_x * x_t)   # generate vs copy
  P_vocab = softmax(W_out * h_t)                         # vocabulary dist
  P_copy  = q_attention_weights over q_token_ids         # copy dist
  P_final = p_gen * P_vocab + (1-p_gen) * P_copy         # mixed dist
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerGeneratorHead(nn.Module):
    """
    Mixes vocabulary generation with copying from question token sequence.
    Added to LSTMDecoderWithAttention via use_pgn=True.
    """
    def __init__(self, hidden_size, embed_size):
        super().__init__()
        # p_gen computation: context(H) + h_t(H) + x_t(embed_size) → scalar
        self.W_c  = nn.Linear(hidden_size, 1, bias=False)
        self.W_h  = nn.Linear(hidden_size, 1, bias=False)
        self.W_x  = nn.Linear(embed_size,  1, bias=True)

    def forward(self, context, h_t, x_t, P_vocab, q_alpha, q_token_ids, vocab_size):
        """
        context:     (B, hidden_size) — weighted image context
        h_t:         (B, hidden_size) — LSTM top hidden state
        x_t:         (B, embed_size)  — input embedding at step t
        P_vocab:     (B, vocab_size)  — softmax over vocabulary
        q_alpha:     (B, q_len)       — attention weights over question tokens
        q_token_ids: (B, q_len)       — question token indices (from input)
        vocab_size:  int

        returns: P_final (B, vocab_size)
        """
        # 1. Compute generation probability
        p_gen = torch.sigmoid(self.W_c(context) + self.W_h(h_t) + self.W_x(x_t))  # (B,1)

        # 2. Build copy distribution: scatter q_alpha onto vocab positions
        P_copy = torch.zeros_like(P_vocab)   # (B, vocab_size)
        P_copy.scatter_add_(1, q_token_ids, q_alpha)

        # 3. Mix
        P_final = p_gen * P_vocab + (1.0 - p_gen) * P_copy
        return P_final
```

### Modify LSTMDecoderWithAttention
**File: `src/models/decoder_attention.py`**

In `__init__`, add `use_pgn=False`:
```python
if use_pgn:
    from models.pointer_generator import PointerGeneratorHead
    self.pgn = PointerGeneratorHead(hidden_size, embed_size)
self.use_pgn = use_pgn
```

In `decode_step`, after computing logit:
```python
if self.use_pgn and q_token_ids is not None:
    P_vocab = F.softmax(logit, dim=-1)        # (B, vocab_size)
    logit   = self.pgn(
        context=img_context,                  # image context (B, H)
        h_t=hidden[0][-1],                    # top hidden state
        x_t=embedded_t,                       # embedding at this step (B, embed_size)
        P_vocab=P_vocab,
        q_alpha=q_alpha,
        q_token_ids=q_token_ids,
        vocab_size=self.vocab_size,
    )
    # logit is now P_final, a probability distribution — take log for loss
    logit = torch.log(logit + 1e-9)
```

In `forward`, pass `q_token_ids` (the raw question token indices) to `decode_step`.

---

## Tier 6 — Abstention-Aware CSS
**Status: [ ] TODO**
**Effort: 2–3 days**
**Dependencies: Tier 3B (BUTD) for visual masking; Tier 2 (DCAN) for linguistic masking**
**Files: `src/training/css_augment.py` (new), `src/train.py`, `src/scripts/1_build_vocab.py`**

### Step 0: Add abstention to vocabulary
**File: `src/scripts/1_build_vocab.py`**

After building answer vocab, force-add abstention sentence:
```python
ABSTENTION = "i cannot answer because the object is hidden"
for word in vocab_a.tokenize(ABSTENTION):
    if word not in vocab_a.word2idx:
        vocab_a.add_word(word)
```

### New file: `src/training/css_augment.py`

```python
"""
Abstention-Aware Counterfactual Samples Synthesizing (CSS).
Gemini v2-update recommendation.

For each batch:
  1. Visual masking: zero out img_feats for critical object region.
     Target sequence → abstention sentence.
  2. Linguistic masking: replace key question nouns/verbs with <mask> token.
     Target sequence → generic abstention.

Returns augmented batch with 50% original + 50% counterfactual samples.
"""

import torch
import random


ABSTENTION_TEXT = "i cannot answer because the object is hidden"
LINGUISTIC_ABSTENTION = "i cannot answer because the question is unclear"


class AbstentionCSSAugmenter:
    def __init__(self, vocab_a, mask_token_id=3, css_ratio=0.5):
        """
        vocab_a:       Vocabulary object for answers
        mask_token_id: token ID to use for <mask> (default 3 = <unk>)
        css_ratio:     fraction of batch to replace with counterfactuals
        """
        self.vocab_a      = vocab_a
        self.mask_id      = mask_token_id
        self.css_ratio    = css_ratio
        # Pre-compute abstention token sequences
        self.abstention_ids = torch.tensor(
            vocab_a.numericalize(ABSTENTION_TEXT), dtype=torch.long)
        self.ling_abstention_ids = torch.tensor(
            vocab_a.numericalize(LINGUISTIC_ABSTENTION), dtype=torch.long)

    def visual_mask(self, img_feats, img_alpha):
        """
        img_feats: (B, num_regions, H)
        img_alpha: (B, num_regions) — attention weights indicating critical regions
        Returns: masked_feats (B, num_regions, H)
        """
        # Zero out top-3 most-attended regions (critical object)
        masked = img_feats.clone()
        _, top_idx = img_alpha.topk(3, dim=-1)   # (B, 3)
        for b in range(img_feats.size(0)):
            masked[b, top_idx[b]] = 0.0
        return masked

    def linguistic_mask(self, q_tokens, question_texts, vocab_q):
        """
        q_tokens: (B, q_len) — question token indices
        Masks nouns and verbs (heuristic: content words not in stopwords)
        Returns masked q_tokens.
        """
        STOPWORDS = {'is', 'the', 'a', 'an', 'are', 'what', 'which', 'how',
                     'who', 'where', 'when', 'does', 'do', 'this', 'that',
                     'of', 'in', 'on', 'at', 'to', 'and', 'or', 'there'}
        masked = q_tokens.clone()
        for b, text in enumerate(question_texts):
            words = text.lower().split()
            for t_idx, word in enumerate(words):
                if word not in STOPWORDS and t_idx < q_tokens.size(1):
                    if random.random() < 0.5:   # mask 50% of content words
                        masked[b, t_idx] = self.mask_id
        return masked

    def augment_batch(self, imgs, questions, answers, img_feats=None,
                      img_alpha=None, question_texts=None, vocab_q=None):
        """
        Creates counterfactual variants for css_ratio of the batch.
        Returns augmented (imgs, questions, answers) with double batch size.
        """
        B = imgs.size(0)
        n_css = max(1, int(B * self.css_ratio))
        css_idx = random.sample(range(B), n_css)

        cf_imgs      = imgs[css_idx].clone()
        cf_questions = questions[css_idx].clone()
        cf_answers   = []

        for b in css_idx:
            if img_feats is not None and img_alpha is not None:
                # Visual counterfactual — use abstention target
                cf_answers.append(self.abstention_ids)
            else:
                # Linguistic counterfactual — use linguistic abstention target
                cf_answers.append(self.ling_abstention_ids)

        # Pad cf_answers to same length
        from torch.nn.utils.rnn import pad_sequence
        cf_answers_padded = pad_sequence(cf_answers, batch_first=True)

        # Stack original + counterfactual
        aug_imgs      = torch.cat([imgs, cf_imgs], dim=0)
        aug_questions = torch.cat([questions, cf_questions], dim=0)
        # Pad answers to same length
        max_len = max(answers.size(1), cf_answers_padded.size(1))
        answers_padded    = torch.zeros(B, max_len, dtype=torch.long)
        answers_padded[:, :answers.size(1)] = answers
        cf_ans_padded     = torch.zeros(n_css, max_len, dtype=torch.long)
        cf_ans_padded[:, :cf_answers_padded.size(1)] = cf_answers_padded
        aug_answers       = torch.cat([answers_padded, cf_ans_padded], dim=0)

        return aug_imgs, aug_questions, aug_answers
```

### train.py changes
Add to argparse:
```python
parser.add_argument('--css', action='store_true', help='Enable Abstention-Aware CSS augmentation')
parser.add_argument('--css_ratio', type=float, default=0.5)
```

In training loop (after batch load):
```python
if args.css and css_augmenter is not None:
    imgs, questions, answers = css_augmenter.augment_batch(
        imgs, questions, answers,
        question_texts=q_texts_raw   # need to pass raw text alongside tensors
    )
```

---

## Tier 7 — Deep BiLSTM + GloVe 840B + Char-CNN Question Encoder
**Status: [ ] TODO**
**Effort: 3–5 days**
**Dependencies: none**
**Files: `src/models/encoder_question.py`**

### 7A — GloVe 840B initialization
Already supported via `pretrained_embeddings` parameter in `QuestionEncoder.__init__`.
Just load and pass:

```python
# In train.py, after loading vocab:
if args.glove:
    glove_path = "data/raw/glove/glove.840B.300d.txt"
    pretrained_q_emb = load_glove(glove_path, vocab_q, args.glove_dim)
    # load_glove already exists in codebase for answer vocab
    # Ensure it's used for question vocab too
```

### 7B — Deeper BiLSTM (num_layers=3)
`QuestionEncoder` already supports `num_layers` param. Just set:
```python
model = VQAModelE(..., num_layers=3, ...)  # or pass to QuestionEncoder directly
```
Add highway connections between BiLSTM layers using `HighwayLayer` from Tier 1.

### 7C — Char-CNN (optional, OOV handling)
Add to `src/models/encoder_question.py`:

```python
class CharCNNEmbedding(nn.Module):
    """
    Character-level CNN embedding.
    Handles OOV words (proper nouns, rare words) that GloVe misses.
    Prepended to word embeddings before QuestionEncoder BiLSTM.
    """
    MAX_WORD_LEN = 20
    ALPHABET_SIZE = 70  # printable ASCII

    def __init__(self, embed_dim=50, num_filters=100, kernel_sizes=(3, 4, 5)):
        super().__init__()
        self.char_embed = nn.Embedding(self.ALPHABET_SIZE + 1, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
        self.output_size = num_filters * len(kernel_sizes)   # 300

    def forward(self, char_ids):
        """
        char_ids: (B, seq_len, MAX_WORD_LEN)
        returns:  (B, seq_len, output_size)
        """
        B, S, L = char_ids.shape
        char_ids = char_ids.view(B * S, L)
        emb = self.char_embed(char_ids).transpose(1, 2)   # (B*S, embed_dim, L)
        feats = [F.max_pool1d(F.relu(conv(emb)), emb.size(-1)).squeeze(-1)
                 for conv in self.convs]
        out = torch.cat(feats, dim=-1)   # (B*S, output_size)
        return out.view(B, S, -1)
```

Modify `QuestionEncoder.__init__` to accept `use_char_cnn=False`:
```python
if use_char_cnn:
    self.char_cnn = CharCNNEmbedding(embed_dim=50, num_filters=100, kernel_sizes=(3,4,5))
    bilstm_input_size = embed_size + 300  # GloVe + char features
else:
    bilstm_input_size = embed_size
self.bilstm = nn.LSTM(bilstm_input_size, hidden_size//2, num_layers=num_layers,
                      bidirectional=True, batch_first=True, dropout=dropout)
```

---

## Tier 8 — SCST Reinforcement Learning
**Status: [ ] TODO**
**Effort: 2–3 days**
**Dependencies: Best model (D or E) trained through Phase 3**
**Files: `src/train.py`, `src/training/scst.py` (new)**

### New file: `src/training/scst.py`

```python
"""
Self-Critical Sequence Training (SCST).
REINFORCE with greedy baseline.
Reward: BLEU-4 (simple) or CIDEr-D (better, requires IDF corpus stats).
"""
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def compute_bleu4_reward(hypotheses, references):
    """
    hypotheses: list[str]  — sampled sequences
    references: list[str]  — ground truth sequences
    returns: tensor(B,) of BLEU-4 scores
    """
    smooth = SmoothingFunction().method1
    rewards = []
    for hyp, ref in zip(hypotheses, references):
        hyp_tokens = hyp.split()
        ref_tokens = [ref.split()]
        score = sentence_bleu(ref_tokens, hyp_tokens,
                              weights=(0.25,0.25,0.25,0.25),
                              smoothing_function=smooth)
        rewards.append(score)
    return torch.tensor(rewards, dtype=torch.float32)


def scst_loss(model, imgs, questions, target_texts, vocab_a,
              max_len=50, device='cpu'):
    """
    Compute SCST policy gradient loss.

    1. Greedy decode  → baseline reward r_greedy
    2. Sample decode  → sample reward r_sample
    3. REINFORCE: loss = -(r_sample - r_greedy) * log P(sampled sequence)
    """
    from inference import batch_greedy_decode_with_attention

    # Greedy baseline (no gradient)
    with torch.no_grad():
        greedy_texts = batch_greedy_decode_with_attention(
            model, imgs, questions, vocab_a, max_len=max_len, device=device)

    # Sample decode (with gradient through log_probs)
    model.train()
    # ... sample from model distribution, record log probabilities ...
    # (full implementation: forward pass with temperature sampling,
    #  accumulate log probs per step, stop at <end>)

    r_greedy = compute_bleu4_reward(greedy_texts, target_texts)
    r_sample = compute_bleu4_reward(sampled_texts, target_texts)
    advantage = (r_sample - r_greedy).to(device)

    # Policy gradient loss: negate because we maximize reward
    loss = -(advantage.unsqueeze(1) * log_probs).mean()
    return loss
```

### train.py changes
Add to argparse:
```python
parser.add_argument('--scst', action='store_true', help='Phase 4: SCST RL training')
parser.add_argument('--scst_reward', type=str, default='bleu4', choices=['bleu4','cider'])
```

Phase 4 loop (after Phase 3 converges):
```python
if args.scst:
    for epoch in range(scst_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            loss = scst_loss(model, imgs, questions, target_texts, vocab_a, device=device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
```

---

## Tier 3B — Faster R-CNN BUTD Encoder (Model F)
**Status: [ ] TODO**
**Effort: 1–2 weeks**
**Dependencies: Tier 2 (DCAN), Tier 4 (MUTAN), Tier 5 (PGN) should be done first (test on E)**
**Files: `src/models/encoder_cnn.py`, `src/models/vqa_models.py`, `src/dataset.py`**

### New class: FasterRCNNEncoder

```python
class FasterRCNNEncoder(nn.Module):
    """
    Bottom-Up Top-Down (BUTD) object-level features.
    Uses Faster R-CNN to extract k=36 region proposals per image.
    Output: (B, k, output_size)

    Includes spatial coordinates appended to feature vectors.
    """
    def __init__(self, output_size=1024, k=36, freeze=True):
        super().__init__()
        from torchvision.models.detection import fasterrcnn_resnet50_fpn, \
            FasterRCNN_ResNet50_FPN_Weights
        self.k = k
        self.output_size = output_size

        detector = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        # Extract backbone + RoI head — remove box predictor
        self.backbone = detector.backbone
        self.roi_pool = detector.roi_heads.box_roi_pool
        # RoI feature dim: 256 * 7 * 7 = 12544 for FPN, or use AdaptiveAvgPool
        self.roi_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),    # (B*k, 256, 1, 1) → flatten
            nn.Flatten(),
            nn.Linear(256, output_size),
            nn.ReLU(),
        )
        # Spatial embedding: 5 coords [x1/W, y1/H, x2/W, y2/H, area/(W*H)]
        self.spatial_proj = nn.Linear(5, output_size)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, images):
        """images: (B, 3, H, W) — normalized ImageNet tensors
        returns: (B, k, output_size)
        """
        # This requires torchvision detection pipeline — run in eval mode
        # Pre-extraction recommended: cache features to disk for speed
        raise NotImplementedError(
            "FasterRCNNEncoder.forward: Use pre-extracted features (see Tier 3B notes)"
        )

    def extract_and_cache(self, image_dir, output_dir, split='train2014'):
        """Pre-extract BUTD features to disk. Run once before training."""
        # Loop over images, extract top-k RoI features + boxes
        # Save as .pt files: {'feats': (k, 2048), 'boxes': (k, 4), 'img_id': int}
        pass
```

**Critical note:** Faster R-CNN inference during training is too slow (~0.5s/image). **Pre-extract features to disk** and create a `BUTDDataset` that loads cached `.pt` files instead of raw images.

### Collate function for variable k
`vqa_collate_fn` needs updating for variable-length region sequences:
```python
def vqa_collate_fn_butd(batch):
    imgs, questions, answers, img_feats = zip(*batch)
    questions_padded = pad_sequence(questions, batch_first=True)
    answers_padded   = pad_sequence(answers, batch_first=True)
    img_feats_padded = pad_sequence(img_feats, batch_first=True)  # (B, max_k, H)
    imgs_stacked     = torch.stack(imgs, dim=0)
    return imgs_stacked, questions_padded, answers_padded, img_feats_padded
```

---

## Tier 9 — ConceptNet + GNN (Optional)
**Status: [ ] TODO (advanced, after E/F are working)**
**Effort: 2–3 weeks**
**Dependencies: Tier 3B (BUTD) for entity extraction**

High-level steps:
1. Install `torch_geometric`
2. Download ConceptNet as SQLite or JSON (~6GB)
3. New `ConceptNetRetriever`: given detected object labels → fetch 2-hop subgraph
4. New `KnowledgeGNN(nn.Module)`: 2-layer GraphSAGE → knowledge embedding (B, H)
5. New `AdaptiveScoreAttention`: weights visual vs knowledge based on question type
6. Integrate into `VQAModelF.forward()` as optional `knowledge_context`

---

---

# PART II — DATA TIERS

---

## Data Tier D2 — Dataset Expansion
**Status: [ ] TODO**
**Effort: 1–3 days setup**
**Dependencies: none**

### D2A — VQA v2.0 Full Training Set
- Download: `v2_Questions_Train_mscoco.zip` + `v2_Annotations_Train_mscoco.zip`
- ~443K QA pairs (vs ~183K VQA-E) — same COCO images, no new download
- **Strategy:** Pre-train Phase 1 on VQA v2.0, fine-tune Phase 2–3 on VQA-E
- `VQADataset` (already in dataset.py) loads this format

### D2B — Visual Genome Pre-training
- Download: `region_descriptions.json` (~600MB), images (~15GB)
- ~5.4M region descriptions + scene graphs
- **Convert to QA pairs:**
  ```python
  # region: {"region_id": X, "image_id": Y, "phrase": "A brown dog sitting on a mat"}
  # → Q: "Describe what you see in this region."
  #   A: "A brown dog sitting on a mat"
  # OR template: subject-verb-object extraction via spaCy
  ```
- **Pre-training config:** 5 epochs, lr=1e-3, no scheduled sampling, just CE loss

### D2C — GQA
- Download: `train_all_questions.json` (~21MB) — uses Visual Genome images
- 22M compositional questions, scene-graph-grounded, bias-free
- Mix into Phase 2 at 25% ratio alongside VQA-E

---

## Data Tier D3 — Answer Distribution Balancing
**Status: [ ] TODO**
**Effort: 1 day**
**Files: `src/train.py`**

### D3A — Inverse Frequency Weighting
```python
# After loading vocab, compute answer frequencies from training set:
from collections import Counter
answer_counts = Counter()
for ann in train_annotations:
    answer_counts[ann['multiple_choice_answer']] += 1
# Build weight tensor:
weights = torch.ones(len(vocab_a))
for word, idx in vocab_a.word2idx.items():
    if word in answer_counts:
        weights[idx] = 1.0 / (answer_counts[word] ** 0.5)
criterion = nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=0,
                                label_smoothing=args.label_smoothing)
```

### D3B — Soft Label Training
```python
# In VQADataset, change target from one-hot to soft distribution:
answers = self.qid2answers.get(q_id, [""])
# Build soft target: count occurrences among 10 annotations
from collections import Counter
cnt = Counter(answers)
soft = torch.zeros(len(self.vocab_a))
for ans, count in cnt.items():
    idx = self.vocab_a.word2idx.get(ans, 3)  # <unk>
    soft[idx] = count / len(answers)
# Use KLDivLoss instead of CrossEntropyLoss for soft targets
```

### D3C — Answer-Type Balanced Sampling
VQA annotations include `answer_type` field: `yes/no`, `number`, `other`.
```python
# In VQADataset, record answer_type per sample
# In train.py, use WeightedRandomSampler:
type_counts = Counter(answer_types)
sample_weights = [1.0 / type_counts[t] for t in answer_types]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights))
train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
```

---

## Data Tier D4 — Task Progressive Curriculum Learning (TPCL)
**Status: [ ] TODO**
**Effort: 1–2 days**
**Files: `src/training/curriculum.py` (new), `src/train.py`**

### New file: `src/training/curriculum.py`

```python
"""
Task Progressive Curriculum Learning (TPCL).
Sorts training samples by answer sequence complexity.
LSTMs converge better when easy samples come first.
"""
import torch
from torch.utils.data import Sampler


def compute_complexity_scores(annotations):
    """
    Returns list of complexity scores for each sample.
    Score = sequence length + syntactic depth (approximated by clause count).
    """
    scores = []
    for ann in annotations:
        answer = ann.get('multiple_choice_answer', '')
        explanation = ann.get('explanation', [''])[0] if ann.get('explanation') else ''
        full_text = f"{answer} because {explanation}" if explanation else answer
        # Complexity: word count + clause indicators
        words = full_text.split()
        clauses = sum(1 for w in words if w.lower() in
                      {'because', 'since', 'although', 'however', 'therefore',
                       'which', 'that', 'who', 'while', 'when', 'where'})
        scores.append(len(words) + clauses * 3)
    return scores


class CurriculumSampler(Sampler):
    """
    Returns indices in curriculum order.
    Stage 1: short answers (len ≤ 3)
    Stage 2: medium (4 ≤ len ≤ 8)
    Stage 3: long + complex (len > 8)

    Uses pacing function: 20% stage1 → 30% stage2 → 50% stage3
    """
    def __init__(self, complexity_scores, current_epoch, total_epochs):
        self.scores     = complexity_scores
        self.epoch      = current_epoch
        self.total      = total_epochs
        self._build_ordered_indices()

    def _build_ordered_indices(self):
        scored = sorted(enumerate(self.scores), key=lambda x: x[1])
        self.ordered_indices = [i for i, _ in scored]
        n = len(self.ordered_indices)
        # Stage boundaries
        self.s1 = int(0.20 * n)
        self.s2 = int(0.50 * n)
        # self.s3 = n

    def __iter__(self):
        progress = self.epoch / max(self.total, 1)
        if progress < 0.2:
            pool = self.ordered_indices[:self.s1]
        elif progress < 0.5:
            pool = self.ordered_indices[:self.s2]
        else:
            pool = self.ordered_indices   # all samples
        import random
        indices = pool.copy()
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.ordered_indices)
```

---

## Data Tier D5 — LLM Synthetic Data + Hallucination Noise Filter
**Status: [ ] TODO**
**Effort: 2–3 days**
**Files: `src/scripts/generate_synthetic_qa.py` (new), `src/scripts/filter_hallucinations.py` (new)**

### Hallucination Noise Filter (apply to ALL training data)

```python
# src/scripts/filter_hallucinations.py
"""
Discard training samples where answer references objects
not detected by Faster R-CNN in the image.
Removes ~8-12% noisy human annotations.
"""
import json
import spacy
import torch

nlp = spacy.load("en_core_web_sm")

COMMON_WORDS = {'yes', 'no', 'not', 'i', 'the', 'a', 'an', 'it', 'is',
                'are', 'was', 'be', 'to', 'of', 'in', 'on', 'at', 'and',
                'or', 'there', 'because', 'this', 'that', 'some', 'many',
                'can', 'cannot', 'answer', 'object', 'hidden', 'unclear'}

def get_answer_nouns(text):
    doc = nlp(text)
    return {token.lemma_.lower() for token in doc
            if token.pos_ in ('NOUN', 'PROPN') and
            token.lemma_.lower() not in COMMON_WORDS}

def filter_dataset(annotations, faster_rcnn_labels_dict):
    """
    annotations: list of VQA-E annotation dicts
    faster_rcnn_labels_dict: {img_id: set_of_detected_labels}
    Returns filtered list.
    """
    kept, dropped = [], 0
    for ann in annotations:
        img_id = ann['img_id']
        answer = ann.get('multiple_choice_answer', '')
        exp    = ann.get('explanation', [''])[0] if ann.get('explanation') else ''
        full   = f"{answer} {exp}"
        nouns  = get_answer_nouns(full)
        detected = faster_rcnn_labels_dict.get(img_id, set())
        # If no nouns or all nouns in detected labels → keep
        if not nouns or nouns.issubset(detected | COMMON_WORDS):
            kept.append(ann)
        else:
            dropped += 1
    print(f"Kept {len(kept)}, dropped {dropped} ({100*dropped/(len(kept)+dropped):.1f}%)")
    return kept
```

---

---

# TRAINING SCHEDULE (Full Pipeline)

## Training Command Reference

```bash
# Phase 0: Visual Genome pre-training (after D2B setup)
python src/train.py --model E --epochs 5 --lr 1e-3 --batch_size 128 \
    --train_json data/processed/visual_genome_qa.json \
    --augment --weight_decay 1e-5

# Phase 1: VQA v2.0 full (2.4× data)
python src/train.py --model E --epochs 10 --lr 1e-3 --batch_size 128 \
    --train_json data/raw/vqa_v2/v2_train.json \
    --resume checkpoints/model_e_vg_pretrain.pth \
    --augment --weight_decay 1e-5 --curriculum --early_stopping 3

# Phase 2: VQA-E fine-tune
python src/train.py --model E --epochs 10 --lr 5e-4 --batch_size 128 \
    --resume checkpoints/model_e_phase1.pth \
    --finetune_cnn --cnn_lr_factor 0.01 \
    --augment --coverage --use_mutan --css --early_stopping 3

# Phase 3: Scheduled sampling
python src/train.py --model E --epochs 5 --lr 2e-4 --batch_size 128 \
    --resume checkpoints/model_e_phase2.pth \
    --scheduled_sampling --ss_k 5 --coverage --use_mutan --css

# Phase 4: SCST RL
python src/train.py --model E --epochs 5 --lr 5e-5 --batch_size 64 \
    --resume checkpoints/model_e_phase3.pth \
    --scst --scst_reward bleu4

# Evaluation
python src/evaluate.py --model_type E --checkpoint checkpoints/model_e_best.pth \
    --beam_width 5 --no_repeat_ngram 3
```

## Expected BLEU-4 Trajectory

| Milestone | Expected BLEU-4 | Notes |
|---|---|---|
| Baseline Model D | 11.59% | Current best |
| + Tier D1 (bug fixes) | ~12.5% | Flip guard alone worth ~0.5% |
| + Tier 0 (GloVe + beam fix) | ~13.0% | |
| + Tier D2A (VQA v2 full) | ~14.0% | 2.4× data |
| + Tier D2B (VG pre-train) | ~15.5% | Spatial language understanding |
| + Tier 1 (LSTM fortification) | ~16.0% | Better regularization |
| + Tier 2 (DCAN) | ~17.0% | Richer attention |
| + Tier 3A (ConvNeXt = Model E) | ~18.5% | Better visual features |
| + Tier 4 (MUTAN) | ~19.5% | Richer fusion |
| + Tier 5 (PGN) | ~21.0% | Eliminates OOV |
| + Tier 6 (CSS) | ~22.0% | Robustness to bias |
| + Tier 7 (deep BiLSTM) | ~23.0% | Better question encoding |
| + Tier 8 (SCST) | ~25.0% | Direct metric optimization |
| + Tier 3B (BUTD = Model F) | ~28–32% | Object-level features |
| + D4+D5 (curriculum + filter) | ~30–35% | Ceiling estimate |

---

## Quick Reference: Source → Tier Mapping

| Source | Contributions |
|---|---|
| **My analysis** | Tier 0, Tier 3A (ConvNeXt), Tier 7, D1 (bug fixes), D3 (answer balancing) |
| **Gemini v1** | Tier 1, Tier 2 (DCAN), Tier 3B (BUTD), Tier 4 (MUTAN), Tier 5 (PGN), Tier 6, Tier 8, Tier 9 |
| **Gemini v2-update** | D2B (VG pre-training), D2C (GQA), D4 (TPCL), Tier 6 upgrade (abstention-aware), D5 (hallucination filter) |
| **Codebase (existing)** | GloVe support, coverage mechanism, scheduled sampling, beam search, all 10 annotations |
