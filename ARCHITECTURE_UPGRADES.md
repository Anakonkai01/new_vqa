# VQA Architecture Upgrades — Analysis & Roadmap

## 1. Current Flagship: Model E

```
Image    → CLIP ViT-B/32 → 49 patch features (B, 49, 768) → Linear → (B, 49, 1024)
Question → BiLSTM (trained from scratch, 2 layers × 512-dim) → q_feat (B, 1024)
Fusion   → FiLMFusion: [γ, β] = MLP(q_feat), output = γ ⊙ LayerNorm(img) + β
Init     → h_0 = init_h_proj(FiLM(img_mean, q)), c_0 = init_c_proj(...)
Decoder  → LSTM (2-layer, 1024-hidden)
           + single-head Bahdanau attention → image context (B, 1024)
           + single-head Bahdanau attention → question context (B, 1024)
Output   → Linear(1024 → vocab_size)
```

---

## 2. Three Identified Weaknesses

### Weakness 1 — BiLSTM Question Encoder (Vanishing Gradients)

The BiLSTM hidden state update:

```
h_t = tanh(W_h · h_{t-1} + W_x · x_t + b)
```

The gradient of the loss w.r.t. the token at position 1 in a T-word question:

```
∂L/∂x_1 = ∂L/∂h_T · ∏_{t=2}^{T} ∂h_t/∂h_{t-1}
```

For T=20 (typical VQA-E question length), this is a product of **19 Jacobian matrices**.
Because `tanh` squashes activations into `(-1, 1)`, the spectral radius of each Jacobian
is typically < 1, so the product decays **exponentially** toward zero. Early question words
("what", "how many", "is there") have near-zero gradient influence on the loss.

The global feature `q_feat = concat(h_fwd[-1], h_bwd[-1])` compresses the entire
question into 1024 dimensions — a hard information bottleneck for long explanatory answers.

**Additionally**: CLIP's vision encoder was pre-trained contrastively on 400M image-text
pairs so that `img_feat · text_feat` is already a similarity score. The BiLSTM has never
seen an image — it learns language from VQA-E supervision alone (~200K samples). Using
CLIP's text encoder puts image and question features in the **same pre-aligned space**.

---

### Weakness 2 — Single-Head Bahdanau Attention (Representational Bottleneck)

Current attention at decode step t:

```
energy_i = v · tanh(W_h · h_t + W_img · img_i)    # scalar per region
alpha    = softmax(energy)                           # (49,)  — single distribution
context  = Σ_i alpha_i · img_i                      # (1024,) — one context vector
```

A single attention head produces **one context vector per step**. It can express only
one "query intent" at a time — e.g., "where is the main object?". It cannot
simultaneously ask "what color?" and "where is it?" and "is this yes/no?".

**Multi-Head Attention** (Vaswani et al., 2017) runs `h` independent sub-spaces:

```
head_i  = Attention(h_t · W_i^Q,  img · W_i^K,  img · W_i^V)
        = softmax( (h_t · W_i^Q)(img · W_i^K)^T / √d_k ) · (img · W_i^V)

output  = concat(head_1, ..., head_h) · W_O          # (1024,)
```

where `d_k = hidden_size / num_heads = 1024 / 8 = 128`.

The **`1/√d_k` scaling** is critical: without it, as `d_k` grows, the dot products
`QK^T` grow in magnitude as `O(√d_k)`, pushing softmax into near-zero gradient
saturation regions. Dividing by `√d_k` keeps the dot-product variance at 1.0
regardless of dimension, preventing gradient collapse.

---

### Weakness 3 — LSTM Decoder (Sequential Bottleneck)

At each decode step:

```
h_t = LSTM(h_{t-1}, [embed_t ; ctx_img ; ctx_q])
```

The entire context of all previously generated tokens must fit inside a 1024-d vector
`h_{t-1}`. For VQA-E explanations of 15–25 tokens, this vector must simultaneously
encode: all token history, grammatical state, and semantic intent. It is a **hard
compression** with no direct gradient path between distant tokens.

**Transformer Decoder** eliminates this by keeping all previous token representations
explicit and attending over them with masked self-attention:

```
# At position t, directly attends over ALL previous positions
self_attn_t   = MaskedMHA(y_t, [y_1, ..., y_{t-1}])          # O(1) gradient path
cross_img_t   = MHA(y_t, img_patches, img_patches)             # 49 visual regions
cross_q_t     = MHA(y_t, q_tokens, q_tokens)                  # 77 language tokens
y_t_out       = FFN(LayerNorm(self_attn_t + cross_img_t + cross_q_t))
```

During **training**, all T positions are processed in **parallel** (via causal mask) —
`O(T)` becomes `O(1)` in gradient path depth.

For **SCST/RL**: Transformer logits are typically better-calibrated (sharper post-training
distributions), reducing REINFORCE variance. The policy entropy `H(π) = -Σ p·log p` is
lower for confident policies, meaning `Var(∇J_REINFORCE) = Σ_t Var(A_t · ∇log π_t)`
has smaller terms since `A_t` (advantage) selects over a less diffuse distribution.

---

## 3. Upgrade Roadmap (Cumulative Chain)

Each model adds **one** improvement on top of the previous — making the mathematical
contribution of each upgrade directly measurable.

| Model | Question Encoder | Fusion | Decoder | New Component |
|-------|-----------------|--------|---------|---------------|
| **E** | BiLSTM (scratch) | FiLMFusion | LSTM + Bahdanau | ← baseline |
| **F** | **CLIP Text Transformer** | FiLMFusion | LSTM + Bahdanau | CLIP text encoder |
| **G** | CLIP Text Transformer | FiLMFusion | LSTM + **MHA** | Multi-Head Cross-Attn |
| **H** | CLIP Text Transformer | FiLMFusion | **Transformer Decoder** | Full Transformer |

---

## 4. Model F — CLIP Text Encoder

**File**: `src/models/encoder_text_clip.py`

### Why CLIP text > BiLSTM

CLIP was trained with the InfoNCE contrastive objective:

```
L = -log [ exp(sim(I_i, T_i) / τ) / Σ_j exp(sim(I_i, T_j) / τ) ]
```

where `sim(I, T) = (I · T) / (‖I‖ · ‖T‖)` (cosine similarity) and τ is a learned
temperature. This forces the model to place matching image–text pairs close together
in embedding space across 400M training examples.

Result: without any VQA fine-tuning, `CLIP_img_feat · CLIP_text_feat` is already
a meaningful cross-modal similarity score. The FiLM generator `MLP(q_feat)` starts
from a representation that already "understands" visual concepts.

### Architecture

```
input_ids (B, 77)  →  CLIPTextTransformer  →  last_hidden_state (B, 77, 512)
                                            →  pooler_output (B, 512)  [EOS token]
                   →  Linear(512, 1024)    →  q_hidden_states (B, 77, 1024)
                                            →  q_feat (B, 1024)
```

Output matches `QuestionEncoder` API exactly — drop-in replacement.

---

## 5. Model G — Multi-Head Cross-Attention Decoder

**File**: `src/models/decoder_mha_attention.py`

### Why MHA > Bahdanau

| Property | Bahdanau (additive) | Multi-Head (scaled dot-product) |
|----------|--------------------|---------------------------------|
| Heads | 1 | 8 |
| Query form | `tanh(W_h h + W_img img)` | `(h W^Q)(img W^K)^T / √d_k` |
| Gradient saturation | Yes, via tanh | Controlled by `1/√d_k` |
| Params per attention | `O(H · A)` | `O(3H²/h)` per head |
| Representational capacity | One pattern | Eight parallel patterns |

The decoder structure is identical to Model E/F except `BahdanauAttention` is
replaced by `nn.MultiheadAttention(num_heads=8, batch_first=True)`:

```python
# h_top: (B, H)  →  query: (B, 1, H)
query       = h_top.unsqueeze(1)
img_ctx, _  = self.img_mha(query, img_features, img_features)   # (B, 1, H)
q_ctx, _    = self.q_mha(query, q_hidden_states, q_hidden_states)
img_ctx     = img_ctx.squeeze(1)   # (B, H)
q_ctx       = q_ctx.squeeze(1)     # (B, H)
```

LSTM input is still `[embed ; img_ctx ; q_ctx]` — same structure, upgraded attention.

---

## 6. Model H — Full Transformer Decoder

**File**: `src/models/decoder_transformer.py`

### Architecture (Pre-Norm style)

Each of `N=4` decoder layers:

```
# 1. Masked self-attention over generated tokens
x = x + MaskedMHA(LayerNorm(x), LayerNorm(x), causal_mask)

# 2. Cross-attention over 49 image patches
x = x + MHA(LayerNorm(x), img_features, img_features)

# 3. Cross-attention over 77 question tokens
x = x + MHA(LayerNorm(x), q_hidden_states, q_hidden_states)

# 4. Position-wise FFN
x = x + FFN(LayerNorm(x))    # FFN: Linear(H,4H) → GELU → Linear(4H,H)

logits = Linear(LayerNorm(x), vocab_size)  # final layer norm before projection
```

**Pre-norm** (LayerNorm before attention, not after) is used because it produces
more stable gradients during early training — the residual stream maintains unit
variance regardless of depth.

### Training vs Inference

- **Training**: all T positions processed in **parallel** using a causal mask
  `mask[i,j] = -inf if j > i else 0`. One forward pass computes all T logits.
- **Inference**: autoregressive. At each step t, we concatenate the new token
  embedding with accumulated past embeddings (KV cache), apply the full decoder
  with causal mask, and read out position t's output.

### SCST Compatibility

The `TransformerDecoder` implements the same `sample(encoder_hidden, img_features,
q_hidden, max_len, start_idx, end_idx, method)` signature as `LSTMDecoderWithAttention`.
The `encoder_hidden` argument is ignored (no LSTM state needed); state is maintained
via the accumulated token buffer (KV cache).

---

## 7. Implementation Files

| File | Purpose |
|------|---------|
| `src/models/encoder_text_clip.py` | CLIPTextEncoder — CLIP text transformer wrapper |
| `src/models/decoder_mha_attention.py` | LSTMDecoderWithMHA — LSTM + MH cross-attention |
| `src/models/decoder_transformer.py` | TransformerDecoder — full autoregressive transformer |
| `src/dataset_clip.py` | VQAEDatasetCLIP — CLIP BPE tokenizer for questions |
| `src/models/vqa_models.py` | VQAModelF, VQAModelG, VQAModelH added |
| `src/train.py` | get_model() extended; dataset auto-selected by model type |
| `src/inference.py` | encode/decode helpers for F, G, H |

---

## 8. Expected Performance Trajectory

Each upgrade is designed to be independently measurable:

```
Model E  →  Model F:  +BLEU from better question understanding (CLIP alignment)
Model F  →  Model G:  +BLEU from richer attention (8 patterns vs 1)
Model G  →  Model H:  +BLEU from parallel token context (no LSTM bottleneck)
```

All models share the same vocabulary and training data (VQA-E + COCO Captions),
so metric differences are solely attributable to the architectural change.
