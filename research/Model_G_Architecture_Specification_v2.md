---
title: "Model G: Definitive Architectural and Algorithmic Specification for State-of-the-Art Long-Form Generative Visual Question Answering within the CNN-LSTM Paradigm"
subtitle: "Technical Design Specification · Research Architecture Document · Implementation Blueprint"
author: "VQA-E Research Team"
date: "March 2026, Revision 2.0"
abstract: |
  This document presents the definitive, mathematically complete specification for
  Model G — the culminating system in a progressive series of generative Visual
  Question Answering models (A through G) constrained to the CNN-LSTM paradigm.
  Unlike classification-based VQA that outputs a single label, Model G produces
  explanatory natural language answers of 15–30 words, e.g., "The man is riding a
  bicycle because he appears to be commuting along a city street during the
  daytime." The architecture builds incrementally upon Model F (BUTD Faster R-CNN +
  MUTAN Tucker Fusion + Dual MHCA + LayerNorm-BiLSTM + Pointer-Generator + SCST RL)
  by integrating five precisely targeted enhancements: (G1) extended 7-dimensional
  spatial geometry, (G2) a three-way Pointer-Generator enabling copy from visual
  object labels, (G3) an InfoNCE contrastive multimodal alignment loss, (G4) an
  improved Object Hallucination Penalty in the SCST reward, and (G5) length-conditioned
  decoding to prevent short-answer collapse. Each enhancement is motivated by critical
  analysis of an external AI-generated proposal (Gemini) and validated against
  published literature. The document provides: (i) complete mathematical formulations
  with verified tensor dimensions, (ii) a zero-cost data engineering strategy
  consolidating 225K filtered samples from four free sources, (iii) a four-phase
  curriculum training pipeline with experience replay, (iv) a systematic 10-experiment
  ablation plan, (v) VRAM budget analysis for RTX 5070 Ti, (vi) a multi-metric
  evaluation protocol anchored by CIDEr-D/SPICE/METEOR, and (vii) a four-week
  implementation roadmap. Total additional parameters: +0.55M (+0.26%) over Model F.
  Total monetary cost: $0.
geometry: margin=1in
fontsize: 11pt
toc: true
toc-depth: 3
numbersections: true
header-includes:
  - \usepackage{amsmath}
  - \usepackage{amssymb}
  - \usepackage{booktabs}
---

\newpage

# Problem Definition: Long-Form Generative VQA

## Task Formulation

This project addresses **Generative Visual Question Answering** — a task that
requires the model to *generate* natural language answers, not *classify* from
a fixed answer set. The distinction is fundamental: a model that outputs "yes" or
"blue" is performing classification with unnecessary autoregressive overhead. The
value of the generative formulation lies in producing explanatory answers that
demonstrate visual reasoning.

**Formal definition.** Given an image $I \in \mathbb{R}^{3 \times H_0 \times W_0}$
and a question $Q = (q_1, q_2, \ldots, q_L)$ of $L$ tokens, produce an
answer-explanation sequence $Y = (y_1, y_2, \ldots, y_T)$ with target length
$15 \leq T \leq 30$ tokens that maximizes:

$$P(Y \mid I, Q; \theta) = \prod_{t=1}^{T} P(y_t \mid y_{<t}, I, Q; \theta) \tag{1}$$

The target output follows the format: *"{answer} because {explanation}"*, where
the explanation provides visual evidence supporting the answer.

**Example:**

- **Image**: A photograph of a man cycling on a city street
- **Question**: "What is the man doing?"
- **Target output**: "riding a bicycle because he appears to be commuting along a city street during the daytime" (17 words)

## Why This Is Hard for CNN-LSTM

Three architectural constraints make long-form generation challenging:

1. **Sequential bottleneck.** LSTM processes tokens one at a time — $O(T)$ for a sequence of $T$ tokens. Generating 20 tokens requires 20 sequential decode steps, each involving two full cross-attention computations. This is fundamentally slower than parallel Transformer decoding.

2. **Length bias from data.** The VQA v2.0 warm-up data contains answers averaging 1.1 words. This creates an extreme *early termination prior*: the LSTM learns to spike the $\texttt{<end>}$ token probability after 1–3 tokens. Overcoming this requires explicit length conditioning.

3. **Memory degradation.** LSTM hidden states degrade over long sequences despite gating mechanisms. By token 20, the initial fusion context has been overwritten multiple times. Coverage mechanisms and attention provide partial mitigation but cannot fully replace the lossless memory of Transformer key-value caches.

## Project Context: Model A→F Progression

| Model | Visual Encoder | Decoder | BLEU-4 | Key Innovation |
|:------|:---------------|:--------|:-------|:---------------|
| A | SimpleCNN (scratch) | LSTM | 0.0915 | Baseline |
| B | ResNet-101 (pretrained) | LSTM | 0.1127 | Transfer learning |
| C | SimpleCNNSpatial (49 regions) | LSTM + Bahdanau | 0.0988 | Spatial attention |
| D | ResNetSpatial (49 regions) | LSTM + Bahdanau | **0.1159** | Best of A–D |
| E | ConvNeXt-Base (49 regions) | LSTM + MHCA + MUTAN | — | Tucker fusion, DCAN |
| F | BUTD Faster R-CNN (variable $k$) | LSTM + MHCA + MUTAN + PGN | — | Object features, pointer-gen |
| **G** | **BUTD + 7D geometry** | **LSTM + MHCA + MUTAN + PGN3 + LenCond** | — | **This document** |

Model D surpassed Li et al. (ECCV 2018) by +23.3% on BLEU-4. Models E/F add
8 architectural tiers. Model G adds 5 targeted enhancements plus a complete
data strategy overhaul.


\newpage

# Critical Analysis of the Gemini Proposal

An external AI system (Google Gemini) proposed a maximalist "Model G" architecture.
This section documents the module-by-module evaluation that informed our
accept/reject decisions. The full Gemini blueprint is attached as Appendix.

## Accepted (with Modification)

| Component | Gemini's Version | Our Adaptation | Rationale |
|:----------|:----------------|:--------------|:----------|
| 3-way PGN | $P = p_g P_V + p_{cQ}\alpha_Q + p_{cV}\alpha_V$ | Same math, different implementation (scatter\_add\_ with multi-word tokenization) | Eliminates OOV for visual labels |
| Extended geometry | 7-dim spatial vector + 2-layer MLP projection | 7-dim appended to ROI features, shared projection | MLP projection unnecessary when concatenated before shared encoder |
| InfoNCE loss | $\ell_2$-norm projected embeddings, $\tau$-scaled | Identical formulation, $\tau=0.07$, $d_z=256$ | Sound self-supervised alignment objective |
| OHP in SCST | $\max(0, 1.0 - \cos\_sim)$ over all tokens | $\max(0, \delta - \cos\_sim)$ over content words only with stopword filter | Gemini's version penalizes function words incorrectly |

## Rejected

| Component | Reason | VRAM Impact if Kept |
|:----------|:-------|:-------------------|
| **Online Faster R-CNN** | 4–6 GB VRAM for backbone alone; $B \leq 16$ max | +4–6 GB |
| **ELMo dual encoding** | Nested BiLSTM redundancy; +93M params; obsolete (2018) | +0.4–1.0 GB |
| **Multi-round attention** (2 rounds × 2 modalities) | 9 sequential ops/step; 3–5× latency increase; dimension mismatch ($\mathbb{R}^{310}$ vs $\mathbb{R}^{510}$) | +800 MB activations |
| **Sentinel mechanism** | Overlaps with PGN $p_{\text{gen}}$; adds parameters without additive benefit | +15 MB |
| **Bilinear context fusion** | Undefined rank $R$; dimensions 310, 510 are non-standard and appear fabricated | +50 MB |
| **GCN/ConceptNet** ($N=50$ neighbors) | 3× dataloader bottleneck; deferred to Model G+ via existing ConceptGNN module | +200 MB |

## Newly Added (Not in Gemini)

| Component | Motivation |
|:----------|:----------|
| **Length-conditioned decoding (G5)** | Prevents short-answer collapse — the single most critical training issue for Generative VQA. Gemini did not address this. |
| **Per-token loss normalization** | Equalizes gradient contribution across variable-length samples. Essential for mixed-data training. |
| **Multi-word label tokenization** | Gemini's Challenge 1: "fire hydrant" → tokens ["fire", "hydrant"] with shared attention weights. Required for correct 3-way PGN. |


\newpage

# Model G: Complete Architecture

## Design Principles

1. **Incremental over Model F**: Every Model F component is retained; all changes are additive modules or parameter extensions controlled by flags.
2. **Ablation-first**: Each of the 5 enhancements (G1–G5) is toggled by a single CLI flag, enabling systematic isolation of contributions.
3. **VRAM-conscious**: All additions fit within RTX 5070 Ti (16 GB) at $B=192$.
4. **Mathematically complete**: Every tensor dimension, activation function, and learnable parameter is specified. No undefined ranks, no inconsistent dimensions.
5. **Generative-first**: Architecture decisions prioritize long-form generation quality over classification accuracy.

## Enhancement Summary

| ID | Enhancement | Flag | +Params | +VRAM | Impact Target |
|:---|:-----------|:-----|:--------|:------|:-------------|
| G1 | Extended spatial geometry (7-dim) | `--geo7` | +2K | +2 MB | Spatial reasoning |
| G2 | Three-way Pointer-Generator | `--pgn3` | +14K | +15 MB | OOV elimination |
| G3 | InfoNCE contrastive loss | `--infonce` | +524K | +8 MB | Multimodal alignment |
| G4 | Object Hallucination Penalty | `--ohp` | 0 | +5 MB | Hallucination reduction |
| G5 | Length-conditioned decoding | `--len_cond` | +12K | +1 MB | Prevent short-answer collapse |
| | **Total** | | **+552K** | **+31 MB** | |


## 3.1 Visual Encoder: BUTD with Extended Geometry (G1)

### Pre-Extracted Features (Unchanged from Model F)

For each image $I$, Detectron2 Faster R-CNN with ResNet-101-FPN backbone
extracts $k$ region proposals ($10 \leq k \leq 36$, variable per image).
Each region $i$ has:

- **ROI feature**: $v_i^{\text{roi}} \in \mathbb{R}^{2048}$ (ROI-pooled from ResNet)
- **Bounding box**: $(x_1^i, y_1^i, x_2^i, y_2^i)$ in pixel coordinates
- **Object label**: $l_i \in \{1, \ldots, 1600\}$ (Visual Genome class index)
- **Object name**: $\text{name}_i$ (e.g., "dog", "fire hydrant", "traffic light")

### Spatial Descriptor: 5-dim → 7-dim (G1)

**Model F** (5-dim):
$$s_i^{(F)} = \left[\frac{x_1}{W}, \frac{y_1}{H}, \frac{x_2}{W}, \frac{y_2}{H}, \frac{(x_2-x_1)(y_2-y_1)}{WH}\right] \in \mathbb{R}^5$$

**Model G** (7-dim, G1):
$$s_i^{(G)} = \left[\frac{x_1}{W}, \frac{y_1}{H}, \frac{x_2}{W}, \frac{y_2}{H}, \frac{x_2-x_1}{W}, \frac{y_2-y_1}{H}, \frac{(x_2-x_1)(y_2-y_1)}{WH}\right] \in \mathbb{R}^7 \tag{2}$$

The additional dimensions $\frac{w}{W}$ and $\frac{h}{H}$ provide explicit
aspect ratio information. This allows the model to distinguish tall/narrow
objects (people, poles: $h/H \gg w/W$) from wide/flat objects (tables, cars:
$w/W \gg h/H$) without requiring implicit subtraction ($x_2/W - x_1/W$) that
the linear projection may not learn efficiently.

### Feature Projection

$$v_i^{\text{raw}} = [v_i^{\text{roi}}; s_i^{(G)}] \in \mathbb{R}^{2055}$$

$$v_i = \text{LayerNorm}\big(\text{ReLU}(W_2 \cdot \text{ReLU}(W_1 \cdot v_i^{\text{raw}} + b_1) + b_2)\big) \in \mathbb{R}^{1024} \tag{3}$$

where $W_1 \in \mathbb{R}^{1024 \times 2055}$, $W_2 \in \mathbb{R}^{1024 \times 1024}$.

$$V = \ell_2\text{-norm}(v_1, v_2, \ldots, v_k) \in \mathbb{R}^{k \times 1024} \tag{4}$$

**Masked mean** (handles variable $k$ with padding):
$$\bar{v} = \frac{\sum_{i: M_i = 1} V_i}{\sum_i M_i} \in \mathbb{R}^{1024} \tag{5}$$

where $M \in \{0,1\}^k$ is the padding mask from `butd_collate_fn`.


## 3.2 Question Encoder: BiLSTM + GloVe + CharCNN (Unchanged)

### Word Embedding

$$e_i^{\text{word}} = \text{GloVe}_{840\text{B}}(q_i) \in \mathbb{R}^{300} \tag{6}$$

Pretrained, fine-tuned during training (`freeze=False`).

### Character-Level CNN (Tier 7C)

For word $q_i$ with character sequence $(c_1, \ldots, c_M)$, $M = \min(|q_i|, 20)$:

$$E_{\text{char}} = \text{CharEmbed}(c_1, \ldots, c_M) \in \mathbb{R}^{M \times 50}$$

$$h_k = \text{MaxPool}_{1\text{d}}(\text{Conv1d}(E_{\text{char}}, \text{filters}=100, \text{kernel}=k))$$

$$h_i^{\text{char}} = [h_3; h_4; h_5] \in \mathbb{R}^{300} \tag{7}$$

### Combined Embedding

$$x_i = [e_i^{\text{word}}; h_i^{\text{char}}] \in \mathbb{R}^{600} \tag{8}$$

### BiLSTM with Highway Connections (Tier 7B)

2-layer BiLSTM with $H/2 = 512$ per direction:

$$\overrightarrow{h}_i^{(l)}, \overleftarrow{h}_i^{(l)} = \text{BiLSTM}^{(l)}(x_i^{(l)}, h_{i-1}^{(l)}) \tag{9}$$

$$h_i^{(l)} = [\overrightarrow{h}_i^{(l)}; \overleftarrow{h}_i^{(l)}] \in \mathbb{R}^{1024}$$

Highway between layers ($l > 0$):

$$g = \sigma(W_{\text{gate}} h_i^{(l-1)}), \quad x_i^{(l)} = g \odot h_i^{(l-1)} + (1-g) \odot h_{\text{lstm}}^{(l-1)} \tag{10}$$

### Attention-Pooled Summary

$$\alpha_i = \text{Softmax}_i(w^T \tanh(W_a h_i^{(L)})) \tag{11}$$

$$q = \sum_{i=1}^{L} \alpha_i h_i^{(L)} \in \mathbb{R}^{1024} \tag{12}$$

Full sequence: $Q_H = (h_1^{(L)}, \ldots, h_L^{(L)}) \in \mathbb{R}^{L \times 1024}$ retained for decoder cross-attention.


## 3.3 Multimodal Fusion: MUTAN Tucker Decomposition (Unchanged)

$$q_p = \text{Drop}_{0.5}(\tanh(W_q q)) \in \mathbb{R}^{360} \tag{13}$$

$$v_p = \text{Drop}_{0.5}(\tanh(W_v \bar{v})) \in \mathbb{R}^{360} \tag{14}$$

$$y_{\text{fused}} = \text{BN}\big(\underbrace{\text{einsum}(\texttt{bj,bjk→bk}; v_p, \underbrace{\text{einsum}(\texttt{bi,ijk→bjk}; q_p, T_c)}_{\text{query-core interaction}})}_{\text{full Tucker contraction}}\big) \in \mathbb{R}^{1024} \tag{15}$$

where $T_c \in \mathbb{R}^{360 \times 360 \times 1024}$ is the learnable core tensor, $W_q \in \mathbb{R}^{360 \times 1024}$, $W_v \in \mathbb{R}^{360 \times 1024}$.

**Decoder initialization:**
$$h_0^{(l)} = y_{\text{fused}}, \quad c_0^{(l)} = \mathbf{0} \quad \forall\, l \in \{0, \ldots, N_L - 1\} \tag{16}$$


## 3.4 LSTM Decoder with Dual MHCA, Three-Way PGN, and Length Conditioning

This is the central module of Model G. It combines the existing Model F decoder
with three new capabilities: visual-label copying (G2), and length-conditioned
generation (G5), plus the unchanged InfoNCE projections (G3) and OHP reward (G4)
which operate outside the decoder.

### 3.4.1 Length-Conditioned Embedding (G5)

Define 3 learnable length-bin embeddings:

$$E_{\text{len}} = \{e_{\texttt{SHORT}}, e_{\texttt{MED}}, e_{\texttt{LONG}}\} \subset \mathbb{R}^{64} \tag{17}$$

| Bin | Token Count | Assigned To |
|:----|:-----------|:-----------|
| `SHORT` | $1 \leq T \leq 5$ | VQA v2.0 samples |
| `MEDIUM` | $6 \leq T \leq 14$ | Transitional samples |
| `LONG` | $15 \leq T \leq 35$ | VQA-E, VQA-X, A-OKVQA |

During training, each sample's bin is computed from the ground-truth target length.
During inference, **always feed `LONG`** to condition the decoder for explanatory output.

### 3.4.2 Token Embedding

At decode step $t$:

$$e_t = \text{Drop}_{0.5}\big(\text{Proj}(\text{GloVe}(y_{t-1}))\big) \in \mathbb{R}^{512} \tag{18}$$

where $\text{Proj}: \mathbb{R}^{300} \to \mathbb{R}^{512}$ if GloVe dim $\neq$ embed\_size.

### 3.4.3 Dual Multi-Head Cross-Attention (MHCA)

**Image-side MHCA** with $N_h = 4$ heads, $d_k = 256$:

$$Q_j^{\text{img}} = h_{t-1}^{\text{top}} W_j^{Q} \in \mathbb{R}^{d_k}, \quad K_j = V_{\text{img}} W_j^{K}, \quad V_j = V_{\text{img}} W_j^{V} \in \mathbb{R}^{k \times d_k} \tag{19}$$

$$\text{score}_j = \frac{Q_j (K_j)^T}{\sqrt{d_k}} + \underbrace{s_{\text{cov}} \cdot \text{cov}_{t-1}}_{\text{coverage bias}} \tag{20}$$

where $s_{\text{cov}}$ is a learned scalar (initialized to 0) and $\text{cov}_{t-1} = \sum_{\tau < t} \alpha_\tau^{\text{img}}$.

Apply padding mask ($-\infty$ for invalid regions), then:

$$\alpha_j^{\text{img}} = \text{Softmax}(\text{score}_j + M_{\text{pad}}) \in \mathbb{R}^k \tag{21}$$

$$c_t^{\text{img}} = \text{Concat}(\alpha_1 V_1, \ldots, \alpha_{N_h} V_{N_h}) W^O \in \mathbb{R}^{1024} \tag{22}$$

Head-averaged attention: $\alpha_t^{\text{img}} = \frac{1}{N_h}\sum_j \alpha_j^{\text{img}} \in \mathbb{R}^k \tag{23}$

**Question-side MHCA**: identical architecture, own parameters, no coverage.
Produces $c_t^Q \in \mathbb{R}^{1024}$ and $\alpha_t^Q \in \mathbb{R}^L$.

### 3.4.4 LSTM Step with LayerNorm and DropConnect

**Input construction** (with length embedding, G5):

$$x_t = [e_t; c_t^{\text{img}}; c_t^Q; E_{\text{len}}[\text{bin}]] \in \mathbb{R}^{512 + 1024 + 1024 + 64} = \mathbb{R}^{2624} \tag{24}$$

**LayerNorm LSTM Cell** (Tier 1A, per gate):

$$\text{gates} = W_x x_t + \underbrace{(W_h \odot M_{\text{dc}})}_{\text{DropConnect}} h_{t-1}, \quad M_{\text{dc}} \sim \text{Bernoulli}(0.7) \tag{25}$$

$$(i_r, f_r, g_r, o_r) = \text{chunk}(\text{gates}, 4)$$

$$i_t = \sigma(\text{LN}_i(i_r)), \quad f_t = \sigma(\text{LN}_f(f_r)), \quad g_t = \tanh(\text{LN}_g(g_r)), \quad o_t = \sigma(\text{LN}_o(o_r)) \tag{26}$$

$$c_t = f_t \odot c_{t-1} + i_t \odot g_t, \quad h_t = o_t \odot \tanh(\text{LN}_c(c_t)) \in \mathbb{R}^{1024} \tag{27}$$

**Highway connections** between layers ($N_L = 2$):
$$g^{(l)} = \sigma(W_g^{(l)} h_{\text{in}}^{(l)}), \quad h_{\text{out}}^{(l)} = g^{(l)} \odot h_{\text{lstm}}^{(l)} + (1 - g^{(l)}) \odot h_{\text{in}}^{(l)} \tag{28}$$

### 3.4.5 Three-Way Pointer-Generator Network (G2)

The PGN head computes blending probabilities over three sources:

**Source 1 — Vocabulary generation:**

$$P_{\text{vocab}}(w) = \text{Softmax}\big(W_{\text{fc}}(\text{ReLU}(W_{\text{out}} h_t))\big) \in \mathbb{R}^{|V_A|} \tag{29}$$

Weight tying: $W_{\text{fc}} = W_{\text{embed}}^T$ when GloVe is not used.

**Source 2 — Copy from question tokens:**

$$P_{\text{copy\_Q}}(w) = \sum_{j: Q_j = w} \alpha_{t,j}^Q \tag{30}$$

**Source 3 — Copy from visual object labels (G2, NEW):**

Each region $i$ has label name $\text{name}_i$ (e.g., "fire hydrant"). For multi-word labels (Gemini Challenge 1), we tokenize: $\text{name}_i \to (\text{tok}_1^i, \text{tok}_2^i, \ldots)$. The attention weight $\alpha_{t,i}^{\text{img}}$ is distributed equally across the label's tokens:

$$P_{\text{copy\_V}}(w) = \sum_{i=1}^{k} \frac{\alpha_{t,i}^{\text{img}}}{|\text{tokens}(\text{name}_i)|} \cdot \mathbb{1}[w \in \text{tokens}(\text{name}_i)] \tag{31}$$

**Blending probabilities:**

$$[p_g, p_{cQ}, p_{cV}] = \text{Softmax}(W_{\text{ptr}} [c_t^{\text{img}}; h_t; x_t]) \in \mathbb{R}^3 \tag{32}$$

where $W_{\text{ptr}} \in \mathbb{R}^{3 \times (1024 + 1024 + 2624)} = \mathbb{R}^{3 \times 4672}$.

**Final distribution:**

$$\boxed{P(w \mid y_{<t}, I, Q) = p_g \cdot P_{\text{vocab}}(w) + p_{cQ} \cdot P_{\text{copy\_Q}}(w) + p_{cV} \cdot P_{\text{copy\_V}}(w)} \tag{33}$$

**Implementation** via `scatter_add_`:

```python
ext_size = vocab_size + q_len + max_label_tokens
P = torch.zeros(B, ext_size)
P[:, :V] += p_gen.unsqueeze(1) * P_vocab
P.scatter_add_(1, q_idx_expanded, p_cQ.unsqueeze(1) * alpha_Q)
P.scatter_add_(1, v_idx_expanded, p_cV.unsqueeze(1) * alpha_V_distributed)
```

At inference, indices $\geq |V_A|$ are mapped back to source tokens.


## 3.5 InfoNCE Contrastive Alignment Loss (G3)

### Projection Heads (Training Only)

$$z_i^{\text{img}} = \ell_2\text{-norm}(W_{\text{proj\_i}} \bar{v}_i) \in \mathbb{R}^{256} \tag{34}$$

$$z_i^{\text{text}} = \ell_2\text{-norm}(W_{\text{proj\_t}} h_{i,L}^{(L)}) \in \mathbb{R}^{256} \tag{35}$$

### Symmetric InfoNCE

$$\mathcal{L}_{\text{i2t}} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(z_i^{\text{img}} \cdot z_i^{\text{text}} / \tau)}{\sum_{j=1}^{B} \exp(z_i^{\text{img}} \cdot z_j^{\text{text}} / \tau)} \tag{36}$$

$$\mathcal{L}_{\text{InfoNCE}} = \frac{1}{2}(\mathcal{L}_{\text{i2t}} + \mathcal{L}_{\text{t2i}}), \quad \tau = 0.07 \tag{37}$$

Projection heads are **discarded at inference** — zero overhead at test time.


## 3.6 Object Hallucination Penalty in SCST (G4)

For generated sequence $\hat{Y}$, visual labels $\mathcal{L}_V$, stopword set $\mathcal{S}$:

$$\hat{Y}_c = \{\hat{y}_t \mid \hat{y}_t \notin \mathcal{S}\} \quad \text{(content words only)} \tag{38}$$

$$\text{ground}(\hat{y}_t) = \max_{l \in \mathcal{L}_V} \cos(\text{GloVe}(\hat{y}_t), \text{GloVe}(l)) \tag{39}$$

$$\text{OHP}(\hat{Y}) = \frac{1}{|\hat{Y}_c|} \sum_{\hat{y}_t \in \hat{Y}_c} \max(0, \delta - \text{ground}(\hat{y}_t)), \quad \delta = 0.5 \tag{40}$$

**SCST reward** (Phase 4):

$$R(\hat{Y}) = \text{CIDEr-D}(\hat{Y}) + 0.5\,\text{BLEU-4}(\hat{Y}) + 0.5\,\text{METEOR}(\hat{Y}) - 0.3\,\text{OHP}(\hat{Y}) \tag{41}$$

$$\nabla_\theta \mathcal{L}_{\text{SCST}} = -(R(\hat{Y}^s) - R(\hat{Y}^g)) \nabla_\theta \log P(\hat{Y}^s) \tag{42}$$


\newpage

# Complete Forward Pass (Single Sample)

**Input:** $F \in \mathbb{R}^{k \times 2055}$, mask $M \in \{0,1\}^k$, question $Q \in \mathbb{Z}^L$, target $Y \in \mathbb{Z}^T$, labels $L_V$, length bin $b$.

| Step | Operation | Output Shape |
|:-----|:----------|:------------|
| 1 | $V = \ell_2\text{-norm}(\text{BUTD}(F))$ | $(k, 1024)$ |
| 2 | $\bar{v} = \text{masked\_mean}(V, M)$ | $(1024,)$ |
| 3 | $Q_H, q = \text{BiLSTM\_Highway}([\text{GloVe}(Q); \text{CharCNN}(Q)])$ | $(L, 1024), (1024,)$ |
| 4 | $y_f = \text{MUTAN}(q, \bar{v})$ | $(1024,)$ |
| 5 | $h_0 = c_0 = y_f$ (broadcast to $N_L$ layers) | $(2, 1024)$ |
| 6 | $z^{\text{img}}, z^{\text{text}} = \text{InfoNCE\_proj}(\bar{v}, Q_{H,L})$ | $(256,), (256,)$ |
| 7 | **For** $t = 1, \ldots, T$: | |
| 7a | $\quad e_t = \text{Embed}(y_{t-1})$ | $(512,)$ |
| 7b | $\quad c_t^{\text{img}}, \alpha_t^{\text{img}} = \text{MHCA}_{\text{img}}(h_{t-1}, V, \text{cov}, M)$ | $(1024,), (k,)$ |
| 7c | $\quad c_t^Q, \alpha_t^Q = \text{MHCA}_Q(h_{t-1}, Q_H)$ | $(1024,), (L,)$ |
| 7d | $\quad x_t = [e_t; c_t^{\text{img}}; c_t^Q; E_{\text{len}}[b]]$ | $(2624,)$ |
| 7e | $\quad h_t, c_t = \text{LN-LSTM}(x_t, h_{t-1}, c_{t-1})$ | $(1024,), (1024,)$ |
| 7f | $\quad P(w) = p_g P_V(w) + p_{cQ} P_{cQ}(w) + p_{cV} P_{cV}(w)$ | $(|V_{\text{ext}}|,)$ |
| 8 | $\mathcal{L} = \mathcal{L}_{\text{Focal}} + \lambda_c \mathcal{L}_{\text{cov}} + \beta \mathcal{L}_{\text{InfoNCE}}$ | scalar |


\newpage

# Loss Functions

## Sequence Focal Loss (Phases 1–3)

$$\mathcal{L}_{\text{Focal}} = -\frac{1}{T}\sum_{t=1}^{T} (1-p_t)^\gamma \log p_t, \quad \gamma = 2.0 \tag{43}$$

with label smoothing $\epsilon = 0.1$:

$$\tilde{y}_t = (1 - \epsilon)\mathbf{1}_{y_t^*} + \frac{\epsilon}{|V_A|} \tag{44}$$

**Critical: per-token normalization** (divide by $T$, not sum). Without this,
short VQA v2.0 sequences ($T \approx 3$) dominate gradients over long
explanation sequences ($T \approx 20$).

## Coverage Loss

$$\mathcal{L}_{\text{cov}} = \frac{1}{T}\sum_{t=1}^{T}\sum_{i=1}^{k} \min(\alpha_{t,i}^{\text{img}}, \text{cov}_{t-1,i}), \quad \lambda_c = 0.5 \tag{45}$$

## InfoNCE (G3, all phases)

$$\mathcal{L}_{\text{InfoNCE}} \text{ — Eq. (37)}, \quad \beta = 0.1 \tag{46}$$

## SCST RL (Phase 4 only, G4)

$$\mathcal{L}_{\text{SCST}} \text{ — Eq. (42)}, \quad \lambda_{\text{scst}} = 0.5 \tag{47}$$

Mixed: $\mathcal{L}_4 = (1-\lambda_{\text{scst}})\mathcal{L}_{\text{CE}} + \lambda_{\text{scst}}\mathcal{L}_{\text{SCST}} + \beta\mathcal{L}_{\text{InfoNCE}}$

## Total Loss Per Phase

| Phase | $\mathcal{L}_{\text{total}}$ |
|:------|:---------------------------|
| 1 | $\mathcal{L}_{\text{Focal}} + 0.5\mathcal{L}_{\text{cov}} + 0.1\mathcal{L}_{\text{InfoNCE}}$ |
| 2 | Same as Phase 1 |
| 3 | Same as Phase 1 (with scheduled sampling) |
| 4 | $(0.5)\mathcal{L}_{\text{CE}} + (0.5)\mathcal{L}_{\text{SCST}} + 0.1\mathcal{L}_{\text{InfoNCE}}$ |


\newpage

# Data Strategy (Summary)

Full details in companion document: *Model G Data Strategy*.

## Data Sources (Zero Cost)

| Source | Raw | Filtered | Quality | Human-Written? |
|:-------|:----|:---------|:--------|:--------------|
| VQA-E (existing) | 210K | ~95K | 66.5% accept | No (auto-gen) |
| VQA-X (download) | 29K | ~27K | 91.4% accept | **Yes** |
| A-OKVQA ×3 rationales (download) | 51K | ~45K | High | **Yes** |
| COCO Caption templates (generate) | 60K | ~40K | Medium | Rule-based |
| Local VLM (optional) | 20K | ~12K | Medium | LLM-gen |
| **Total** | **370K** | **~219K** | | **72K human** |

## Five-Stage Quality Filter

1. **Length gate**: $5 \leq |\text{explanation}| \leq 35$ words
2. **Copy-of-question**: Jaccard$(Q, E) < 0.6$
3. **Visual grounding**: $\geq$30% content nouns match BUTD labels (with WordNet synonyms)
4. **Answer consistency**: annotated answer appears in first 5 words
5. **Deduplication**: MinHash Jaccard $< 0.85$

## Four-Phase Curriculum

| Phase | Epochs | Data Mix | Key Mechanism |
|:------|:-------|:---------|:-------------|
| 1 — Alignment | 15 | 40% VQA v2.0 + 30% VQA-E + 30% A-OKVQA | Length conditioning (SHORT+LONG) |
| 2 — Mastery | 10 | 100% explanation data + 20% VQA v2.0 replay | Experience replay |
| 3 — Self-correction | 7 | Same as Phase 2 | Scheduled sampling |
| 4 — Optimization | 3 | VQA-E + VQA-X only | SCST RL + OHP reward |


\newpage

# Training Hyperparameters

| Parameter | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|:----------|:--------|:--------|:--------|:--------|
| Learning rate | $1\text{e-}3$ | $5\text{e-}4$ | $2\text{e-}4$ | $5\text{e-}5$ |
| Warmup | 2 ep (linear) | 0 | 0 | 0 |
| Scheduler | Cosine | Cosine | ReduceOnPlateau | Constant |
| Batch size | 192 | 192 | 192 | 64 |
| Gradient clip | 2.0 | 2.0 | 2.0 | 2.0 |
| Weight decay | $1\text{e-}4$ | $1\text{e-}4$ | $1\text{e-}4$ | $1\text{e-}5$ |
| Dropout | 0.3 | 0.3 | 0.3 | 0.3 |
| DropConnect ($p_{\text{dc}}$) | 0.3 | 0.3 | 0.3 | 0.3 |
| Label smoothing | 0.1 | 0.1 | 0.1 | 0 |
| Sched. sampling $k$ | — | — | 5.0 | — |
| SCST $\lambda$ | — | — | — | 0.5 |
| InfoNCE $\beta$ | 0.1 | 0.1 | 0.1 | 0.1 |
| Coverage $\lambda_c$ | 0.5 | 0.5 | 0.5 | 0.5 |
| Mix VQA | ON | OFF (replay) | OFF (replay) | OFF |
| Focal $\gamma$ | 2.0 | 2.0 | 2.0 | OFF (plain CE) |
| Augmentation | ON | ON | ON | ON |
| Length conditioning | ON | ON | ON | ON |
| Min-length beam ($t_{\min}$) | — | — | — | 8 tokens |
| **Total epochs** | **15** | **10** | **7** | **3** |

**Total: 35 epochs. Estimated GPU time: ~30 hours on RTX 5070 Ti.**


\newpage

# Ablation Study Plan

## Experiment Matrix (10 experiments)

| # | Config | G1 | G2 | G3 | G4 | G5 | Primary Metric |
|:--|:-------|:--:|:--:|:--:|:--:|:--:|:-------------|
| 0 | Model F baseline | | | | | | All |
| 1 | +G5 only (length cond.) | | | | | $\checkmark$ | Avg output length, BLEU-4 |
| 2 | +G1 only (geo7) | $\checkmark$ | | | | | Spatial reasoning subset |
| 3 | +G2 only (pgn3) | | $\checkmark$ | | | | OOV rate, copy ratio |
| 4 | +G3 only (infonce) | | | $\checkmark$ | | | BERTScore, retrieval R@1 |
| 5 | +G4 only (ohp) | | | | $\checkmark$ | | CHAIR$_i$, hallucination |
| 6 | +G5+G2 (length+pgn3) | | $\checkmark$ | | | $\checkmark$ | Synergy: long + copy |
| 7 | +G5+G3 (length+infonce) | | | $\checkmark$ | | $\checkmark$ | Synergy: long + aligned |
| 8 | +G2+G3+G4 (arch. only) | | $\checkmark$ | $\checkmark$ | $\checkmark$ | | Arch improvements |
| 9 | **Full Model G** | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | **All metrics** |

**All experiments use identical data (219K filtered pool) and hyperparameters.**
Only architectural flags differ. Random seed fixed at 42.

## Hypotheses

1. **G5 is the single highest-impact enhancement** — expected +3–8% BLEU-4 by resolving short-answer collapse.
2. **G2 reduces OOV rate to near zero** and improves factual precision.
3. **G3 improves BERTScore more than BLEU** — alignment helps semantic quality.
4. **G4 has minimal BLEU impact** but measurably reduces CHAIR$_i$.
5. **Full Model G achieves +5–15% over Model F** across primary metrics.

## Evaluation Metrics

| Tier | Metrics |
|:-----|:--------|
| Primary (always report) | CIDEr-D, METEOR, SPICE, ROUGE-L |
| Secondary | BLEU-4, BERTScore F1, Avg Length, OOV Rate |
| Diagnostic | CHAIR$_i$, Copy Ratio ($p_{cQ}$, $p_{cV}$), Length Distribution |


\newpage

# VRAM Budget

## RTX 5070 Ti Specifications

| Property | Value |
|:---------|:------|
| GPU | NVIDIA RTX 5070 Ti (Blackwell SM 12.0) |
| VRAM | 16 GB GDDR7 |
| BF16 | Native support |
| CUDA cores | 8960 |

## Component Breakdown ($B = 192$, BF16)

| Component | Params | Activations | Total |
|:----------|:-------|:-----------|:------|
| BUTDFeatureEncoder | 4.2M (8 MB) | 14 MB | 22 MB |
| QuestionEncoder (BiLSTM+GloVe+CharCNN) | 18.5M (37 MB) | 42 MB | 79 MB |
| MUTAN ($T_c$ + projections + BN) | 134M (268 MB) | 3 MB | 271 MB |
| LSTMDecoder + dual MHCA + coverage | 42M (84 MB) | 380 MB | 464 MB |
| 3-way PGN head (G2) | 14K (0.03 MB) | 2 MB | 2 MB |
| InfoNCE projections (G3) | 524K (1 MB) | 1 MB | 2 MB |
| Length embeddings (G5) | 12K (0.02 MB) | <1 MB | <1 MB |
| Answer embeddings + FC | 12M (24 MB) | 8 MB | 32 MB |
| **Model subtotal** | **~211.3M** | | **~872 MB** |
| Optimizer (Adam moments × 2) | | | ~850 MB |
| Gradient buffers | | | ~420 MB |
| PyTorch/CUDA overhead | | | ~800 MB |
| **Total** | | | **~2.9 GB** |

**Headroom**: $16.0 - 2.9 = 13.1$ GB. Comfortably fits $B = 192$.

## vs. Gemini Proposal

| Config | Est. VRAM | Max Batch |
|:-------|:---------|:---------|
| **Model G (ours)** | ~2.9 GB | $B = 192$ |
| Gemini full (R-CNN + ELMo + multi-round) | ~12–14 GB | $B \approx 32$ |


\newpage

# Parameter Count

| Component | Parameters | Notes |
|:----------|:----------|:------|
| BUTDFeatureEncoder | 4,196K | $W_1 \in \mathbb{R}^{1024 \times 2055}$ (G1: was 2053) |
| QuestionEncoder | 18,500K | BiLSTM 2L + GloVe proj + CharCNN + attn pool |
| MUTANFusion | 134,000K | $T_c \in \mathbb{R}^{360 \times 360 \times 1024}$ |
| LSTMDecoder + MHCA | 42,200K | 2L LN-LSTM (input 2624, G5) + 2×4H MHCA + cov |
| PGN head (3-way, G2) | 14K | $W_{\text{ptr}} \in \mathbb{R}^{3 \times 4672}$ |
| InfoNCE projections (G3) | 524K | $2 \times (1024 \times 256)$ |
| Length embeddings (G5) | 12K | $3 \times 64 + \Delta W_x$ input dim adjustment |
| Answer embeddings + FC | 12,000K | $|V_A| \times 512$ + output projection |
| **Total Model G** | **~211,446K** | |
| **Total Model F** | **~210,895K** | |
| **Delta** | **+551K (+0.26%)** | |


\newpage

# Implementation Roadmap

## Four-Week Plan

| Week | Tasks | Hours |
|:-----|:------|:------|
| **1** | Download VQA-X + A-OKVQA; preprocessors; filter extension; vocab rebuild; length-conditioned decoding (G5); per-token loss norm; multi-source DataLoader | ~20h |
| **2** | 3-way PGN (G2) with multi-word tokenization; InfoNCE loss module (G3); geo7 feature extraction (G1); experience replay buffer; integration tests | ~18h |
| **3** | OHP in SCST (G4); `train_model_g.sh`; SPICE metric; full 4-phase training (30h GPU) | ~8h eng + 30h GPU |
| **4** | Ablation experiments (10 runs × ~30h each, can parallelize); evaluation; analysis | ~10h eng + GPU |
| **Total** | | **~56h eng + ~330h GPU** |

## CLI Specification

```bash
python src/train.py \
  --model G \
  --geo7 --pgn3 --infonce --ohp --len_cond \
  --use_mutan --layer_norm --dropconnect \
  --q_highway --char_cnn \
  --glove --glove_dim 300 \
  --coverage --coverage_lambda 0.5 \
  --infonce_beta 0.1 --infonce_tau 0.07 \
  --ohp_lambda 0.3 --ohp_threshold 0.5 \
  --batch_size 192 --lr 1e-3 \
  --weight_decay 1e-4 --grad_clip 2.0 \
  --label_smoothing 0.1 --dropout 0.3 \
  --augment --focal
```


\newpage

# Conclusion

Model G represents the definitive integration of architectural innovation and
data engineering within the CNN-LSTM paradigm for long-form generative VQA.
The system is defined by five enhancements (G1–G5) over Model F, each with
complete mathematical specification, verified tensor dimensions, and independent
ablation toggles:

1. **G1 — Extended 7D spatial geometry**: Explicit aspect ratio encoding for improved spatial reasoning. (+2K params)

2. **G2 — Three-way Pointer-Generator**: Copy from vocabulary, question tokens, *and* visual object labels with multi-word tokenization. Eliminates OOV errors for detected entities. (+14K params)

3. **G3 — InfoNCE contrastive alignment**: Self-supervised auxiliary loss enforcing tighter multimodal representation quality. Discarded at inference. (+524K params)

4. **G4 — Object Hallucination Penalty**: Content-word-only grounding check in SCST reward with GloVe semantic similarity and stopword filter. (0 params)

5. **G5 — Length-conditioned decoding**: Learned length-bin embeddings + per-token loss normalization + minimum-length beam constraint. The single most impactful enhancement for Generative VQA. (+12K params)

The companion *Data Strategy* document details a zero-cost data pipeline that
consolidates 225K filtered samples from four free sources, with 72K human-written
explanations as the quality backbone.

Total parameter increase: **+0.26%** over Model F. Total monetary cost: **$0**.
VRAM: **2.9 GB** at $B = 192$ on RTX 5070 Ti (16 GB). Expected improvement:
**+5–15%** across CIDEr-D, METEOR, SPICE, ROUGE-L.


\newpage

# References

1. Anderson, P., et al. (2018). Bottom-Up and Top-Down Attention for Image Captioning and VQA. *CVPR*.
2. Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization. *arXiv:1607.06450*.
3. Ben-Younes, H., et al. (2017). MUTAN: Multimodal Tucker Fusion for VQA. *ICCV*.
4. Hsieh, C.-Y., et al. (2023). Distilling Step-by-Step! *ACL Findings*.
5. Kayser, M., et al. (2021). e-ViL: A Dataset and Benchmark for NLE in Vision-Language Tasks. *ICCV*.
6. Kikuchi, Y., et al. (2016). Controlling Output Length in Neural Encoder-Decoders. *EMNLP*.
7. Li, Q., et al. (2018). VQA-E: Explaining, Elaborating, and Enhancing Your Answers. *ECCV*.
8. Mañas, O., et al. (2024). LAVE: LLM-Assisted VQA Evaluation. *AAAI*.
9. Merity, S., Keskar, N. S., & Socher, R. (2018). Regularizing and Optimizing LSTM Language Models. *ICLR*. (AWD-LSTM)
10. Oord, A. v. d., Li, Y., & Vinyals, O. (2018). Representation Learning with Contrastive Predictive Coding. *arXiv:1807.03748*. (InfoNCE)
11. Park, D. H., et al. (2018). Multimodal Explanations: Justifying Decisions and Pointing to the Evidence. *CVPR*. (VQA-X)
12. Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. *EMNLP*.
13. Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML*. (CLIP temperature)
14. Ren, S., et al. (2015). Faster R-CNN. *NeurIPS*.
15. Rennie, S. J., et al. (2017). Self-Critical Sequence Training for Image Captioning. *CVPR*.
16. Rohrbach, A., et al. (2018). Object Hallucination in Image Captioning. *EMNLP*. (CHAIR)
17. Schwenk, D., et al. (2022). A-OKVQA: A Benchmark for VQA using World Knowledge. *ECCV*.
18. See, A., Liu, P. J., & Manning, C. D. (2017). Get To The Point: Summarization with Pointer-Generator Networks. *ACL*.
19. Zhang, T., et al. (2020). BERTScore: Evaluating Text Generation with BERT. *ICLR*.
