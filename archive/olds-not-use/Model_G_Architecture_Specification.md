---
title: "Model G: A Comprehensive Architectural Specification for SOTA Generative Visual Question Answering within the CNN-LSTM Paradigm"
subtitle: "Technical Design Specification · Research Architecture Document · Implementation Roadmap"
author: "VQA-E Research Team"
date: "March 2026"
abstract: |
  This document presents a rigorous, mathematically complete architectural specification for
  Model G — the culminating system in a progressive series of generative Visual Question
  Answering with Explanation (VQA-E) models (A through G). Model G builds incrementally upon
  Model F (BUTD + MHCA + MUTAN + LayerNorm-BiLSTM) by integrating four carefully selected
  enhancements: (1) an InfoNCE contrastive alignment loss for tighter multimodal grounding,
  (2) a three-way Pointer-Generator Network enabling copying from both question tokens and
  visual object labels, (3) an improved Object Hallucination Penalty within the SCST
  reinforcement learning reward, and (4) extended spatial geometry features for richer
  region-level representation. Each addition is motivated by a critical analysis of an
  external AI-generated proposal (Gemini), from which we retain scientifically sound
  components and discard those that are redundant, computationally infeasible, or
  mathematically inconsistent. The document provides complete mathematical formulations,
  a four-phase training pipeline, a systematic ablation study plan, VRAM budget analysis
  for RTX 5070 Ti (16 GB), and a detailed implementation roadmap with dependency ordering
  and estimated engineering effort.
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

# Introduction and Motivation

## The VQA-E Task

Visual Question Answering with Explanation (VQA-E) extends the standard VQA classification
paradigm by requiring the model to generate a natural language explanation of the form
"*{answer}* because *{explanation}*" given an image $I$ and a question $Q$. This
generative formulation demands that the system not merely select from a fixed set of
answers, but produce fluent, factually grounded, and logically coherent text that
explicates the reasoning path from visual evidence to conclusion.

Formally, given an image $I \in \mathbb{R}^{3 \times H_0 \times W_0}$ and a question
$Q = (q_1, q_2, \ldots, q_L)$ consisting of $L$ word tokens, the model must produce an
answer-explanation sequence $Y = (y_1, y_2, \ldots, y_T)$ that maximizes:

$$P(Y \mid I, Q; \theta) = \prod_{t=1}^{T} P(y_t \mid y_{<t}, I, Q; \theta)$$

where $\theta$ denotes all learnable parameters.

## Project Context: The Model A$\to$F Progression

This project enforces a strict **CNN-LSTM architectural constraint** — no Transformer
encoder or decoder modules are permitted. Within this constraint, the research has
progressed through six model variants of increasing sophistication:

| Model | Visual Encoder | Decoder | Key Innovation |
|:------|:---------------|:--------|:---------------|
| A | SimpleCNN (scratch) | LSTM (no attention) | Baseline |
| B | ResNet-101 (pretrained) | LSTM (no attention) | Transfer learning |
| C | SimpleCNNSpatial (scratch, 49 regions) | LSTM + Bahdanau Attention | Spatial attention |
| D | ResNetSpatialEncoder (pretrained, 49 regions) | LSTM + Bahdanau Attention | Best of A-D (BLEU-4: 0.1159) |
| E | ConvNeXt-Base (49 regions) | LSTM + MHCA + DCAN + MUTAN | Multi-head cross-attention, Tucker fusion |
| F | BUTD Faster R-CNN (pre-extracted, variable $k$) | LSTM + MHCA + MUTAN + PGN | Object-level features, pointer-generator |

Model D achieved BLEU-4 of 0.1159, surpassing Li et al. (ECCV 2018) by +23.3%.
Models E and F represent the advanced tier with dense co-attention, Tucker fusion,
LayerNorm LSTM with Highway connections, DropConnect regularization, coverage mechanism,
Char-CNN embeddings, GloVe 840B pretrained vectors, and SCST reinforcement learning.

## The Gemini Proposal and Its Limitations

An external AI system (Google Gemini) was consulted to propose a "Model G" architecture
targeting SOTA performance. The resulting four-page blueprint specified an ambitious
system combining: online Faster R-CNN inference, GloVe+ELMo dual encoding with gated
Highway fusion, a 2-layer GCN over ConceptNet with $N=50$ neighbors, a multi-round
stacked attention decoder (2 rounds $\times$ 2 modalities = 4 attention operations per
decode step), bilinear context fusion with unspecified rank, a sentinel mechanism, a
3-way Pointer-Generator, Focal Loss, InfoNCE contrastive loss, SCST with an Object
Hallucination Penalty, and a novel reward function.

A thorough critical analysis (Section 2) identifies the following categories of issues:

1. **Computational infeasibility**: Online Faster R-CNN + ELMo + multi-round attention would exceed the 16 GB VRAM budget by a factor of $\geq 2\times$.
2. **Mathematical inconsistency**: Dimension mismatches ($\tilde{c}_q \in \mathbb{R}^{310}$ fed into $H_q \in \mathbb{R}^{R \times 510}$), undefined rank parameter $R$, undefined reward scalar $n$.
3. **Architectural redundancy**: ELMo (a BiLSTM) nested inside another BiLSTM encoder; sentinel mechanism overlapping with PGN generation probability.
4. **Information leakage**: $v_{\text{conf}} = [\text{objectness}, \max P(\text{class})]$ encodes the class prediction, allowing the decoder to shortcut visual reasoning.
5. **Obsolescence**: ELMo (2018) is superseded by contextual embedding distillation from BERT/RoBERTa.

The Gemini proposal is therefore treated as an **inspiration catalog** rather than an
implementation specification. Section 3 presents the redesigned Model G architecture that
retains the scientifically meritorious components while resolving all identified issues.


\newpage

# Critical Analysis of the Gemini Proposal

This section provides a module-by-module evaluation of the Gemini blueprint, establishing
the technical rationale for each accept/reject/modify decision.

## Visual Encoder Analysis

**Gemini proposes**: Online Faster R-CNN with ResNet backbone ($\texttt{requires\_grad}=\texttt{False}$), extracting $k$ ROI features ($10 \leq k \leq 100$) filtered by objectness $> 0.2$, with fused representation:

$$v_{\text{fused}} = \text{LayerNorm}\big(W_{\text{proj}}[v_{\text{roi}}; v_{\text{geo}}; v_{\text{conf}}]\big) \in \mathbb{R}^{1024}$$

where $v_{\text{roi}} \in \mathbb{R}^{1024}$ (ROI-pooled features after $3 \times 3$ convolution), $v_{\text{conf}} = [\text{objectness}, \max P(\text{class})] \in \mathbb{R}^{2}$, and:

$$v_{\text{geo}} = \text{ReLU}\big(W_{g2}(\text{ReLU}(W_{g1} \cdot v_{\text{geo\_raw}}))\big) \in \mathbb{R}^{64}$$

$$v_{\text{geo\_raw}} = \left[\frac{x_{\min}}{W}, \frac{y_{\min}}{H}, \frac{x_{\max}}{W}, \frac{y_{\max}}{H}, \frac{w}{W}, \frac{h}{H}, \text{area}\right] \in \mathbb{R}^{7}$$

**Critical issues**:

- **VRAM**: Detectron2 Faster R-CNN with ResNet-101-FPN consumes 4--6 GB VRAM for the backbone alone. Combined with the decoder pipeline on an RTX 5070 Ti (16 GB), this leaves insufficient memory for batch sizes above $B=16$, making training prohibitively slow.
- **Information leakage**: The $\max P(\text{class})$ component in $v_{\text{conf}}$ provides the model with the detector's class prediction. For questions like "What animal is this?", the decoder can simply read the class label rather than learning visual reasoning. This inflates training metrics while degrading out-of-distribution generalization.
- **Redundancy with Model F**: The existing `BUTDFeatureEncoder` already projects pre-extracted Faster R-CNN features ($\mathbb{R}^{2053} \to \mathbb{R}^{1024}$), where the 2053-dimensional input comprises 2048-dim ROI features concatenated with a 5-dim spatial descriptor $[x_1/W, y_1/H, x_2/W, y_2/H, \text{area}/(WH)]$.

**Decision**: **MODIFY**. Retain Model F's pre-extracted BUTD approach. Extend the spatial descriptor from 5 to 7 dimensions by appending normalized width and height: $[x_1/W, y_1/H, x_2/W, y_2/H, w/W, h/H, \text{area}/(WH)]$. Discard $v_{\text{conf}}$ entirely. Discard the 2-layer MLP geometry projection (unnecessary when spatial features are concatenated to ROI features before the shared projection).

## Text Encoder Analysis

**Gemini proposes**: Dual embedding via GloVe 300D and ELMo, combined through a gated Highway network:

$$g_e = \sigma\big(W_{\text{gate}}[e_{\text{glove}}; e_{\text{elmo}}]\big)$$
$$e_t = \text{Highway}\big(\text{LayerNorm}(g_e \odot e_{\text{glove}} + (1 - g_e) \odot e_{\text{elmo}})\big) \in \mathbb{R}^{1024}$$

**Critical issues**:

- **VRAM overhead**: The ELMo model (a 2-layer BiLSTM trained on 1 Billion Word Benchmark) consumes 400 MB--1 GB VRAM and requires a forward pass through its own BiLSTM for every batch. This is a substantial fixed cost.
- **Nested BiLSTM redundancy**: ELMo is itself a BiLSTM. Feeding ELMo embeddings into another BiLSTM question encoder creates a 4-layer deep LSTM chain without clear theoretical justification for why this outperforms a single deeper BiLSTM with the same parameter count.
- **Gradient degradation**: The proposed pipeline $\text{GloVe} + \text{ELMo} \to \text{Gate} \to \text{LayerNorm} \to \text{Highway} \to \text{BiLSTM}$ forces gradients through five sequential non-linear transformations before reaching the word embedding layer, increasing vanishing gradient risk at the very start of the encoding pipeline.
- **Obsolescence**: ELMo (Peters et al., NAACL 2018) is superseded by BERT-family models. If contextualized embeddings are desired, one-time feature extraction from a frozen BERT-base (stored to disk) is both cheaper and more powerful.

**Decision**: **REJECT**. Retain the current GloVe 840B 300D + Char-CNN (Tier 7C) configuration already implemented in Models E/F. This provides strong word-level embeddings (GloVe) with OOV resilience (Char-CNN) without the VRAM or computational overhead of ELMo.

## Knowledge Graph Integration Analysis

**Gemini proposes**: A 2-layer GCN over ConceptNet with $N=50$ neighbors per node, using GELU activation and an additive knowledge query:

$$h_i^{(l+1)} = \text{GELU}\bigg(\sum_{r \in R} \sum_{j \in \mathcal{N}_i} W_r h_j^{(l)} + W_0 h_i^{(l)}\bigg)$$

$$\text{query}_{\text{kg}} = \text{LayerNorm}(W_{kq} \cdot q_{\text{final}} + W_{kv} \cdot v_{\text{global}}) \in \mathbb{R}^{1024}$$

$$\alpha_i = \text{Softmax}\big(w_a^T \tanh(W_g \cdot \text{query}_{\text{kg}} + W_n \cdot \text{node}_i^{(2)})\big)$$

$$K_v = \sum_{i=1}^{N} \alpha_i \cdot \text{node}_i^{(2)} \in \mathbb{R}^{1024}$$

**Critical issues**:

- **Scalability**: $N=50$ neighbors per node $\times$ $L$ question words $\times$ 2 GCN layers creates a large computation graph. Message passing over 1000+ nodes per sample is expensive.
- **Notation inconsistency**: The formula uses $R$ to denote both the set of relation types (in $\sum_{r \in R}$) and the rank parameter (in $H_q \in \mathbb{R}^{R \times 510}$ later in the document). This ambiguity suggests incomplete review.
- **Bahdanau-style attention for graph**: The attention mechanism uses a simple additive score $w_a^T \tanh(\cdot)$, which does not leverage the relational structure of the graph.

**Decision**: **DEFER to future work**. The existing `ConceptGNN` module (Tier 9) is already implemented in the codebase with co-occurrence graph fallback. Model G focuses on the four highest-impact additions. Knowledge graph integration remains architecturally compatible and can be enabled as Model G+ via a single flag without modifying the core pipeline.

## Decoder Analysis

**Gemini proposes**: A multi-round stacked attention decoder with 2 rounds of alternating question/visual attention, bilinear context fusion, sentinel mechanism, knowledge gating, and 3-way Pointer-Generator.

**Critical issues**:

- **Latency**: Each decode step requires 9 sequential operations (4 attention computations, 2 gating operations, bilinear fusion, LSTM step, pointer blending). For a sequence of 20 tokens, this totals 180 matrix multiplications per sample, reducing throughput by an estimated 3--5$\times$ compared to Model F.
- **Dimension mismatch**: $\tilde{c}_q \in \mathbb{R}^{310}$ but $H_q \in \mathbb{R}^{R \times 510}$; no projection layer is specified to bridge this gap.
- **Undefined parameters**: Bilinear rank $R$ is never specified. Dimensions 310 and 510 are non-standard and appear fabricated.
- **Sentinel redundancy**: The sentinel gate $\beta_t$ controls whether to attend or use a learned sentinel vector. The PGN generation probability $p_{\text{gen}}$ already controls whether to generate from vocabulary (no attention needed) or copy from attended sources. Both mechanisms serve overlapping purposes, adding parameters without clear additive benefit.

**Decision**: **PARTIAL ACCEPT**. Retain the 3-way PGN concept (generate / copy-from-question / copy-from-visual-labels) as the single highest-impact decoder enhancement. Discard multi-round attention (keep Model F's single-round dual MHCA), discard sentinel mechanism, and discard bilinear context fusion (keep MUTAN).


\newpage

# Model G: Redesigned Architecture

Model G is defined as **Model F + four targeted enhancements**, each independently
evaluable through ablation. This incremental approach ensures scientific rigor (each
component's contribution is measurable) and engineering feasibility (implementation
builds on a verified codebase).

## Design Principles

1. **Incremental over Model F**: No component of Model F is removed; all additions are additive or modular replacements of individual sub-modules.
2. **Ablation-first design**: Each enhancement is controlled by a single command-line flag, enabling systematic ablation studies.
3. **VRAM-conscious**: All additions are designed to fit within the 16 GB VRAM budget of the RTX 5070 Ti at batch size $\geq 128$.
4. **Mathematically complete**: Every tensor dimension, every activation function, and every learnable parameter is fully specified.

## Architecture Overview

The four enhancements composing Model G are:

| ID | Enhancement | Flag | Estimated VRAM | Impact Target |
|:---|:-----------|:-----|:---------------|:-------------|
| G1 | Extended spatial geometry (7-dim) | `--geo7` | +2 MB | Spatial reasoning precision |
| G2 | Three-way Pointer-Generator | `--pgn3` | +15 MB | OOV elimination, factual grounding |
| G3 | InfoNCE contrastive alignment loss | `--infonce` | +8 MB | Multimodal representation quality |
| G4 | Object Hallucination Penalty in SCST | `--ohp` | +5 MB | Reduced hallucination, RL reward quality |

Total estimated additional VRAM: $\sim$30 MB over Model F baseline, well within budget.

## Module 1: Visual Encoder — Extended BUTD Features (G1)

### Baseline (Model F)

Model F uses pre-extracted Faster R-CNN features. For each image $I$, a Detectron2
Faster R-CNN with ResNet-101-FPN backbone extracts $k$ region proposals (with $k$
varying per image, typically $10 \leq k \leq 36$). Each region $i$ is represented by:

$$v_i^{\text{raw}} = [v_i^{\text{roi}}; v_i^{\text{spatial}}] \in \mathbb{R}^{2053}$$

where $v_i^{\text{roi}} \in \mathbb{R}^{2048}$ is the ROI-pooled feature from the
ResNet backbone and $v_i^{\text{spatial}} = [x_1/W, y_1/H, x_2/W, y_2/H, \text{area}/(WH)] \in \mathbb{R}^{5}$.

The `BUTDFeatureEncoder` projects this to the hidden dimension:

$$v_i = \text{LayerNorm}\big(\text{ReLU}(W_2 \cdot \text{ReLU}(W_1 \cdot v_i^{\text{raw}} + b_1) + b_2)\big) \in \mathbb{R}^{1024}$$

where $W_1 \in \mathbb{R}^{1024 \times 2053}$, $W_2 \in \mathbb{R}^{1024 \times 1024}$.

### Enhancement G1: Extended Spatial Descriptor

Model G extends the spatial descriptor from 5 to 7 dimensions:

$$v_i^{\text{spatial\_G}} = \left[\frac{x_1}{W}, \frac{y_1}{H}, \frac{x_2}{W}, \frac{y_2}{H}, \frac{x_2 - x_1}{W}, \frac{y_2 - y_1}{H}, \frac{(x_2-x_1)(y_2-y_1)}{WH}\right] \in \mathbb{R}^{7}$$

The normalized width $w/W$ and height $h/H$ provide the model with explicit aspect
ratio information that is only implicitly available (as the difference $x_2/W - x_1/W$)
in the 5-dim representation. This is particularly valuable for distinguishing tall/narrow
objects (e.g., people, poles) from wide/flat objects (e.g., tables, cars).

The raw feature dimension changes from 2053 to 2055, requiring only a change in the
input dimension of $W_1$:

$$W_1^{G} \in \mathbb{R}^{1024 \times 2055}$$

**Implementation**: Modify `extract_butd_features.py` to append the two additional
spatial dimensions during feature extraction. Update `BUTDFeatureEncoder.__init__` to
accept `feat_dim=2055` when `--geo7` is active.


## Module 2: Text Encoder — Unchanged from Model F

Model G retains the Model F text encoder without modification:

### Word Embedding Layer

For a question $Q = (q_1, q_2, \ldots, q_L)$ with $q_i \in \{1, \ldots, |V_Q|\}$:

$$e_i^{\text{word}} = \text{GloVe}_{840B}(q_i) \in \mathbb{R}^{300}$$

GloVe embeddings are initialized from the pretrained 840B-token Common Crawl vectors
and fine-tuned during training (`freeze=False`).

### Character-Level CNN Embedding (Tier 7C)

For each word $q_i$, its character sequence $(c_1, c_2, \ldots, c_M)$ with
$M = \min(\text{len}(q_i), 20)$ is processed by:

$$h_i^{\text{char}} = \text{MaxPool}(\text{Conv1d}_{k=3}(E_{\text{char}})) \mathbin\Vert \text{MaxPool}(\text{Conv1d}_{k=4}(E_{\text{char}})) \mathbin\Vert \text{MaxPool}(\text{Conv1d}_{k=5}(E_{\text{char}}))$$

where $E_{\text{char}} \in \mathbb{R}^{M \times 50}$ is the character embedding matrix
and each convolution has 100 filters, yielding $h_i^{\text{char}} \in \mathbb{R}^{300}$.

### Combined Input to BiLSTM

$$x_i = [e_i^{\text{word}}; h_i^{\text{char}}] \in \mathbb{R}^{600}$$

### Bidirectional LSTM with Highway Connections (Tier 7B)

The combined embeddings are processed by a 2-layer BiLSTM:

$$\overrightarrow{h}_i^{(l)}, \overleftarrow{h}_i^{(l)} = \text{BiLSTM}^{(l)}(x_i^{(l)}, h_{i-1}^{(l)})$$

$$h_i^{(l)} = [\overrightarrow{h}_i^{(l)}; \overleftarrow{h}_i^{(l)}] \in \mathbb{R}^{1024}$$

Between layers, Highway connections allow gradient bypass:

$$g = \sigma(W_{\text{gate}} \cdot h_i^{(l-1)})$$

$$x_i^{(l)} = g \odot h_i^{(l-1)} + (1 - g) \odot \text{LSTM\_out}_i^{(l-1)}$$

### Attention-Pooled Question Representation

Rather than using only the final hidden state, the encoder computes an attention-weighted
summary over all $L$ positions:

$$\alpha_i = \text{Softmax}_i\big(w^T \tanh(W_a h_i^{(L)})\big)$$

$$q = \sum_{i=1}^{L} \alpha_i \cdot h_i^{(L)} \in \mathbb{R}^{1024}$$

The full hidden state sequence $Q_H = (h_1^{(L)}, h_2^{(L)}, \ldots, h_L^{(L)}) \in \mathbb{R}^{L \times 1024}$ is also retained for use in the decoder's question-side cross-attention.


## Module 3: Multimodal Fusion — MUTAN Tucker Decomposition

Unchanged from Model F. The MUTAN fusion combines the attention-pooled question
representation $q \in \mathbb{R}^{1024}$ with the mean-pooled visual representation
$\bar{v} = \frac{1}{k'}\sum_{i: \text{mask}_i=1} v_i \in \mathbb{R}^{1024}$ (where $k'$ is the number of valid, non-padding regions):

$$q_{\text{proj}} = \text{Dropout}_{0.5}\big(\tanh(W_q \cdot q)\big) \in \mathbb{R}^{t_q}$$

$$v_{\text{proj}} = \text{Dropout}_{0.5}\big(\tanh(W_v \cdot \bar{v})\big) \in \mathbb{R}^{t_v}$$

$$y = \text{BatchNorm}\bigg(\sum_{j=1}^{t_v} v_{\text{proj},j} \cdot \bigg(\sum_{i=1}^{t_q} q_{\text{proj},i} \cdot T_c[i, j, :]\bigg)\bigg) \in \mathbb{R}^{d_{\text{out}}}$$

With $t_q = t_v = 360$, $d_{\text{out}} = 1024$, and $T_c \in \mathbb{R}^{360 \times 360 \times 1024}$ is the learnable core tensor. Equivalently in Einstein notation:

$$y = \text{BatchNorm}\big(\text{einsum}(\text{`bj,bjk}\to\text{bk'}; v_{\text{proj}}, \text{einsum}(\text{`bi,ijk}\to\text{bjk'}; q_{\text{proj}}, T_c))\big)$$

The fused output initializes the LSTM decoder:

$$h_0^{(l)} = y \quad \forall\, l \in \{0, \ldots, L_{\text{layers}}-1\}$$

$$c_0^{(l)} = \mathbf{0} \in \mathbb{R}^{1024}$$


## Module 4: LSTM Decoder with Dual MHCA and Three-Way Pointer-Generator (G2)

This is the central module of Model G and the primary architectural contribution
beyond Model F. It extends the existing decoder with a third copying source.

### 4.1 Embedding and Input Construction

At each decode step $t$, the input token $y_{t-1}$ (previous prediction or ground truth
during teacher forcing) is embedded:

$$e_t = W_{\text{embed}} \cdot y_{t-1} \in \mathbb{R}^{512}$$

where $W_{\text{embed}} \in \mathbb{R}^{|V_A| \times 512}$ is initialized from GloVe
and fine-tuned. If GloVe dimension differs from 512, a linear projection is applied.

### 4.2 Dual Multi-Head Cross-Attention (MHCA)

Model G retains the dual MHCA from Model F, which performs separate cross-attention
over image regions and question tokens at each decode step.

**Image-side MHCA** with $N_h = 4$ heads:

For head $j \in \{1, \ldots, N_h\}$ with $d_k = 1024 / N_h = 256$:

$$Q_j^{\text{img}} = h_{t-1} W_j^{Q,\text{img}} \in \mathbb{R}^{d_k}$$

$$K_j^{\text{img}} = V_{\text{img}} W_j^{K,\text{img}} \in \mathbb{R}^{k \times d_k}$$

$$V_j^{\text{img}} = V_{\text{img}} W_j^{V,\text{img}} \in \mathbb{R}^{k \times d_k}$$

$$\text{score}_j^{\text{img}} = \frac{Q_j^{\text{img}} (K_j^{\text{img}})^T}{\sqrt{d_k}}$$

If coverage is enabled (Tier 5), a learned coverage bias is added to the scores
before softmax to discourage re-attendance:

$$\text{score}_j^{\text{img}} \mathrel{+}= W_{\text{cov}} \cdot \text{cov}_{t-1}$$

where $\text{cov}_{t-1} = \sum_{\tau=0}^{t-2} \alpha_\tau^{\text{img}} \in \mathbb{R}^k$
is the cumulative attention distribution.

$$\alpha_j^{\text{img}} = \text{Softmax}\big(\text{score}_j^{\text{img}} + M_{\text{img}}\big)$$

where $M_{\text{img}}$ is a mask setting padding positions to $-\infty$.

$$\text{head}_j^{\text{img}} = \alpha_j^{\text{img}} V_j^{\text{img}} \in \mathbb{R}^{d_k}$$

$$c_t^{\text{img}} = \text{Concat}(\text{head}_1^{\text{img}}, \ldots, \text{head}_{N_h}^{\text{img}}) W^{O,\text{img}} \in \mathbb{R}^{1024}$$

The per-step image attention distribution (averaged across heads) is:

$$\alpha_t^{\text{img}} = \frac{1}{N_h} \sum_{j=1}^{N_h} \alpha_j^{\text{img}} \in \mathbb{R}^k$$

**Question-side MHCA** follows an identical structure with its own parameters,
producing $c_t^{Q} \in \mathbb{R}^{1024}$ and $\alpha_t^{Q} \in \mathbb{R}^L$.

### 4.3 LSTM Step

The LSTM input concatenates the token embedding with both context vectors:

$$x_t = [e_t; c_t^{\text{img}}; c_t^{Q}] \in \mathbb{R}^{512 + 1024 + 1024} = \mathbb{R}^{2560}$$

The LSTM cell (with LayerNorm on each gate, Tier 1A) computes:

$$\text{gates} = W_x \cdot x_t + W_h \cdot h_{t-1}$$

$$(i_t, f_t, g_t, o_t) = \text{split}(\text{gates}, 4)$$

$$i_t = \sigma(\text{LN}_i(i_t)), \quad f_t = \sigma(\text{LN}_f(f_t))$$

$$g_t = \tanh(\text{LN}_g(g_t)), \quad o_t = \sigma(\text{LN}_o(o_t))$$

$$c_t = f_t \odot c_{t-1} + i_t \odot g_t$$

$$h_t = o_t \odot \tanh(\text{LN}_c(c_t)) \in \mathbb{R}^{1024}$$

DropConnect (Tier 1B) is applied to $W_h$ during training:

$$W_h \leftarrow W_h \odot M, \quad M_{ij} \sim \text{Bernoulli}(1 - p_{\text{dc}}), \quad p_{\text{dc}} = 0.3$$

Highway connections between LSTM layers ($L_{\text{layers}} = 2$) follow the same gating
mechanism described in the encoder.

### 4.4 Three-Way Pointer-Generator Network (G2)

This is the key enhancement in Model G's decoder. The existing Model F PGN supports
two modes: generate from vocabulary or copy from question. Model G adds a third mode:
copy from visual object labels.

**Source 1: Vocabulary generation**

$$P_{\text{vocab}}(w) = \text{Softmax}(W_{\text{fc}} \cdot W_{\text{out}} \cdot h_t^{\text{know}})$$

where $W_{\text{out}} \in \mathbb{R}^{512 \times 1024}$, $W_{\text{fc}} \in \mathbb{R}^{|V_A| \times 512}$, and $h_t^{\text{know}}$ is the knowledge-enhanced hidden state (defined below).

**Source 2: Copy from question tokens**

$$P_{\text{copy\_Q}}(w) = \sum_{j: Q_j = w} \alpha_{t,j}^{Q,(2)}$$

where $\alpha_t^{Q,(2)}$ is the question attention distribution from the MHCA.

**Source 3: Copy from visual object labels (NEW in G2)**

Each visual region $i$ has an associated object label $l_i$ (the Faster R-CNN detection
class name, e.g., "dog", "person", "car"). These labels are extracted during the BUTD
feature pre-extraction phase and stored alongside the ROI features.

$$P_{\text{copy\_V}}(w) = \sum_{i: l_i = w} \alpha_{t,i}^{\text{img},(2)}$$

where $\alpha_t^{\text{img},(2)}$ is the image attention distribution.

**Blending probabilities**:

$$[p_{\text{gen}}, p_{\text{copy\_Q}}, p_{\text{copy\_V}}] = \text{Softmax}(W_{\text{ptr}} \cdot [c_t^{\text{img}}; h_t^{\text{know}}; x_t]) \in \mathbb{R}^3$$

where $W_{\text{ptr}} \in \mathbb{R}^{3 \times (1024 + 1024 + 2560)}$.

**Final word probability**:

$$P(w \mid y_{<t}, I, Q) = p_{\text{gen}} \cdot P_{\text{vocab}}(w) + p_{\text{copy\_Q}} \cdot P_{\text{copy\_Q}}(w) + p_{\text{copy\_V}} \cdot P_{\text{copy\_V}}(w)$$

**Implementation via `scatter_add_`**: The vocabulary, question tokens, and visual labels
occupy different index ranges in an extended vocabulary of size $|V_A| + L + k$. The three
distributions are computed on their respective ranges, then merged via:

```
extended_vocab = torch.zeros(B, |V_A| + L + k)
extended_vocab[:, :V_A] += p_gen * P_vocab
extended_vocab.scatter_add_(1, q_indices, p_copy_Q * alpha_Q)
extended_vocab.scatter_add_(1, v_indices, p_copy_V * alpha_img)
```

where `q_indices` and `v_indices` map question tokens and visual labels to their
positions in the extended vocabulary. At inference time, indices beyond $|V_A|$ are
mapped back to their source tokens for output.


## Module 5: InfoNCE Contrastive Alignment Loss (G3)

### Motivation

Standard cross-entropy training optimizes word-level prediction accuracy but does not
explicitly enforce that the joint multimodal representation faithfully aligns image
content with textual semantics. InfoNCE (Noise-Contrastive Estimation) provides a
self-supervised auxiliary objective that maximizes mutual information between the visual
and textual embeddings of matched image-question pairs while minimizing it for
mismatched pairs within the same mini-batch.

### Formulation

For a mini-batch of $B$ samples, define projected embeddings:

$$z_i^{\text{img}} = \ell_2\text{-norm}\big(W_{\text{proj\_img}} \cdot \bar{v}_i\big) \in \mathbb{R}^{d_z}$$

$$z_i^{\text{text}} = \ell_2\text{-norm}\big(W_{\text{proj\_text}} \cdot h_{i,L}^{(L)}\big) \in \mathbb{R}^{d_z}$$

where $\bar{v}_i$ is the masked-mean visual representation for sample $i$,
$h_{i,L}^{(L)}$ is the final hidden state of the BiLSTM question encoder,
$W_{\text{proj\_img}}, W_{\text{proj\_text}} \in \mathbb{R}^{d_z \times 1024}$
are learnable projection matrices, and $d_z = 256$.

The InfoNCE loss for the image-to-text direction is:

$$\mathcal{L}_{\text{i2t}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(z_i^{\text{img}} \cdot z_i^{\text{text}} / \tau)}{\sum_{j=1}^{B} \exp(z_i^{\text{img}} \cdot z_j^{\text{text}} / \tau)}$$

Symmetrically, the text-to-image direction:

$$\mathcal{L}_{\text{t2i}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(z_i^{\text{text}} \cdot z_i^{\text{img}} / \tau)}{\sum_{j=1}^{B} \exp(z_i^{\text{text}} \cdot z_j^{\text{img}} / \tau)}$$

The combined contrastive loss:

$$\mathcal{L}_{\text{InfoNCE}} = \frac{1}{2}(\mathcal{L}_{\text{i2t}} + \mathcal{L}_{\text{t2i}})$$

where $\tau = 0.07$ is the temperature hyperparameter (following CLIP conventions).

### Integration with Training Pipeline

The InfoNCE loss is added as an auxiliary objective during all four training phases:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{primary}} + \beta \cdot \mathcal{L}_{\text{InfoNCE}}$$

where $\mathcal{L}_{\text{primary}}$ is the phase-dependent primary loss (Focal CE in
Phases 1--3, SCST in Phase 4) and $\beta = 0.1$ is a weighting coefficient. The
projection heads $W_{\text{proj\_img}}$ and $W_{\text{proj\_text}}$ are discarded at
inference time.


## Module 6: Object Hallucination Penalty in SCST Reward (G4)

### Motivation

Standard SCST optimizes CIDEr-D and BLEU-4, which reward fluent text that matches
reference $n$-grams but do not explicitly penalize hallucinated visual content. A model
can achieve high CIDEr by generating plausible-sounding descriptions that mention objects
not actually present in the image.

### Improved OHP Design

For a generated sequence $\hat{Y} = (\hat{y}_1, \ldots, \hat{y}_T)$ and the set of
visual labels $\mathcal{L}_V = \{l_1, l_2, \ldots, l_k\}$ detected by Faster R-CNN:

**Step 1: Filter content words.** Define a stopword set $\mathcal{S}$ containing function
words (articles, prepositions, conjunctions, auxiliary verbs). Only content words are
evaluated:

$$\hat{Y}_{\text{content}} = \{\hat{y}_t \mid \hat{y}_t \notin \mathcal{S}\}$$

**Step 2: Compute grounding score.** For each content word $\hat{y}_t$, compute
the maximum semantic similarity to any visual label using a *frozen* embedding space
(GloVe 840B, which provides decent synonym coverage):

$$\text{ground}(\hat{y}_t) = \max_{l \in \mathcal{L}_V} \cos\big(\text{GloVe}(\hat{y}_t), \text{GloVe}(l)\big)$$

**Step 3: Compute penalty.** Words with low grounding scores incur a penalty:

$$\text{OHP}(\hat{Y}) = \frac{1}{|\hat{Y}_{\text{content}}|} \sum_{\hat{y}_t \in \hat{Y}_{\text{content}}} \max\big(0, \delta - \text{ground}(\hat{y}_t)\big)$$

where $\delta = 0.5$ is the grounding threshold. Words with cosine similarity $\geq 0.5$
to at least one visual label are considered grounded and incur no penalty.

**Step 4: Integrate into SCST reward.**

$$R(\hat{Y}) = w_{\text{CIDEr}} \cdot \text{CIDEr-D}(\hat{Y}) + w_{\text{BLEU}} \cdot \text{BLEU-4}(\hat{Y}) + w_{\text{METEOR}} \cdot \text{METEOR}(\hat{Y}) - \lambda_{\text{OHP}} \cdot \text{OHP}(\hat{Y})$$

with default weights $w_{\text{CIDEr}} = 1.0$, $w_{\text{BLEU}} = 0.5$,
$w_{\text{METEOR}} = 0.5$, and $\lambda_{\text{OHP}} = 0.3$.

The SCST gradient update remains:

$$\nabla_\theta \mathcal{L}_{\text{SCST}} = -\big(R(\hat{Y}^s) - R(\hat{Y}^g)\big) \nabla_\theta \log P(\hat{Y}^s \mid I, Q; \theta)$$

where $\hat{Y}^s$ is the sampled sequence and $\hat{Y}^g$ is the greedy baseline.

### Advantages over Gemini's OHP

| Aspect | Gemini's OHP | Model G's OHP |
|:-------|:-------------|:-------------|
| Scope | All generated tokens | Content words only (stopword filter) |
| Similarity | Raw cosine (harsh on synonyms) | GloVe cosine with $\delta=0.5$ threshold |
| Penalty form | $\max(0, 1.0 - \text{sim})$ (always nonzero) | $\max(0, \delta - \text{sim})$ (zero when grounded) |
| Wrong-answer penalty | $-1.0$ (high variance) | Part of CIDEr/BLEU (smooth) |


\newpage

# Complete Forward Pass Specification

This section traces a single training sample through the entire Model G pipeline,
specifying every tensor shape transformation.

**Input**: Image features $F \in \mathbb{R}^{k \times 2055}$, mask $M \in \{0,1\}^k$,
question $Q \in \mathbb{Z}^L$, target $Y \in \mathbb{Z}^T$, visual labels
$L_V \in \mathbb{Z}^k$.

## Step 1: Visual Encoding

$$V = \text{BUTDFeatureEncoder}(F) = \text{LN}(\text{ReLU}(W_2 \text{ReLU}(W_1 F))) \in \mathbb{R}^{k \times 1024}$$

$$V \leftarrow \ell_2\text{-norm}(V, \text{dim}=-1)$$

$$\bar{v} = \frac{\sum_{i: M_i=1} V_i}{\sum_{i} M_i} \in \mathbb{R}^{1024}$$

## Step 2: Question Encoding

$$E_Q = [\text{GloVe}(Q); \text{CharCNN}(Q)] \in \mathbb{R}^{L \times 600}$$

$$Q_H, q = \text{BiLSTM\_Highway}(E_Q) \quad Q_H \in \mathbb{R}^{L \times 1024}, q \in \mathbb{R}^{1024}$$

## Step 3: Multimodal Fusion

$$\text{fused} = \text{MUTAN}(q, \bar{v}) \in \mathbb{R}^{1024}$$

$$h_0 = c_0 = \text{fused} \quad (\text{broadcast to all layers})$$

## Step 4: InfoNCE Projections (G3, training only)

$$z^{\text{img}} = \ell_2\text{-norm}(W_{\text{proj\_img}} \bar{v}) \in \mathbb{R}^{256}$$

$$z^{\text{text}} = \ell_2\text{-norm}(W_{\text{proj\_text}} Q_{H,L}) \in \mathbb{R}^{256}$$

## Step 5: Autoregressive Decoding (for each step $t = 1, \ldots, T$)

$$e_t = W_{\text{embed}} y_{t-1} \in \mathbb{R}^{512}$$

$$c_t^{\text{img}}, \alpha_t^{\text{img}} = \text{MHCA\_img}(h_{t-1}, V, M) \in \mathbb{R}^{1024}, \mathbb{R}^k$$

$$c_t^Q, \alpha_t^Q = \text{MHCA\_Q}(h_{t-1}, Q_H) \in \mathbb{R}^{1024}, \mathbb{R}^L$$

$$x_t = [e_t; c_t^{\text{img}}; c_t^Q] \in \mathbb{R}^{2560}$$

$$h_t, c_t = \text{LayerNormLSTM}(x_t, (h_{t-1}, c_{t-1}))$$

$$[p_{\text{gen}}, p_{\text{cQ}}, p_{\text{cV}}] = \text{Softmax}(W_{\text{ptr}}[c_t^{\text{img}}; h_t; x_t])$$

$$P(w) = p_{\text{gen}} P_{\text{vocab}}(w) + p_{\text{cQ}} \sum_{j:Q_j=w} \alpha_{t,j}^Q + p_{\text{cV}} \sum_{i:l_i=w} \alpha_{t,i}^{\text{img}}$$

## Step 6: Loss Computation

$$\mathcal{L} = \mathcal{L}_{\text{Focal}}(P, Y) + \lambda_{\text{cov}} \mathcal{L}_{\text{cov}} + \beta \mathcal{L}_{\text{InfoNCE}}$$

In Phase 4 (SCST):

$$\mathcal{L} = (1 - \lambda_{\text{scst}}) \mathcal{L}_{\text{CE}} + \lambda_{\text{scst}} \mathcal{L}_{\text{SCST}} + \beta \mathcal{L}_{\text{InfoNCE}}$$


\newpage

# Training Pipeline

Model G follows the same four-phase progressive training regime as Model F, with
the InfoNCE loss active throughout and the OHP penalty active in Phase 4 only.

## Phase 1: Baseline Training (15 epochs)

| Parameter | Value | Rationale |
|:----------|:------|:----------|
| Learning rate | $1 \times 10^{-3}$ | Standard Adam starting rate |
| Warmup | 2 epochs (linear) | Prevents early divergence |
| Loss | SequenceFocalLoss ($\gamma = 2.0$) + InfoNCE ($\beta=0.1$) | Focus on hard tokens + alignment |
| Scheduled sampling | OFF | Pure teacher forcing |
| Curriculum | OFF | Full dataset |
| CNN backbone | Frozen (N/A for BUTD) | Features pre-extracted |
| Augmentation | ON (HFlip + ColorJitter on features) | Standard VQA augmentation |
| Label smoothing | 0.1 | Prevents overconfident predictions |
| Gradient clipping | 2.0 | Prevents explosion |
| Weight decay | $1 \times 10^{-4}$ | AdamW regularization |
| Mix VQA | ON | Blend VQA + VQA-E data |

## Phase 2: Fine-tuning (10 epochs)

| Parameter | Value | Change from Phase 1 |
|:----------|:------|:-------------------|
| Learning rate | $5 \times 10^{-4}$ | Reduced by $2\times$ |
| Warmup | 0 | Warm start from Phase 1 |
| Curriculum | ON | Progressive question-type difficulty |
| Mix VQA | OFF | VQA-E only |

## Phase 3: Scheduled Sampling (7 epochs)

| Parameter | Value | Change from Phase 2 |
|:----------|:------|:-------------------|
| Learning rate | $2 \times 10^{-4}$ | Further reduced |
| Scheduled sampling | ON ($k=5$) | $\epsilon_t = k/(k + \exp(t/k))$ |
| Early stopping | Patience 3 | Prevent overfitting |

## Phase 4: SCST Reinforcement Learning (3 epochs)

| Parameter | Value | Change from Phase 3 |
|:----------|:------|:-------------------|
| Learning rate | $5 \times 10^{-5}$ | Very low for stability |
| Loss | $(1-\lambda) \text{CE} + \lambda \text{SCST}$, $\lambda = 0.5$ | Mixed objective |
| Focal loss | OFF | Plain CE for RL stability |
| Reward | CIDEr + 0.5 BLEU-4 + 0.5 METEOR $-$ 0.3 OHP | G4 penalty active |
| Beam width (baseline) | 1 (greedy) | SCST baseline |

**Total training**: 35 epochs across all four phases.


\newpage

# Loss Functions — Complete Specification

## Sequence Focal Loss (Phases 1--3)

$$\mathcal{L}_{\text{Focal}} = -\frac{1}{T} \sum_{t=1}^{T} (1 - p_t)^\gamma \log(p_t)$$

where $p_t = P(y_t^* \mid y_{<t}, I, Q)$ is the model's predicted probability of the
ground truth token at step $t$, and $\gamma = 2.0$. Label smoothing with
$\epsilon = 0.1$ modifies the target distribution:

$$y_t^{\text{smooth}} = (1 - \epsilon) \cdot \mathbf{1}_{y_t^*} + \frac{\epsilon}{|V_A|}$$

## Coverage Loss

$$\mathcal{L}_{\text{cov}} = \frac{1}{T} \sum_{t=1}^{T} \sum_{i=1}^{k} \min(\alpha_{t,i}^{\text{img}}, \text{cov}_{t-1,i})$$

This penalizes attending to the same image region repeatedly.
Weight: $\lambda_{\text{cov}} = 0.5$.

## InfoNCE Contrastive Loss (G3, all phases)

$$\mathcal{L}_{\text{InfoNCE}} = \frac{1}{2}\bigg(-\frac{1}{B}\sum_{i=1}^{B} \log \frac{e^{z_i^{\text{img}} \cdot z_i^{\text{text}}/\tau}}{\sum_j e^{z_i^{\text{img}} \cdot z_j^{\text{text}}/\tau}} - \frac{1}{B}\sum_{i=1}^{B} \log \frac{e^{z_i^{\text{text}} \cdot z_i^{\text{img}}/\tau}}{\sum_j e^{z_i^{\text{text}} \cdot z_j^{\text{img}}/\tau}}\bigg)$$

Weight: $\beta = 0.1$. Temperature: $\tau = 0.07$.

## SCST Reinforcement Learning Loss (Phase 4 only, G4)

$$\mathcal{L}_{\text{SCST}} = -\big(R(\hat{Y}^s) - R(\hat{Y}^g)\big) \sum_{t=1}^{T} \log P(\hat{y}_t^s \mid \hat{y}_{<t}^s, I, Q)$$

$$R(\hat{Y}) = \text{CIDEr-D}(\hat{Y}) + 0.5 \cdot \text{BLEU-4}(\hat{Y}) + 0.5 \cdot \text{METEOR}(\hat{Y}) - 0.3 \cdot \text{OHP}(\hat{Y})$$


\newpage

# Ablation Study Plan

A rigorous ablation study isolates the contribution of each enhancement. All experiments
use identical hyperparameters, data splits, and random seeds. Evaluation is performed
on the VQA-E validation set with $n = 88{,}488$ samples.

## Ablation Matrix

| Experiment | G1 (geo7) | G2 (pgn3) | G3 (infonce) | G4 (ohp) | Primary Metric |
|:-----------|:---------:|:---------:|:------------:|:--------:|:--------------|
| **Baseline (Model F)** | | | | | BLEU-4, METEOR, ROUGE-L |
| **+G1 only** | $\checkmark$ | | | | Spatial reasoning accuracy |
| **+G2 only** | | $\checkmark$ | | | OOV rate, copy accuracy |
| **+G3 only** | | | $\checkmark$ | | Retrieval R@1, alignment |
| **+G4 only** | | | | $\checkmark$ | CHAIR$_i$, hallucination rate |
| **+G1+G2** | $\checkmark$ | $\checkmark$ | | | Combined copy + spatial |
| **+G2+G3** | | $\checkmark$ | $\checkmark$ | | Copy + alignment synergy |
| **+G3+G4** | | | $\checkmark$ | $\checkmark$ | Alignment + grounding |
| **Full Model G** | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | All metrics |

## Evaluation Metrics

| Metric | Type | Purpose |
|:-------|:-----|:--------|
| BLEU-4 | NLG quality | 4-gram precision against references |
| METEOR | Semantic | Synonym-aware alignment with human judgment |
| ROUGE-L | Structural | Longest common subsequence F1 |
| CIDEr-D | Image-specific | TF-IDF weighted consensus (primary for SCST) |
| BERTScore | Contextual | Embedding-based semantic similarity |
| Exact Match | Strict | Full string equality |
| OOV Rate | Copy effectiveness | Percentage of $\langle\text{unk}\rangle$ in outputs |
| CHAIR$_i$ | Hallucination | Fraction of hallucinated object mentions |
| Copy Ratio | PGN analysis | $\mathbb{E}[p_{\text{copy\_Q}} + p_{\text{copy\_V}}]$ per sample |

## Ablation Hypotheses

1. **G1 (geo7)**: Expected $\leq$0.5\% improvement on BLEU-4; primary value is on spatial reasoning subset (position/counting questions).
2. **G2 (pgn3)**: Expected 1--3\% improvement on BLEU-4 due to exact copying of visual labels; OOV rate should drop to near zero.
3. **G3 (infonce)**: Expected 0.5--1\% improvement on BLEU-4; primary value visible in BERTScore and retrieval metrics (R@1).
4. **G4 (ohp)**: Expected minimal change in BLEU-4; primary value is reduced CHAIR$_i$ (hallucination metric) and qualitative improvement.
5. **Full Model G**: Expected 2--5\% aggregate improvement over Model F across primary metrics.


\newpage

# VRAM Budget Analysis

## Hardware Specification

| Component | Specification |
|:----------|:-------------|
| GPU | NVIDIA RTX 5070 Ti |
| Architecture | Blackwell (SM 12.0) |
| VRAM | 16 GB GDDR7 |
| BF16 | Native support |
| Compute | CUDA cores: 8960 |

## Per-Component VRAM Breakdown (BF16, batch size $B=192$)

| Component | Parameters | Activations (B=192) | Total |
|:----------|:-----------|:-------------------|:------|
| BUTDFeatureEncoder ($W_1, W_2$, LN) | 4.2M ($\approx$ 8 MB) | 14 MB | 22 MB |
| QuestionEncoder (BiLSTM + GloVe + CharCNN) | 18.5M ($\approx$ 37 MB) | 42 MB | 79 MB |
| MUTAN ($T_c$, $W_q$, $W_v$, BN) | 134M ($\approx$ 268 MB) | 3 MB | 271 MB |
| LSTMDecoderWithAttention (MHCA + LN-LSTM) | 42M ($\approx$ 84 MB) | 380 MB | 464 MB |
| PointerGeneratorHead (3-way, $W_{\text{ptr}}$) | 0.014M ($\approx$ 0.03 MB) | 2 MB | 2 MB |
| InfoNCE projections ($W_{\text{proj}} \times 2$) | 0.52M ($\approx$ 1 MB) | 1 MB | 2 MB |
| Embeddings (GloVe Q + A, answer FC) | 12M ($\approx$ 24 MB) | 8 MB | 32 MB |
| **Subtotal (model)** | **$\approx$211M** | **$\approx$422 MB** | **$\approx$450 MB** |
| **Subtotal (activations + gradients)** | | | **$\approx$870 MB** |
| Optimizer states (Adam: 2 moments) | | | **$\approx$850 MB** |
| Gradient accumulation buffers | | | **$\approx$420 MB** |
| PyTorch overhead + CUDA context | | | **$\approx$800 MB** |
| **Total estimated** | | | **$\approx$3.4 GB** |

**Headroom**: $16.0 - 3.4 = 12.6$ GB available for activation checkpointing peaks and dynamic allocation. Model F at $B=192$ has been benchmarked at approximately 3.0 GB peak VRAM, so Model G's additional 30 MB of parameters and minimal extra activations remain well within budget.

## Comparison with Gemini's Proposal

| Configuration | Estimated VRAM | Feasible at $B \geq 128$? |
|:-------------|:--------------|:-------------------------|
| **Model G (ours)** | $\sim$3.4 GB | Yes ($B=192$ safe) |
| Gemini proposal (online Faster R-CNN) | $\sim$8--10 GB backbone alone | No |
| Gemini + ELMo | +0.4--1.0 GB | Marginal |
| Gemini + multi-round attention (2$\times$) | +$\sim$800 MB activations | Tight at $B=128$ |
| **Gemini full proposal** | **$\sim$12--14 GB** | **No** (max $B \approx 32$) |


\newpage

# Implementation Roadmap

## Dependency Graph

The four enhancements have minimal interdependencies:

```
G1 (geo7) ─────────────────────────────> G2 (pgn3)
  [extract_butd_features.py]              [decoder_attention.py]
  [encoder_cnn.py]                        [vqa_models.py]
                                              │
                                              ├──> G4 (ohp)
                                              │      [training/scst.py]
                                              │
G3 (infonce) ─────────────────────────────────┘
  [losses/infonce.py]                     [train.py integration]
  [vqa_models.py forward()]
```

G1 and G3 are fully independent. G2 must be implemented before G4 (OHP requires visual
labels in the decoding pipeline). G3 can be developed and tested in parallel with G2.

## Task Breakdown and Estimated Effort

| Task | Files Modified | New Files | Est. Hours | Dependencies |
|:-----|:--------------|:----------|:-----------|:------------|
| **G1: Extended geometry** | `extract_butd_features.py`, `encoder_cnn.py` | — | 2--3h | None |
| **G1: Re-extract features** | — | — | 4--6h (GPU time) | G1 code |
| **G2: 3-way PGN head** | `decoder_attention.py` | — | 6--8h | None |
| **G2: Visual label storage** | `extract_butd_features.py`, `dataset.py` | — | 3--4h | G1 or parallel |
| **G2: Extended collate function** | `dataset.py` | — | 2--3h | G2 storage |
| **G3: InfoNCE loss** | — | `losses/infonce.py` | 3--4h | None |
| **G3: Integration** | `vqa_models.py`, `train.py` | — | 2--3h | G3 loss |
| **G4: OHP reward** | `training/scst.py` | — | 4--5h | G2 (labels available) |
| **G4: Stopword filter** | — | `utils/stopwords.py` | 1h | None |
| **Testing: Unit tests** | — | `tests/test_model_g.py` | 4--5h | All |
| **Training: Full 4-phase** | `train_model_g.sh` | `train_model_g.sh` | 1--2h (script) + 24--36h (GPU) | All |
| **Ablation: 9 experiments** | — | — | 9 $\times$ 24h GPU | All |
| **Total engineering** | | | **$\sim$30--40h** | |
| **Total GPU time** | | | **$\sim$250--350h** | |

## Recommended Implementation Order

1. **Week 1**: G1 (geometry extension) + G3 (InfoNCE loss) — independent, can be parallel.
2. **Week 2**: G2 (3-way PGN) — most complex, needs careful testing.
3. **Week 3**: G4 (OHP reward) — depends on G2 for visual labels in decode pipeline.
4. **Week 4**: Integration testing, `train_model_g.sh`, and full training run.
5. **Weeks 5--6**: Ablation experiments (can be batched on multiple GPUs if available).

## Command-Line Interface

```bash
# Full Model G training
python src/train.py \
  --model G \
  --use_mutan --layer_norm --dropconnect \
  --q_highway --char_cnn \
  --glove --glove_dim 300 \
  --coverage --coverage_lambda 0.5 \
  --geo7 \
  --pgn3 \
  --infonce --infonce_beta 0.1 --infonce_tau 0.07 \
  --scst --ohp --ohp_lambda 0.3 --ohp_threshold 0.5 \
  --batch_size 192 --epochs 35 \
  --lr 1e-3 --weight_decay 1e-4 --grad_clip 2.0

# Ablation: Model F + G3 only
python src/train.py \
  --model F \
  --infonce --infonce_beta 0.1 \
  [... other Model F flags ...]
```


\newpage

# Parameter Count Summary

| Component | Trainable Parameters | Notes |
|:----------|:--------------------|:------|
| BUTDFeatureEncoder | 4.2M | $W_1 \in \mathbb{R}^{1024 \times 2055}$ + $W_2 \in \mathbb{R}^{1024 \times 1024}$ + LN |
| QuestionEncoder | 18.5M | BiLSTM ($2 \times 2$ layers) + GloVe proj + CharCNN + attn pool |
| MUTANFusion | 134M | $T_c \in \mathbb{R}^{360 \times 360 \times 1024}$ + $W_q, W_v$ |
| LSTMDecoder + MHCA | 42M | 2-layer LN-LSTM + 2 $\times$ 4-head MHCA + coverage |
| PointerGeneratorHead (3-way) | 14K | $W_{\text{ptr}} \in \mathbb{R}^{3 \times 4608}$ |
| InfoNCE projections | 524K | $2 \times (1024 \times 256)$ |
| Answer embeddings + FC | 12M | $|V_A| \times 512$ + output projection |
| **Total** | **$\approx$211M** | +0.5M over Model F |

The parameter increase from Model F to Model G is negligible ($\approx$0.5M, or +0.24\%),
confirming that the enhancements are efficiency-conscious.


\newpage

# Conclusion

Model G represents a disciplined, scientifically rigorous advancement over Model F
within the CNN-LSTM paradigm for generative VQA-E. Rather than pursuing the maximalist
approach proposed by the Gemini blueprint — which would have resulted in a computationally
infeasible, mathematically inconsistent, and architecturally redundant system — Model G
selects four high-impact enhancements with clear theoretical motivation, complete
mathematical specification, and practical implementability:

1. **Extended spatial geometry (G1)**: Provides explicit aspect ratio information to the visual encoder with negligible computational cost.

2. **Three-way Pointer-Generator (G2)**: Extends the copy mechanism to visual object labels, enabling zero-OOV generation and factual grounding in detected entities.

3. **InfoNCE contrastive alignment (G3)**: Enforces tighter multimodal representation alignment through a self-supervised auxiliary objective.

4. **Object Hallucination Penalty (G4)**: Directly penalizes hallucinated content in the SCST reward function, improving factual accuracy of generated explanations.

Each enhancement is independently toggleable, enabling systematic ablation studies
that isolate individual contributions. The total parameter increase is +0.24\% over
Model F, and the estimated VRAM overhead of $\sim$30 MB keeps the system well within the
16 GB budget of the RTX 5070 Ti at batch sizes of 192.

The implementation roadmap specifies a 4-week development timeline with clear dependency
ordering, requiring approximately 30--40 hours of engineering effort and 250--350 hours
of GPU training time for the full ablation matrix.


\newpage

# References

1. Anderson, P., He, X., Buehler, C., et al. (2018). Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering. *CVPR 2018*.

2. Ben-Younes, H., Cadene, R., Cord, M., et al. (2017). MUTAN: Multimodal Tucker Fusion for Visual Question Answering. *ICCV 2017*.

3. Chen, L., Yan, X., Xiao, J., et al. (2020). Counterfactual Samples Synthesizing for Robust Visual Question Answering. *CVPR 2020*.

4. Li, Q., Tao, Q., Joty, S., et al. (2018). VQA-E: Explaining, Elaborating, and Enhancing Your Answers for Visual Questions. *ECCV 2018*.

5. Merity, S., Keskar, N. S., & Socher, R. (2018). Regularizing and Optimizing LSTM Language Models. *ICLR 2018*. (AWD-LSTM)

6. Nguyen, D. K. & Okatani, T. (2018). Improved Fusion of Visual and Language Representations by Dense Symmetric Co-Attention for Visual Question Answering. *CVPR 2018*. (DCAN)

7. Oord, A. v. d., Li, Y., & Vinyals, O. (2018). Representation Learning with Contrastive Predictive Coding. *arXiv:1807.03748*. (InfoNCE)

8. Peters, M. E., Neumann, M., Iyyer, M., et al. (2018). Deep Contextualized Word Representations. *NAACL 2018*. (ELMo)

9. Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML 2021*. (CLIP/InfoNCE temperature)

10. Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. *NeurIPS 2015*.

11. Rennie, S. J., Marcheret, E., Mroueh, Y., et al. (2017). Self-Critical Sequence Training for Image Captioning. *CVPR 2017*. (SCST)

12. See, A., Liu, P. J., & Manning, C. D. (2017). Get To The Point: Summarization with Pointer-Generator Networks. *ACL 2017*.

13. Rohrbach, A., Hendricks, L. A., Burns, K., et al. (2018). Object Hallucination in Image Captioning. *EMNLP 2018*. (CHAIR metric)

14. Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization. *arXiv:1607.06450*.

15. Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. *EMNLP 2014*.
