# Visual Question Answering with Explanations (VQA-E)

## A Comparative Study of CNN-LSTM Architectures with and without Attention

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Dataset](#3-dataset)
4. [System Architecture Overview](#4-system-architecture-overview)
5. [Model Architectures](#5-model-architectures)
   - 5.1 [Model A — Scratch CNN + LSTM Decoder (No Attention)](#51-model-a--scratch-cnn--lstm-decoder-no-attention)
   - 5.2 [Model B — Pretrained ResNet101 + LSTM Decoder (No Attention)](#52-model-b--pretrained-resnet101--lstm-decoder-no-attention)
   - 5.3 [Model C — Scratch CNN Spatial + Bahdanau Attention + LSTM Decoder](#53-model-c--scratch-cnn-spatial--bahdanau-attention--lstm-decoder)
   - 5.4 [Model D — Pretrained ResNet101 Spatial + Bahdanau Attention + LSTM Decoder](#54-model-d--pretrained-resnet101-spatial--bahdanau-attention--lstm-decoder)
6. [Shared Components](#6-shared-components)
7. [Training Pipeline](#7-training-pipeline)
8. [Optimization Techniques](#8-optimization-techniques)
9. [Inference & Decoding](#9-inference--decoding)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Experimental Results](#11-experimental-results)
12. [Comparison & Analysis](#12-comparison--analysis)
13. [Conclusion](#13-conclusion)
14. [References](#14-references)

---

## 1. Introduction

Visual Question Answering (VQA) is a multi-modal task that requires a system to understand both visual content (an image) and natural language (a question), then produce a natural language answer. Unlike classification-based VQA systems that select from a fixed set of answers, this project adopts a **generative approach** where the LSTM decoder produces the answer **token by token**, enabling richer and more expressive outputs.

This report presents the design, implementation, and evaluation of **four VQA architectures** that span two design axes:

|  | No Attention | Bahdanau Attention |
|---|---|---|
| **Scratch CNN** | **Model A** | **Model C** |
| **Pretrained ResNet101** | **Model B** | **Model D** |

By systematically varying these two factors — (1) whether the CNN image encoder is trained from scratch or uses pretrained ImageNet weights, and (2) whether the decoder employs an attention mechanism — we can isolate and analyze the contribution of each component.

---

## 2. Problem Statement

**Task:** Given an image $I$ and a natural language question $Q$, generate a natural language answer $A$ consisting of a sequence of tokens $(a_1, a_2, \dots, a_T)$.

**Formal definition:**

$$A^* = \arg\max_{A} P(A \mid I, Q) = \arg\max_{A} \prod_{t=1}^{T} P(a_t \mid a_{<t}, I, Q)$$

The system must:
- Encode the image using a CNN
- Encode the question using an LSTM
- Fuse the image and question representations
- Decode the answer sequence token-by-token using an LSTM decoder

**Why VQA-E?** The original VQA 2.0 dataset produces very short answers (1–3 words like "yes", "no", "2"), making the LSTM decoder trivial and under-utilized. By adopting the **VQA-E dataset** (Li et al., 2018), the target output becomes a full sentence:

> *"yes because the man has glasses on his face"* (~10–25 tokens)

This provides a meaningful sequence generation task that fully leverages the LSTM decoder's capability.

---

## 3. Dataset

### 3.1 VQA-E Dataset

| Property | Train | Validation |
|---|---|---|
| **Samples** | 181,298 | 88,488 |
| **Images** | 82,783 (COCO train2014) | 40,504 (COCO val2014) |
| **Image size** | Resized to 224×224 | Resized to 224×224 |
| **Source** | VQA-E (Li et al., 2018) | VQA-E (Li et al., 2018) |

VQA-E extends VQA 2.0 by adding human-written explanations. Each sample contains:
- **Question:** A natural language question about the image
- **Answer:** The ground-truth short answer (e.g., "yes", "broccoli")
- **Explanation:** A human-written justification (e.g., "Closeup of bins of food that include broccoli and bread.")

The target sequence for the decoder is constructed as:

$$\text{target} = \texttt{<start>} \; \text{answer} \; \texttt{because} \; \text{explanation} \; \texttt{<end>}$$

For example: `<start> yes because the man has glasses on his face <end>`

### 3.2 Vocabulary

Two separate vocabularies are built from the VQA-E training set:

| Vocabulary | Size | Min frequency threshold |
|---|---|---|
| Question vocab ($V_Q$) | 4,546 | 3 |
| Answer vocab ($V_A$) | 8,648 | 3 |

**Special tokens:**

| Token | Index | Purpose |
|---|---|---|
| `<pad>` | 0 | Padding (ignored in loss) |
| `<start>` | 1 | Start of sequence |
| `<end>` | 2 | End of sequence |
| `<unk>` | 3 | Out-of-vocabulary |

### 3.3 Data Preprocessing

- **Images:** Resize to $224 \times 224$, convert to tensor, normalize with ImageNet statistics:

$$\text{normalize}(x) = \frac{x - \mu}{\sigma}, \quad \mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

- **Questions:** Lowercase, tokenize (word-level), convert to index sequence
- **Answers:** Lowercase, tokenize, wrap with `<start>` and `<end>` tokens
- **Batching:** Variable-length sequences padded with `<pad>` (index 0) using `pad_sequence`

### 3.4 Data Augmentation (Training only)

| Augmentation | Parameters |
|---|---|
| `RandomHorizontalFlip` | $p = 0.5$ |
| `ColorJitter` | brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05 |

---

## 4. System Architecture Overview

All four models share the same high-level pipeline:

```
┌─────────────────┐        ┌──────────────────────┐
│   IMAGE INPUT   │        │   QUESTION INPUT     │
│  (B, 3, 224, 224)│       │  (B, max_q_len)      │
└────────┬────────┘        └──────────┬───────────┘
         │                            │
┌────────▼────────┐        ┌──────────▼───────────┐
│   CNN ENCODER   │        │  BiLSTM Q-ENCODER    │
│   (varies by    │        │  (shared across all  │
│    model)       │        │   4 models)          │
└────────┬────────┘        └──────────┬───────────┘
         │                            │
    img_feature                  q_feature
         │                            │
         └──────────┬─────────────────┘
                    │
           ┌────────▼────────┐
           │  GATED FUSION   │
           │  gate = σ(W_g·[img;q])
           │  out = gate⊙tanh(W_img(img))
           │      + (1-gate)⊙tanh(W_q(q))
           └────────┬────────┘
                    │
               h₀ = fusion → repeat for num_layers
               c₀ = zeros
                    │
           ┌────────▼────────┐
           │  LSTM DECODER   │
           │  (A/B: no attn) │
           │  (C/D: + attn)  │
           └────────┬────────┘
                    │
           (B, seq_len, vocab_size)
```

### Key Design Decisions

1. **L2 Normalization** on image features before fusion — ensures direction matters, not magnitude
2. **Gated Fusion** instead of simple Hadamard product — a learnable gate decides how much image vs. question information to keep
3. **Teacher Forcing** during training — decoder receives ground-truth tokens as input
4. **Autoregressive decoding** during inference — decoder feeds its own predictions back

---

## 5. Model Architectures

### 5.1 Model A — Scratch CNN + LSTM Decoder (No Attention)

**Image Encoder: SimpleCNN**

A custom 5-layer CNN trained from scratch:

| Layer | Operation | Output Shape |
|---|---|---|
| Block 1 | Conv2d(3→64, k=3, p=1) → BN → ReLU → MaxPool(2) | $(B, 64, 112, 112)$ |
| Block 2 | Conv2d(64→128, k=3, p=1) → BN → ReLU → MaxPool(2) | $(B, 128, 56, 56)$ |
| Block 3 | Conv2d(128→256, k=3, p=1) → BN → ReLU → MaxPool(2) | $(B, 256, 28, 28)$ |
| Block 4 | Conv2d(256→512, k=3, p=1) → BN → ReLU → MaxPool(2) | $(B, 512, 14, 14)$ |
| Block 5 | Conv2d(512→1024, k=3, p=1) → BN → ReLU → MaxPool(2) | $(B, 1024, 7, 7)$ |
| Pool | AdaptiveAvgPool2d(1) | $(B, 1024, 1, 1)$ |
| FC | Linear(1024 → hidden_size) | $(B, 1024)$ |

**Output:** A single global image vector $(B, 1024)$.

**Decoder:** LSTMDecoder (no attention) — receives the fused representation as initial hidden state and generates tokens sequentially via teacher forcing.

**Characteristics:**
- Simplest baseline — no pretrained weights, no attention
- The entire image is compressed to a single 1024-dim vector
- Decoder has no mechanism to focus on specific image regions

---

### 5.2 Model B — Pretrained ResNet101 + LSTM Decoder (No Attention)

**Image Encoder: ResNetEncoder**

Uses a pretrained ResNet101 (ImageNet weights) with the final FC layer removed:

$$\text{ResNet101}[:-1] \rightarrow \text{Linear}(2048 \rightarrow 1024)$$

| Component | Details |
|---|---|
| Backbone | ResNet101 (pretrained on ImageNet) |
| Removed layers | Final FC layer (keeps avgpool) |
| Projection | Linear(2048 → 1024) |
| Initial state | `freeze=True` (all backbone parameters frozen) |
| Fine-tuning | `unfreeze_top_layers()` opens layer3 + layer4 (Phase 2) |

**Selective Fine-tuning (Phase 2):**
- Early layers (conv1, layer1, layer2): remain frozen — capture generic low-level features
- layer3 + layer4: unfrozen with a smaller learning rate ($\text{lr} \times 0.1$)
- Prevents catastrophic forgetting of pretrained ImageNet knowledge

**Output:** A single global image vector $(B, 1024)$.

**Decoder:** Same LSTMDecoder as Model A.

**Characteristics:**
- Leverages high-quality ImageNet features
- Significantly richer image representation than scratch CNN
- Still limited by single global vector (no spatial information for decoder)

---

### 5.3 Model C — Scratch CNN Spatial + Bahdanau Attention + LSTM Decoder

**Image Encoder: SimpleCNNSpatial**

Same 5-layer CNN as Model A, but **without** global average pooling:

| Layer | Output Shape |
|---|---|
| 5× conv_block | $(B, 1024, 7, 7)$ |
| Conv2d(1024→hidden, k=1) — per-region projection | $(B, \text{hidden}, 7, 7)$ |
| Flatten + Permute | $(B, 49, 1024)$ |

**Output:** $49$ spatial feature vectors $(B, 49, 1024)$, each representing a $32 \times 32$ pixel region.

**Decoder: LSTMDecoderWithAttention**

At each decode step $t$, the decoder performs **dual attention** — attending over both image regions and question hidden states:

**Step 1 — Image Attention (Bahdanau Additive Attention):**

$$e_{t,i}^{\text{img}} = \tanh(W_h \cdot h_{t-1} + W_{\text{img}} \cdot \text{img}_i)$$

$$\alpha_{t,i}^{\text{img}} = \text{softmax}(v^\top e_{t,i}^{\text{img}})$$

$$c_t^{\text{img}} = \sum_{i=1}^{49} \alpha_{t,i}^{\text{img}} \cdot \text{img}_i$$

**Step 2 — Question Attention:**

$$e_{t,j}^{q} = \tanh(W_h' \cdot h_{t-1} + W_q \cdot q_j)$$

$$\alpha_{t,j}^{q} = \text{softmax}(v'^\top e_{t,j}^{q})$$

$$c_t^{q} = \sum_{j=1}^{L_q} \alpha_{t,j}^{q} \cdot q_j$$

**Step 3 — LSTM Input:**

$$\text{input}_t = [\text{embed}(a_{t-1}) \; ; \; c_t^{\text{img}} \; ; \; c_t^{q}]$$

$$h_t, c_t = \text{LSTM}(\text{input}_t, h_{t-1}, c_{t-1})$$

$$P(a_t) = \text{softmax}(W_o \cdot \text{proj}(h_t))$$

Where $[\cdot ; \cdot]$ denotes concatenation, so the LSTM input size is $\text{embed\_size} + 2 \times \text{hidden\_size}$.

**Coverage Mechanism (Optional):**

To prevent the attention from repeatedly focusing on the same image regions, Model C supports an optional **Coverage Mechanism** (See et al., 2017):

$$\text{coverage}_t = \sum_{\tau=0}^{t-1} \alpha_\tau^{\text{img}}$$

The coverage vector is fed into the attention energy computation:

$$e_{t,i}^{\text{img}} = \tanh(W_h \cdot h_{t-1} + W_{\text{img}} \cdot \text{img}_i + W_{\text{cov}} \cdot \text{coverage}_{t,i})$$

Coverage loss penalizes re-attending:

$$\mathcal{L}_{\text{cov}} = \frac{1}{T} \sum_{t=1}^{T} \sum_{i=1}^{49} \alpha_{t,i} \cdot \log(\text{coverage}_{t,i} + 1)$$

**Total loss:**

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \cdot \mathcal{L}_{\text{cov}}$$

where $\lambda = 1.0$ by default.

**Characteristics:**
- Preserves spatial information (49 regions instead of 1 vector)
- Dual attention dynamically focuses on relevant image regions AND question words
- Much richer decoder input at each step
- Significantly more parameters in the decoder

---

### 5.4 Model D — Pretrained ResNet101 Spatial + Bahdanau Attention + LSTM Decoder

**Image Encoder: ResNetSpatialEncoder**

Uses ResNet101 pretrained on ImageNet, with both avgpool and FC removed to preserve spatial feature maps:

$$\text{ResNet101}[:-2] \rightarrow \text{Conv2d}(2048 \rightarrow 1024, k=1) \rightarrow \text{reshape} \rightarrow (B, 49, 1024)$$

| Component | Details |
|---|---|
| Backbone | ResNet101 (pretrained, `[:-2]`) |
| Projection | Conv2d(2048→1024, kernel=1) — per-region |
| Fine-tuning | Same as Model B (`unfreeze_top_layers()` for Phase 2) |

**Output:** $49$ spatial feature vectors $(B, 49, 1024)$.

**Decoder:** Same `LSTMDecoderWithAttention` as Model C (dual attention + optional coverage).

**Characteristics:**
- Combines the best of both worlds: high-quality pretrained features + attention mechanism
- Expected to achieve the best performance among all 4 models
- Highest computational cost (large backbone + step-by-step attention loop)

---

## 6. Shared Components

### 6.1 Question Encoder (BiLSTM)

All four models share the same **Bidirectional LSTM** question encoder:

$$\text{Embedding}(V_Q, d_{\text{embed}}) \rightarrow \text{BiLSTM}(\text{hidden\_size} // 2 \text{ per direction}) \rightarrow h_{\text{final}}$$

| Parameter | Value |
|---|---|
| Embedding dim | 512 (or 300 if GloVe, projected to 512) |
| LSTM hidden size | 512 per direction (1024 total) |
| Num layers | 2 |
| Dropout | 0.5 (inter-layer) |
| Bidirectional | Yes |

**Output:**
- $q_{\text{feature}} = [\overrightarrow{h_L} \; ; \; \overleftarrow{h_L}] \in \mathbb{R}^{1024}$ — for fusion
- $q_{\text{hidden\_states}} \in \mathbb{R}^{B \times L_q \times 1024}$ — for question attention (Model C/D)

### 6.2 Gated Fusion

Instead of simple element-wise multiplication (Hadamard product), a **Gated Fusion** module learns to combine information:

$$h_{\text{img}} = \tanh(W_{\text{img}} \cdot f_{\text{img}})$$

$$h_q = \tanh(W_q \cdot f_q)$$

$$g = \sigma(W_g \cdot [f_{\text{img}} \; ; \; f_q])$$

$$\text{fusion} = g \odot h_{\text{img}} + (1 - g) \odot h_q$$

where $g \in [0, 1]^{d}$ is a learned gate vector, and $\sigma$ is the sigmoid function.

### 6.3 GloVe Pretrained Embeddings

Both the question encoder and answer decoder support optional **GloVe 6B 300d** pretrained embeddings:

- Words found in GloVe → use pretrained vectors (fine-tuned during training)
- Words not found (OOV) → randomly initialized from $\mathcal{N}(0, 0.1)$
- `<pad>` embedding → zero vector
- When GloVe dim (300) ≠ embed_size (512), a learned linear projection is added

**Coverage:** ~99.6% of answer vocabulary words are covered by GloVe.

### 6.4 Weight Tying

The decoder output layer shares weights with the embedding layer (Press & Wolf, 2017):

$$\text{hidden} \xrightarrow{\text{out\_proj}} \mathbb{R}^{d_{\text{embed}}} \xrightarrow{W_{\text{embed}}^\top} \mathbb{R}^{|V_A|}$$

This reduces the number of parameters and acts as a regularizer by constraining the output distribution to be consistent with the input embedding space.

> **Note:** When GloVe embeddings are used (dim=300), weight tying is disabled to avoid a severe bottleneck ($1024 \rightarrow 300 \rightarrow 8648$).

---

## 7. Training Pipeline

### 7.1 Three-Phase Training Strategy

All 4 models are trained under identical conditions to ensure a fair comparison:

| Phase | Epochs | Learning Rate | Key Technique | Purpose |
|---|---|---|---|---|
| **Phase 1** — Baseline | 10 | $1 \times 10^{-3}$ | Teacher Forcing, ResNet frozen | Decoder + Q-Encoder convergence |
| **Phase 2** — Fine-tune | 5 | $5 \times 10^{-4}$ | Unfreeze ResNet L3+L4 (B/D) | Adapt pretrained features to VQA |
| **Phase 3** — Scheduled Sampling | 5 | $2 \times 10^{-4}$ | Replace GT with model predictions | Reduce exposure bias |

**Total: 20 epochs per model.**

### 7.2 Teacher Forcing

During training, the decoder receives the ground-truth previous token as input:

$$\text{decoder\_input} = \text{answer}[:, :-1] = [\texttt{<start>}, w_1, w_2, \dots, w_{n}]$$

$$\text{decoder\_target} = \text{answer}[:, 1:] = [w_1, w_2, \dots, w_{n}, \texttt{<end>}]$$

$$\mathcal{L}_{\text{CE}} = \text{CrossEntropyLoss}(\text{logits}, \text{target}), \quad \text{ignore\_index} = 0 \; (\texttt{<pad>})$$

### 7.3 Scheduled Sampling (Phase 3)

To bridge the gap between training (teacher forcing) and inference (autoregressive), **Scheduled Sampling** (Bengio et al., 2015) gradually replaces ground-truth tokens with the model's own predictions:

At each decode step $t$, with probability $\epsilon$, use the ground-truth token; with probability $(1 - \epsilon)$, use $\arg\max(\text{logit}_{t-1})$.

The probability follows an **inverse-sigmoid decay**:

$$\epsilon(\text{epoch}) = \frac{k}{k + \exp(\text{epoch} / k)}, \quad k = 5$$

This starts near 1.0 (pure teacher forcing) and gradually decreases, forcing the model to recover from its own mistakes.

### 7.4 Learning Rate Schedule

**LR Warmup (Phase 1):** Linear warmup from $\text{lr}/10$ to $\text{lr}$ over the first 3 epochs:

$$\text{lr}(e) = \text{lr}_{\text{base}} \times \left(0.1 + 0.9 \times \frac{e}{3}\right), \quad e \in [1, 3]$$

**Cosine Annealing (after warmup):** Smooth decay from peak LR to $\eta_{\min} = 0.01 \times \text{lr}$:

$$\text{lr}(e) = \eta_{\min} + \frac{1}{2}(\text{lr}_{\text{base}} - \eta_{\min})\left(1 + \cos\left(\frac{e - e_{\text{warmup}}}{T_{\max}} \pi\right)\right)$$

### 7.5 Differential Learning Rate (Phase 2, Models B/D)

When fine-tuning the ResNet backbone:

| Parameter Group | Learning Rate | Purpose |
|---|---|---|
| Decoder + Q-Encoder + Fusion | $5 \times 10^{-4}$ | Adapt quickly |
| ResNet layer3 + layer4 | $5 \times 10^{-5}$ ($\text{lr} \times 0.1$) | Preserve pretrained knowledge |

### 7.6 Batch Sizes (RTX 3060 12GB VRAM)

| Model | Batch Size | Accumulation Steps | Effective Batch |
|---|---|---|---|
| A (SimpleCNN) | 64 | 2 | 128 |
| B (ResNet101) | 32 | 4 | 128 |
| C (SimpleCNN Spatial + Attn) | 32 | 2 | 64 |
| D (ResNet101 Spatial + Attn) | 16 | 4 | 64 |

---

## 8. Optimization Techniques

### 8.1 Regularization

| Technique | Configuration | Description |
|---|---|---|
| **Label Smoothing** | 0.1 | Softens one-hot targets to prevent overconfidence |
| **Weight Decay** | $1 \times 10^{-5}$ | L2 regularization on all parameters |
| **Embedding Dropout** | 0.5 | Applied after embedding layer in both encoder and decoder |
| **LSTM Inter-layer Dropout** | 0.5 | Between stacked LSTM layers |
| **Gradient Clipping** | max_norm = 5.0 | Prevents exploding gradients |
| **Early Stopping** | patience = 5 | Halts training when validation loss stops improving |

### 8.2 Mixed Precision Training (AMP)

- **Ampere+ GPUs** (compute capability ≥ 8.0): BFloat16 — wider dynamic range, no GradScaler needed
- **Older GPUs**: Float16 + GradScaler — standard mixed precision fallback
- Automatic detection via `torch.cuda.get_device_capability()`

### 8.3 Gradient Accumulation

For models that require small batch sizes due to VRAM constraints, gradient accumulation simulates larger effective batches:

$$\text{effective batch} = \text{batch\_size} \times \text{accum\_steps}$$

Loss is divided by `accum_steps` before `backward()`, and `optimizer.step()` is called every `accum_steps` mini-batches.

### 8.4 GPU Optimizations

| Optimization | Description |
|---|---|
| `cudnn.benchmark = True` | Auto-tune convolution algorithms for fixed input sizes |
| TF32 matmul & convolutions | ~2× faster on Ampere+, near-FP32 accuracy |
| Fused Adam optimizer | Reduces kernel launch overhead (~10–20% faster) |
| `pin_memory=True` | Faster CPU→GPU data transfer |
| `persistent_workers=True` | Avoid DataLoader worker respawn overhead |
| `prefetch_factor=4` | Pre-load next batches while GPU is computing |

---

## 9. Inference & Decoding

### 9.1 Greedy Decoding

At each step, select the token with the highest probability:

$$a_t = \arg\max_{w \in V_A} P(w \mid a_{<t}, I, Q)$$

- Fast single-pass decoding
- May miss globally optimal sequences

### 9.2 Beam Search

Maintains the top-$k$ candidate sequences at each step:

1. Expand each beam by all vocabulary tokens
2. Score each candidate: cumulative log probability
3. Keep top-$k$ candidates
4. Return the sequence with the highest **length-normalized** score:

$$\text{score}(A) = \frac{1}{|A|} \sum_{t=1}^{|A|} \log P(a_t \mid a_{<t}, I, Q)$$

### 9.3 N-gram Blocking

To prevent repetitive output during beam search, trigram blocking sets $\log P(w) = -\infty$ for any token $w$ that would create a repeated n-gram (default: $n = 3$).

---

## 10. Evaluation Metrics

The following metrics are used to evaluate the generative output quality:

| Metric | Type | Description | Expected Range |
|---|---|---|---|
| **BLEU-4** ★ | N-gram precision | Measures 4-gram overlap between prediction and reference | 0.05 – 0.20 |
| **METEOR** ★ | Semantic matching | N-gram + synonym matching via WordNet | 0.10 – 0.30 |
| **BERTScore** ★ | Semantic similarity | Cosine similarity of BERT contextual embeddings | 0.40 – 0.70 |
| BLEU-1 | Unigram precision | Word-level overlap | Reference |
| BLEU-2 | Bigram precision | Phrase-level overlap | Reference |
| BLEU-3 | Trigram precision | Longer phrase overlap | Reference |
| Exact Match | String equality | Strict match — very low for generative tasks | < 5% |

★ = Primary metrics for evaluation and comparison.

> **Note:** Traditional VQA Accuracy (classification-based) has been intentionally excluded as it is not suitable for evaluating generative outputs of varying length and structure.

---

## 11. Experimental Results

### 11.1 Training Curves

<!-- TODO: Insert training/validation loss curve figure here -->
<!-- Run: python src/plot_curves.py -->

> **[To be completed after training]**

### 11.2 Phase 1 Results (After Epoch 10)

| Model | BLEU-4 | METEOR | BERTScore | BLEU-1 | BLEU-2 | BLEU-3 | Exact Match | Checkpoint |
|---|---|---|---|---|---|---|---|---|
| A |  |  |  |  |  |  |  | model_a_epoch10.pth |
| B |  |  |  |  |  |  |  | model_b_epoch10.pth |
| C |  |  |  |  |  |  |  | model_c_epoch10.pth |
| D |  |  |  |  |  |  |  | model_d_epoch10.pth |

### 11.3 Phase 2 Results (After Epoch 15)

| Model | BLEU-4 | METEOR | BERTScore | BLEU-1 | BLEU-2 | BLEU-3 | Exact Match | Checkpoint |
|---|---|---|---|---|---|---|---|---|
| A |  |  |  |  |  |  |  | model_a_epoch15.pth |
| B |  |  |  |  |  |  |  | model_b_epoch15.pth |
| C |  |  |  |  |  |  |  | model_c_epoch15.pth |
| D |  |  |  |  |  |  |  | model_d_epoch15.pth |

### 11.4 Phase 3 Results — Final (After Epoch 20)

| Model | BLEU-4 | METEOR | BERTScore | BLEU-1 | BLEU-2 | BLEU-3 | Exact Match | Checkpoint |
|---|---|---|---|---|---|---|---|---|
| A |  |  |  |  |  |  |  | model_a_epoch20.pth |
| B |  |  |  |  |  |  |  | model_b_epoch20.pth |
| C |  |  |  |  |  |  |  | model_c_epoch20.pth |
| D |  |  |  |  |  |  |  | model_d_epoch20.pth |

### 11.5 Beam Search Results (Final Model, beam_width=5)

| Model | BLEU-4 | METEOR | BERTScore | Decode Mode |
|---|---|---|---|---|
| A |  |  |  | beam (width=5) |
| B |  |  |  | beam (width=5) |
| C |  |  |  | beam (width=5) |
| D |  |  |  | beam (width=5) |

### 11.6 Qualitative Examples

<!-- TODO: Add sample predictions with images -->
<!-- Run: python src/visualize.py --model_type D -->

> **[To be completed after training]**

### 11.7 Attention Visualization (Model C/D)

<!-- TODO: Insert attention heatmap figures -->
<!-- Run: python src/visualize.py --model_type C --output checkpoints/attn_model_c.png -->
<!-- Run: python src/visualize.py --model_type D --output checkpoints/attn_model_d.png -->

> **[To be completed after training]**

---

## 12. Comparison & Analysis

### 12.1 Effect of Pretrained Features (A vs B, C vs D)

<!-- TODO: Analyze the performance gap between scratch CNN and pretrained ResNet101 -->

> **[To be completed after training]**

### 12.2 Effect of Attention Mechanism (A vs C, B vs D)

<!-- TODO: Analyze the contribution of Bahdanau attention -->

> **[To be completed after training]**

### 12.3 Progressive Training Analysis (Phase 1 → 2 → 3)

<!-- TODO: Analyze how each training phase improves metrics -->

> **[To be completed after training]**

### 12.4 Greedy vs Beam Search

<!-- TODO: Compare greedy and beam search decoding performance -->

> **[To be completed after training]**

### 12.5 Error Analysis

<!-- TODO: Analyze common failure cases, error patterns by question type -->

> **[To be completed after training]**

---

## 13. Conclusion

<!-- TODO: Summarize findings after training -->

> **[To be completed after training]**

This project designed and implemented four VQA architectures that systematically vary across two design axes: CNN image encoder (scratch vs. pretrained) and decoder strategy (no attention vs. Bahdanau attention). The VQA-E dataset was adopted to create a meaningful generative task where the LSTM decoder produces full-sentence answers with explanations.

Key architectural contributions include:
- **Gated Fusion** for adaptive multimodal combination
- **Dual Attention** (image + question) for richer contextual decoding
- **Coverage Mechanism** to reduce repetitive generation
- **BiLSTM Question Encoder** with GloVe pretrained embeddings
- **Weight Tying** between decoder embedding and output layers

The three-phase progressive training strategy (teacher forcing → fine-tuning → scheduled sampling) combined with extensive regularization techniques provides a robust and fair framework for comparing the four architectures.

---

## 14. References

1. **Antol, S., et al.** (2015). "VQA: Visual Question Answering." *ICCV 2015.*
2. **Li, Q., et al.** (2018). "VQA-E: Explaining, Elaborating, and Enhancing Your Answers for Visual Questions." *ECCV 2018.*
3. **He, K., et al.** (2016). "Deep Residual Learning for Image Recognition." *CVPR 2016.*
4. **Bahdanau, D., Cho, K., Bengio, Y.** (2015). "Neural Machine Translation by Jointly Learning to Align and Translate." *ICLR 2015.*
5. **Bengio, S., et al.** (2015). "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks." *NeurIPS 2015.*
6. **See, A., Liu, P.J., Manning, C.D.** (2017). "Get To The Point: Summarization with Pointer-Generator Networks." *ACL 2017.*
7. **Press, O., Wolf, L.** (2017). "Using the Output Embedding to Improve Language Models." *EACL 2017.*
8. **Pennington, J., Socher, R., Manning, C.D.** (2014). "GloVe: Global Vectors for Word Representation." *EMNLP 2014.*
9. **Papineni, K., et al.** (2002). "BLEU: a Method for Automatic Evaluation of Machine Translation." *ACL 2002.*
10. **Banerjee, S., Lavie, A.** (2005). "METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments." *ACL Workshop 2005.*
11. **Zhang, T., et al.** (2020). "BERTScore: Evaluating Text Generation with BERT." *ICLR 2020.*
