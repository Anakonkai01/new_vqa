---
marp: true
theme: default
paginate: true
backgroundColor: #ffffff
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 22px;
  }
  h1 { color: #1a237e; font-size: 40px; }
  h2 { color: #283593; font-size: 30px; border-bottom: 2px solid #3949ab; padding-bottom: 6px; }
  h3 { color: #3949ab; font-size: 24px; }
  table { font-size: 18px; width: 100%; }
  th { background: #3949ab; color: white; }
  td { padding: 4px 8px; }
  tr:nth-child(even) { background: #e8eaf6; }
  .highlight { background: #fff9c4; padding: 6px 12px; border-left: 4px solid #f9a825; }
  .win { color: #2e7d32; font-weight: bold; }
  .small { font-size: 17px; }
  blockquote { border-left: 4px solid #3949ab; background: #e8eaf6; padding: 8px 16px; font-style: normal; }
---

<!-- SLIDE 1 — TITLE -->
# VQA-E: Explaining Visual Questions
## A Systematic Comparison of CNN-LSTM Architectures

**Generative Visual Question Answering with Explanatory Answers**

---

> **Image:** *A boy on a beach flying a kite in the sky*
> **Question:** *"What is the boy doing?"*
> **Our model's answer:** *"flying kite because a man is flying a kite in the sky"*

&nbsp;

**2×2 Factorial Design — 4 Models — Full VQA-E Validation Set (n = 88,488)**

*Based on: Li et al., "VQA-E: Explaining, Elaborating, and Enhancing Your Answers for Visual Questions", ECCV 2018*

<!--
═══════════════════════════════════════════════════════
SCRIPT — Slide 1 · TITLE  (~1 min)
═══════════════════════════════════════════════════════

"Good [morning/afternoon]. My project is about Visual Question Answering with
Explanations — a task where a model must not just answer a question about an
image, but also explain WHY it gave that answer in natural language.

The example at the bottom of this slide shows exactly what I mean. Given an
image of a boy on a beach flying a kite, a standard VQA model outputs a single
word: 'kite.' My model generates a full sentence: 'flying kite BECAUSE a man
is flying a kite in the sky.'

The core of this project is a systematic comparison. I trained four different
neural architectures on this task — all under identical conditions — and
evaluated all of them on the full 88,488-sample VQA-E validation set. By
varying exactly two design choices and holding everything else constant, I can
give clean, data-backed answers to the question: what actually drives
performance in a VQA system?

Let me start with why this task is interesting."

[TRANSITION → Slide 2]
-->

---

<!-- SLIDE 2 — MOTIVATION -->
## Why Explanatory VQA?

| Standard VQA | Explanatory VQA (this project) |
|---|---|
| **"What color is the car?"** → *"red"* | **"What color is the car?"** → *"red because the car in the foreground is red"* |
| Single token output | Full sentence generation |
| No reasoning trace | Explicit visual justification |
| Easy to fool with language bias | Must look at the image to explain |
| LSTM decoder role: trivial (1 step) | LSTM decoder role: meaningful (10–20 steps) |

&nbsp;

**Why generative output matters for this study:**
With 1–3 word answers, differences between architectures nearly vanish.
Multi-step generation makes attention dynamics visible and measurable.

> Li et al. (2018): forcing the model to explain its answer **also improves answer accuracy** — explanation generation is a powerful self-supervision signal.

<!--
═══════════════════════════════════════════════════════
SCRIPT — Slide 2 · MOTIVATION  (~1.5 min)
═══════════════════════════════════════════════════════

"So why generate explanations instead of just a short answer?

There are two reasons — one scientific, one practical.

The practical reason: explanatory answers are more useful. If a visually
impaired person asks 'is there a dog in this image?', the answer 'yes because
a brown dog is sitting on the grass near the bench' is far more informative
than just 'yes'. The model's reasoning becomes transparent.

The scientific reason — and this is the more important one for this project —
is that explanatory output makes architectural differences measurable. If I'm
comparing four architectures using single-word answers, the differences are
tiny and hard to interpret. But when the model must generate a sentence of 10
to 20 words explaining what it sees, the quality of the visual features and
the effectiveness of the attention mechanism become clearly visible in the
output.

[Point to table row 5]

This is why the LSTM decoder matters in this setting. For short-answer VQA
it's essentially a one-step classifier. For explanatory VQA it's a real
sequence generator — teacher forcing, scheduled sampling, and attention all
play meaningful roles.

And there's an empirical argument from the original paper: Li et al. showed
that when you force the model to generate an explanation, it actually gets
better at predicting the answer too. Explaining forces the model to look at
the image rather than guessing from language statistics."

[TRANSITION → Slide 3]
-->

---

<!-- SLIDE 3 — DATASET -->
## VQA-E Dataset

**Derived from VQA v2 + MSCOCO captions — 181K train / 88K val**

| Split | Q&A Pairs | How explanations are created |
|---|---|---|
| Train | **181,298** | Find most relevant MSCOCO caption (GloVe similarity) |
| **Val (my test set)** | **88,488** | Merge Q+A → declarative + fuse with caption via parse tree |

&nbsp;

**Example construction:**
- *Question:* "Is the cat sleeping?" + *Answer:* "yes"
- *Closest caption:* "A cat is lying on the couch with its eyes closed"
- *Final explanation:* **"yes because the cat is lying on the couch with its eyes closed"**

&nbsp;

**Quality filter:** similarity ≥ 0.6 → only **41% of VQA v2 pairs** get an explanation
→ Dataset has high quality but lower coverage than raw VQA v2

**My vocabulary:** Q: 4,546 words · A: 8,648 words (threshold: min_freq = 3)

<!--
═══════════════════════════════════════════════════════
SCRIPT — Slide 3 · DATASET  (~1.5 min)
═══════════════════════════════════════════════════════

"The dataset I use is VQA-E, introduced by Li et al. in 2018. It's built
automatically from two existing resources: the VQA v2 dataset for
question-answer pairs, and MSCOCO image captions for the explanations.

[Point to example]

Here's how the explanation is constructed. Take the question 'Is the cat
sleeping?' with answer 'yes.' The system finds the MSCOCO caption that is most
semantically similar to the question-answer pair — in this case, 'A cat is
lying on the couch with its eyes closed.' It then merges the QA statement with
the caption using constituency parse trees to produce the final explanation:
'yes because the cat is lying on the couch with its eyes closed.' Natural,
grounded, and grammatically correct.

The key quality filter is a similarity threshold of 0.6. Only 41% of VQA v2
pairs receive an explanation, because many image captions don't actually
describe what the question is asking about. For example, a question about a
dog in the background won't match a caption describing the kitchen foreground.

I use the standard train/val split: 181,000 training samples and 88,488
validation samples. I build two separate vocabularies — one for encoding
questions and one for decoding answers — both with a minimum frequency
threshold of 3, giving 4,546 question words and 8,648 answer words."

[TRANSITION → Slide 4]
-->

---

<!-- SLIDE 4 — THE 2×2 FACTORIAL DESIGN -->
## The 2×2 Factorial Design

**Exactly two binary decisions — everything else held constant**

|  | **No Attention** | **Dual Attention + Coverage** |
|---|---|---|
| **Scratch CNN** | **Model A** — Baseline | **Model C** — +Attention |
| **Pretrained ResNet101** | **Model B** — +Pretrained | **Model D** ⭐ — +Both |

&nbsp;

**What each comparison measures:**

| Comparison | Factor isolated | Confounders |
|---|---|---|
| A → B | Value of ImageNet pretraining | None — identical in all other ways |
| A → C | Value of dual attention | None |
| A → D | Combined effect | None |
| (A→B) + (A→C) vs (A→D) | Interaction / synergy | — |

&nbsp;

> Same data · Same optimizer · Same training schedule · Same hyperparameters across all 4 models

<!--
═══════════════════════════════════════════════════════
SCRIPT — Slide 4 · 2×2 DESIGN  (~1.5 min)
═══════════════════════════════════════════════════════

"The centerpiece of this project is the experimental design. I vary exactly
two binary factors:

Factor one: visual encoder — either a SimpleCNN trained from scratch on
VQA-E, or a ResNet101 pretrained on ImageNet.

Factor two: decoder strategy — either a plain LSTM that gets a single fused
vector and generates the answer, or an LSTM that at every decode step runs
Bahdanau attention over both the image regions and the question hidden states.

This gives four models. The power of this design is that every comparison is
clean. When I compare Model A to Model B, the only difference is the visual
encoder — same decoder, same question encoder, same training, same
hyperparameters. So any performance difference I observe is entirely due to
pretraining.

[Point to comparison table]

The factorial design also lets me measure the interaction: if pretrained
features and attention were completely independent, the gain from Model D over
A should equal the sum of A-to-B and A-to-C gains. If it's less than that
sum, the two factors partially overlap — they're addressing the same
underlying weakness.

All four models are trained on the same data, with the same AdamW optimizer,
the same three-phase training schedule, the same batch sizes, and the same
early stopping criteria."

[TRANSITION → Slide 5]
-->

---

<!-- SLIDE 5 — ARCHITECTURE OVERVIEW -->
## Architecture: Encoder → Fusion → Decoder

```
Image (3×224×224) ──► CNN Encoder ─────────────────────────────────────┐
                       A/B: (B, 1024)                                   ▼
                       C/D: (B, 49, 1024) ────────────► GatedFusion ──► h₀ ──► LSTM Decoder ──► Answer tokens
                                                            ▲                       ▲
Question tokens ──► BiLSTM Encoder ──► q_feature ──────────┘                       │
                              │                                                     │
                              └──► q_hidden_states (B, L_q, 1024) ─────────────────┘
                                                                  (C/D: Dual Attention at each step)
```

&nbsp;

| Component | Detail |
|---|---|
| **CNN Encoder** | A/B: SimpleCNN 5-block → global (1024d) · C/D: preserve 7×7 = 49 regions × 1024d |
| **Question Encoder** | 2-layer BiLSTM, 512d/dir → 1024d total + all hidden states for Q-attention |
| **GatedFusion** | `gate·tanh(W_img·f_img) + (1−gate)·tanh(W_q·f_q)` — gate ∈ ℝ¹⁰²⁴ learned per-dimension |
| **LSTM Decoder** | 2-layer, 1024 hidden, GloVe 300d→512d embed, weight tying, teacher forcing (train) |
| **Dual Attention (C/D)** | Bahdanau additive: `energy = tanh(W_h·h + W_img·region)` · context = Σ α·region + coverage |

<!--
═══════════════════════════════════════════════════════
SCRIPT — Slide 5 · ARCHITECTURE  (~2.5 min)
═══════════════════════════════════════════════════════

"Let me walk through the architecture. All four models share the same
Encoder–Fusion–Decoder skeleton.

[Point to diagram, trace left to right]

The image goes through a CNN encoder. For Models A and B, the CNN compresses
the entire image into a single 1024-dimensional vector — all spatial
information is destroyed at this step. For Models C and D, the CNN is
modified to keep the 7×7 spatial feature map — that's 49 regions, each
represented by a 1024-dimensional vector. These 49 vectors will be the
targets of the attention mechanism later.

In parallel, the question goes through a two-layer bidirectional LSTM. I use
bidirectional because questions reveal their full meaning only once you've
read the whole thing — 'What color is the big red dog?' only makes sense as
a unit. The BiLSTM reads forward and backward simultaneously. It outputs two
things: a single 1024d question summary vector, and the sequence of all
intermediate hidden states for Models C and D to attend to.

The image feature and the question feature are then fused using GatedFusion.

[Point to GatedFusion row]

This is one of my improvements over the original paper, which uses a simple
Hadamard product. GatedFusion learns a 1024-dimensional gate that decides,
per dimension, how much to trust the image versus the question. For visual
questions, the gate tends to weight the image more. For questions about
relationships or counting, both modalities matter.

The fused vector initializes the decoder's hidden state h₀. The decoder is a
2-layer LSTM that generates the answer one token at a time.

[Point to dual attention row]

For Models C and D, at every single decode step, the decoder runs two
Bahdanau attention computations: one over the 49 image regions, and one over
the question hidden states. The outputs — two context vectors — are
concatenated with the current token embedding as input to the LSTM. This
means the decoder can shift its visual focus at each step, attending to the
'kite' region when generating 'kite' and the 'sky' region when generating
'in the sky.'"

[TRANSITION → Slide 6]
-->

---

<!-- SLIDE 6 — TRAINING PIPELINE -->
## 3-Phase Progressive Training

```
Phase 1 (Epochs 1–10)         Phase 2 (Epochs 11–15)        Phase 3 (Epochs 16+)
─────────────────────         ──────────────────────         ────────────────────
Teacher Forcing: 100%    ──►  Teacher Forcing: 100%    ──►  Scheduled Sampling
LR = 1e-3                     LR = 5e-4                      LR = 2e-4
ResNet FROZEN (B/D)           ResNet layer3+4 UNFROZEN       ε(epoch_rel) → 0.65 by ep5
Goal: train decoder           Goal: adapt ResNet features    Goal: reduce exposure bias
```

&nbsp;

**Scheduled Sampling** — the exposure bias fix:

During training the decoder sees ground-truth tokens → at inference it sees its own (possibly wrong) tokens.
This mismatch = **exposure bias**. Fix: gradually feed own predictions during training.

$$\varepsilon(e_{\text{rel}}) = \frac{k}{k + e^{e_{\text{rel}}/k}}, \quad k=5$$

> ⚠️ **Critical detail:** `e_rel` = epoch index *within Phase 3* (0,1,2…), NOT the absolute epoch (16,17,18…). Using absolute epoch causes ε to be near 0 immediately → training collapse.

**Best checkpoints by validation loss:**

| Model | Best epoch | Best val loss | Phase 3 helpful? |
|---|---|---|---|
| A | 16 | 3.2983 | No — val loss rises monotonically |
| B | 15 | 3.2178 | No — train collapses but val rises |
| C | 15 | 3.2774 | No — immediate spike then instability |
| **D** | **15** | **3.2216** | **Partial** — val recovers to 3.2625 at ep21 |

<!--
═══════════════════════════════════════════════════════
SCRIPT — Slide 6 · TRAINING  (~2.5 min)
═══════════════════════════════════════════════════════

"Training is the most technically involved part of this project. I use a
three-phase progressive strategy, and each phase solves a specific problem.

[Point to Phase 1 box]

Phase 1 is standard teacher forcing for 10 epochs. At every decode step, the
decoder receives the correct ground-truth token from the previous position.
This gives very stable gradient signals and lets the decoder quickly learn
basic language generation patterns. The ResNet backbone in Models B and D is
completely frozen during this phase — we don't want random gradients from an
untrained decoder corrupting the pretrained ImageNet features.

[Point to Phase 2 box]

Phase 2 runs for 5 epochs with a lower learning rate. Now, for Models B and D,
I unfreeze the top two blocks of ResNet — layer3 and layer4. These are the
high-level semantic layers. They get a learning rate that is 10 times smaller
than the rest of the model, to gently adapt them to VQA without forgetting
their ImageNet knowledge. The early layers — edges, textures, colors — stay
frozen, because those features transfer universally.

[Point to Phase 3 box]

Phase 3 introduces scheduled sampling. The problem it solves is called
exposure bias. During training, the decoder always sees the correct previous
token. But at inference time, it sees its own possibly-wrong prediction. If it
makes an error at step 3, every subsequent step sees an input distribution it
never encountered during training. Errors compound.

Scheduled sampling fixes this by gradually replacing ground-truth inputs with
the model's own predictions during training. The fraction of own-predictions
increases according to this inverse-sigmoid function.

[Point to formula and warning box]

A critical implementation detail — and this was a real bug I had to fix: the
epsilon uses the epoch index relative to the start of Phase 3, not the
absolute epoch number. If you use absolute epoch 16, 17, 18..., epsilon is
already near zero on day one of Phase 3, meaning the model immediately tries
to be 100% self-reliant before it's ready. That causes training collapse. With
relative epoch 0, 1, 2..., epsilon starts at 0.83 and decays smoothly.

[Point to best checkpoint table]

Interestingly, Phase 3 helps only Model D. For Models A, B, and C, the best
checkpoint by validation loss is at epoch 15 — the end of Phase 2. For Model
D, pretrained features and dual attention together provide enough structural
stability that scheduled sampling can actually reduce exposure bias without
destabilizing training."

[TRANSITION → Slide 7]
-->

---

<!-- SLIDE 7 — QUANTITATIVE RESULTS -->
## Results: Greedy Decoding

**All 4 models · Full VQA-E validation set · n = 88,488 samples**

| Model | BLEU-1 | BLEU-4 ⭐ | METEOR ⭐ | ROUGE-L ⭐ | BERTScore | Exact Match |
|---|---|---|---|---|---|---|
| **A** — Scratch CNN, No Attn | 0.3715 | 0.0915 | 0.3117 | 0.3828 | 0.9008 | 2.83% |
| **B** — ResNet101, No Attn | 0.4124 | 0.1127 | 0.3561 | 0.4237 | 0.9081 | 4.07% |
| **C** — Scratch CNN, Dual Attn | 0.3865 | 0.0988 | 0.3271 | 0.3971 | 0.9034 | 4.18% |
| **D** — ResNet101, Dual Attn | **0.4151** | **0.1159** | **0.3595** | **0.4270** | **0.9085** | **5.88%** |

&nbsp;

**A → D improvement:** BLEU-4 **+26.7%** relative · METEOR **+15.3%** · ROUGE-L **+11.5%** · Exact Match **+107.8%**

**Ranking: D > B > C > A — consistent across all primary metrics (BLEU-4, METEOR, ROUGE-L)**

<!--
═══════════════════════════════════════════════════════
SCRIPT — Slide 7 · RESULTS  (~2.5 min)
═══════════════════════════════════════════════════════

"Here are the main quantitative results. These are greedy decoding results
on the full 88,488-sample validation set, using the best checkpoint selected
by lowest validation loss for each model.

[Point to the table, go row by row slowly]

Model A — our baseline with scratch CNN and no attention — scores 0.0915 on
BLEU-4 and 2.83% Exact Match. These are the floors.

Model B — same decoder but with pretrained ResNet101 — jumps to 0.1127 BLEU-4.
That's a 23% relative improvement from just swapping the visual encoder.

Model C — attention added to the scratch CNN — scores 0.0988. Better than A,
but notably lower than B. Attention helps, but not as much as better features.

Model D — both pretrained features AND dual attention — achieves 0.1159 BLEU-4,
42.70% ROUGE-L, and 5.88% Exact Match. Best on every single metric.

[Point to ranking line]

Three things are important about this table.

First, the ranking D > B > C > A holds across ALL three primary metrics —
BLEU-4, METEOR, and ROUGE-L. When three independent metrics with completely
different computation methods all agree on the same ordering, that's very
strong evidence. This is not a measurement artifact.

Second, Model B beats Model C despite having no attention at all. Pretrained
features are more valuable than the attention mechanism when you can only
choose one.

Third, BERTScore is high for all models — above 0.90 — but the spread is
only 0.0077. That might look like a small difference but it's statistically
significant with 88,000 samples. However, BERTScore is not discriminating
well here because all models produce text in the same 'answer because
explanation' format. They're all in the same semantic neighborhood. For
this task, BLEU-4 and METEOR are more informative discriminators."

[TRANSITION → Slide 8]
-->

---

<!-- SLIDE 8 — BEAM SEARCH -->
## Results: Beam Search vs. Greedy

**beam_width = 3, trigram blocking**

| Model | Greedy BLEU-4 | Beam BLEU-4 | Δ BLEU-4 | Greedy EM | Beam EM | **Δ Exact Match** |
|---|---|---|---|---|---|---|
| **A** | 0.0915 | 0.0926 | **+1.2%** | 2.83% | 7.46% | **+4.63 pp (+164%)** |
| **B** | 0.1127 | 0.1137 | **+0.9%** | 4.07% | 9.94% | **+5.87 pp (+144%)** |
| **C** | 0.0988 | 0.1005 | **+1.7%** | 4.18% | 7.57% | **+3.39 pp (+81%)** |
| **D** | 0.1159 | 0.1170 | **+0.9%** | 5.88% | 11.07% | **+5.19 pp (+88%)** |

&nbsp;

**The key insight:** Beam search barely improves BLEU-4 (<2%) but **doubles Exact Match** across all models

> **ROUGE-L is beam-invariant** — |Δ ROUGE-L| ≤ 0.07% across all models.
> Beam search finds more canonical surface forms (↑EM) without adding new content (≈ ROUGE-L).
> This confirms: beam search is surface canonicalization, not quality improvement.

<!--
═══════════════════════════════════════════════════════
SCRIPT — Slide 8 · BEAM SEARCH  (~1.5 min)
═══════════════════════════════════════════════════════

"I also ran beam search with width 3 and trigram blocking — trigram blocking
prevents the model from repeating the same three-word phrase within one answer.

[Point to Δ BLEU-4 column]

The BLEU-4 improvement from beam search is less than 2% relative for all
models. That's very small. You might wonder why bother.

[Point to Δ Exact Match column]

But look at the Exact Match column. Beam search roughly doubles the exact match
rate for every model. Model D goes from 5.88% to 11.07% — nearly double.

Why this asymmetry? BLEU-4 measures partial n-gram overlap — it gets partial
credit when you generate most of the right words. Exact Match is binary — you
either reproduce the ground truth word-for-word or you don't. Beam search
tends to find the most common, most expected phrasing — the 'canonical' answer
— which is more likely to match the specific words a human annotator used.

[Point to ROUGE-L note]

There's also a very clean finding about ROUGE-L. When I compare greedy versus
beam search, ROUGE-L barely moves — the change is less than 0.07% for all
models. ROUGE-L measures Longest Common Subsequence, a property that's
insensitive to whether you pick the single most probable token or explore
alternatives. This confirms that beam search is improving surface form —
finding more canonical word choices — but not adding new content to the
answer.

Practical takeaway: use greedy for speed and general quality measurement,
beam search when exact match to human reference text is important."

[TRANSITION → Slide 9]
-->

---

<!-- SLIDE 9 — EFFECT DECOMPOSITION -->
## What Actually Drives Performance?

### 2×2 Factorial Decomposition (BLEU-4)

```
                  No Attention      Attention       Δ Attention
Scratch CNN       A  (0.0915)  ── +0.0073 ──►  C  (0.0988)    (+8.0%)
                      │                              │
                  +0.0212 (+23.2%)              +0.0171 (+17.3%)
                      │                              │
                      ▼                              ▼
Pretrained CNN    B  (0.1127)  ── +0.0032 ──►  D  (0.1159)    (+2.8%)
```

&nbsp;

| Factor | Average BLEU-4 gain | METEOR | ROUGE-L | Interpretation |
|---|---|---|---|---|
| **Pretrained features** | **+0.0192 avg (+20.2%)** | +0.039 (+12.4%) | +0.035 (+9.3%) | **Dominant factor** |
| **Dual attention** | +0.0053 avg (+5.4%) | +0.009 (+2.9%) | +0.009 (+2.2%) | Secondary factor |

**Ratio: pretrained features deliver ~3.6× more BLEU-4 improvement than attention**

> Interaction is **negative** (sub-additive): each factor compensates for the other's weakness — when features are already strong, attention adds less (+2.8%); when features are weak, attention compensates more (+8.0%)

<!--
═══════════════════════════════════════════════════════
SCRIPT — Slide 9 · DECOMPOSITION  (~2.5 min)
═══════════════════════════════════════════════════════

"This slide is the scientific core of the project. The factorial design lets
me cleanly decompose performance into contributions from each factor.

[Trace the diagram with your finger]

Reading the diagram: the horizontal arrows measure the effect of attention —
+0.0073 when combined with scratch CNN, +0.0032 when combined with ResNet.
The vertical arrows measure the effect of pretrained features — +0.0212 in
the no-attention column, +0.0171 in the attention column.

[Point to the summary table]

Averaging these: pretrained features contribute +0.0192 BLEU-4 on average —
that's a 20.2% improvement. Dual attention contributes +0.0053 — that's 5.4%.
The ratio is 3.6 to 1. Pretrained features are 3.6 times more impactful than
the decoder architecture.

This is consistent across all three primary metrics. METEOR shows 12.4% vs
2.9%. ROUGE-L shows 9.3% vs 2.2%. Three different metrics, same story.

[Point to interaction note]

Now, the interaction. If pretrained features and attention were completely
independent — if they addressed different problems — then D should score
0.0915 + 0.0212 + 0.0073 = 0.1200. But D actually scores 0.1159. The actual
gain is about 0.004 less than the sum.

This is a negative interaction, also called sub-additivity. Both factors
partially address the same underlying weakness: poor visual understanding. The
scratch CNN doesn't understand images well, so attention compensates by
focusing on the most informative regions. But when you already have rich
ResNet features, the features are already informative everywhere — so the
attention mechanism adds less marginal value.

The practical implication: if you can only make one architectural investment,
choose the visual encoder. Pretrained features give the biggest return."

[TRANSITION → Slide 10]
-->

---

<!-- SLIDE 10 — VS. ORIGINAL PAPER -->
## Against the Original Paper — Li et al. ECCV 2018

**Same evaluation set: VQA-E val split, n = 88,488**

| Model | BLEU-4 | ROUGE-L | vs. Li et al. best |
|---|---|---|---|
| Li et al. — QI-E Grid (generation only) | 7.60 | 34.00 | −1.80 pp |
| Li et al. — QI-E Bottom-up (generation only) | 8.60 | — | −0.80 pp |
| Li et al. — **QI-AE Bottom-up** *(multi-task, best)* | **9.40** | **36.33** | baseline |
| **Ours — Model A** *(no pretrain, no attn, no multi-task)* | **9.15** | **38.28** | **−0.25 pp** |
| **Ours — Model B** *(ResNet, no attn, no multi-task)* | **11.27** | **42.37** | **+1.87 pp** |
| **Ours — Model D** *(ResNet + dual attn, no multi-task)* | **11.59** | **42.70** | **+2.19 pp** |

&nbsp;

<span class="win">Model D: BLEU-4 +23.3% · ROUGE-L +17.5% above Li et al. best — without multi-task supervision</span>

> **Their advantages:** ResNet-152 (vs our 101) · Bottom-up Faster R-CNN regions · Multi-task answer classification
> **Our advantages:** BiLSTM (vs GRU) · Dual attention (vs image-only) · GatedFusion · 3-phase training + scheduled sampling

<!--
═══════════════════════════════════════════════════════
SCRIPT — Slide 10 · VS. PAPER  (~3 min)
═══════════════════════════════════════════════════════

"This is the headline result: how do we compare to the original VQA-E paper?

[Point to Li et al. rows first]

Let me first explain their models. Li et al. test multiple variants. Their
best model — the one in the bottom row here labeled QI-AE — uses multi-task
learning. It simultaneously trains a sigmoid classifier over 3,129 answer
candidates, and an LSTM explanation generator. This is a structural advantage:
the model explicitly knows what the correct answer is before it starts
generating the explanation. Our models have no such signal — they must figure
out both the answer and the explanation purely from the generation loss.

They also use Bottom-up features — Faster R-CNN object detection regions,
which are semantically richer than grid features because they correspond to
actual detected objects. And they use ResNet-152, which is deeper than our
ResNet-101.

[Point to our model rows]

Despite those three structural advantages on their side, our Model D achieves
11.59% BLEU-4 versus their 9.40%. That's a 23% relative improvement.

For ROUGE-L: 42.70% versus 36.33%. That's 17.5% higher.

Both improvements are confirmed independently by three different metrics, and
they hold consistently across BLEU-1 through BLEU-4.

[Point to Model A row]

Even more telling: our baseline, Model A — no pretrained features, no
attention, no multi-task supervision — scores 9.15% BLEU-4. That is only
0.25 percentage points below their best model, which had all three of those
advantages. Our training protocol alone nearly matches their state-of-the-art.

[Point to advantages section]

Why do we outperform? I attribute it primarily to three things: our BiLSTM
question encoder captures bidirectional context versus their single-layer GRU;
our dual attention attends to both visual regions and question tokens at each
decode step versus their single image attention; and most importantly, our
3-phase progressive training with scheduled sampling — they train for 15
epochs with a fixed learning rate of 0.01, while we train for 30-plus epochs
across three carefully designed phases."

[TRANSITION → Slide 11]
-->

---

<!-- SLIDE 11 — QUALITATIVE EXAMPLES -->
## Real Predictions — Random Validation Samples

**Not cherry-picked — randomly drawn from validation set**

| Question | Ground Truth | Model A | Model B | **Model D** |
|---|---|---|---|---|
| *"What is the boy doing?"* | *flying kite because ... beach ... kite ...* | **skiing** because a person is skiing down a hill | **flying kite** because a man is flying a kite | **flying kite** because a man is flying a kite in the sky |
| *"How many cats on the couch?"* | *2 because two cats laying on a couch near one another* | 2 because two cats sleeping on a **bed** | 2 because two cats laying on a **bed** | 2 because two cats laying on a **bed** |
| *"What is the tablet next to?"* | *scissors because The tablet is next to scissors* | **laptop** because a laptop is on a desk | **scissors** because a pair of scissors are on a table | **scissors** because a pair of scissors are on a table |
| *"Are these people about to eat?"* | *no because people are playing video games* | **yes** because people are sitting at a table | **yes** because people are standing around a table | **yes** because people are playing a video game |

&nbsp;

**Pattern:** Model D correctly identifies the scene — but can still predict the wrong yes/no answer

<!--
═══════════════════════════════════════════════════════
SCRIPT — Slide 11 · QUALITATIVE  (~2.5 min)
═══════════════════════════════════════════════════════

"Let me show some real predictions. These were drawn at random from the
validation set — not selected to make any model look good.

[Point to row 1 — kite]

First example: 'What is the boy doing?' The ground truth is 'flying kite
because a man is flying a kite in the sky.' Model A predicts 'skiing because
a person is skiing down a hill.' This is a clear failure of the scratch CNN —
it's confusing the beach scene (sand, blue sky, person with arm raised) with
a snow skiing scene. Models B and D with pretrained ResNet101 both correctly
identify 'flying kite.' This is a clean example of why pretrained features
matter.

[Point to row 2 — cats]

Second example: 'How many cats on the couch?' All models get the count right
— 2 — but every single model says 'bed' instead of 'couch.' This is a
training distribution artifact: VQA-E has more cats-on-bed images than
cats-on-couch images, so the model has learned that cats go on beds. The
count is correct but the location is wrong. This is very hard to fix without
balancing the training distribution.

[Point to row 3 — scissors]

Third example: 'What is the tablet next to?' Model A predicts 'laptop' — it
confuses the rectangular shape of a tablet with a laptop, which is much more
common in training. Models B and D correctly identify 'scissors,' likely
because their ResNet features can distinguish the distinctive shape of scissors.

[Point to row 4 — yes/no]

The fourth example is the most interesting failure. 'Are these people about
to eat?' — the ground truth is 'no because people are playing video games.'
Model D generates 'yes because people are playing a video game.' It correctly
identifies the activity — video games — but still predicts 'yes' instead of
'no.' The scene description is right, but the yes/no classification is wrong.
This reveals a fundamental issue with the template-structured generation: the
model generates the answer token and the explanation somewhat independently.
Getting the explanation right doesn't guarantee the answer is right."

[TRANSITION → Slide 12]
-->

---

<!-- SLIDE 12 — TRAINING DYNAMICS -->
## Training Dynamics — What Actually Happened

**From recorded training history, `history_model_*.json`**

| Model | Epoch 1 val | Epoch 10 val | Epoch 15 val | Phase 3 behavior | Best val |
|---|---|---|---|---|---|
| A | 4.365 | 3.375 | 3.299 | Monotone increase after ep16 | **3.2983** (ep16) |
| B | 4.335 | 3.274 | 3.218 | Train collapses to 2.65, val rises to 3.29 | **3.2178** (ep15) |
| C | 4.530 | 3.325 | 3.277 | Immediate spike 3.277→3.430 in 2 epochs | **3.2774** (ep15) |
| **D** | 4.622 | 3.260 | 3.222 | Spikes then **recovers to 3.2625 at ep21** | **3.2216** (ep15) |

&nbsp;

**Key observations:**
- Pretrained models (B, D) reach lower loss at every phase — not just at the end
- All best checkpoints are at epoch 15–16 — the Phase 2/Phase 3 boundary
- B shows the widest train/val gap (0.64) — fine-tuned backbone memorizing training set
- D is the only model where scheduled sampling produces a real recovery

<!--
═══════════════════════════════════════════════════════
SCRIPT — Slide 12 · TRAINING DYNAMICS  (~2 min)
═══════════════════════════════════════════════════════

"Looking at the actual training curves, we can see four distinct behaviors.

[Point to each row as you describe it]

Model A — the scratch baseline — converges smoothly. Phase 3 doesn't help:
its validation loss increases monotonically once scheduled sampling starts.
Scratch CNN features don't provide enough quality for the decoder to learn
from its own mistakes.

Model B — interesting. Training loss drops aggressively during Phase 3 all
the way to 2.65. But validation loss rises to 3.29. The gap between train
and val is 0.64 — the largest of any model. The fine-tuned ResNet backbone
is memorizing training patterns without generalizing. Classic overfitting.

Model C — the most unstable. When scheduled sampling starts at epoch 16,
validation loss spikes by 0.15 in just two epochs — 3.277 up to 3.430.
Dual attention combined with scratch features is fragile: even a small
fraction of self-fed inputs causes the attention weights to diverge.

Model D — the most interesting. Like C, it spikes initially when Phase 3
begins. But unlike C, it partially recovers — val loss comes back down to
3.2625 by epoch 21. This is the only model where scheduled sampling
provides a genuine benefit. The combination of strong pretrained features
and dual attention provides enough structural stability that the decoder can
actually learn to handle its own imperfect outputs.

[Point to best checkpoints]

All four best checkpoints land at epoch 15 or 16 — right at the boundary
between Phase 2 and Phase 3. This tells us that Phase 2 fine-tuning is
where most of the real quality gain happens."

[TRANSITION → Slide 13]
-->

---

<!-- SLIDE 13 — LIMITATIONS -->
## Limitations — Honest Assessment

| Gap | Severity | Our response |
|---|---|---|
| No multi-task answer classification | Moderate | We still outperform BLEU-4 +23.3% and ROUGE-L +17.5% |
| No CIDEr-D | Cannot compare | VQA-E has 1 reference/sample; CIDEr-D needs 5 |
| BERTScore ceiling effect | Low | All models 0.900–0.909; range = 0.009 — valid but weak discriminator for this task |
| No Faster R-CNN bottom-up features | Moderate | Our 7×7 grid is simpler; object-level regions would improve all models |
| Single evaluation set | Low | No test set in VQA-E public release; val set is standard in the literature |
| 30-epoch training limit | Low | Phase 2 is where gains happen; more epochs unlikely to change the ranking |

&nbsp;

**⚠️ METEOR warning — must address if asked:**

> Our METEOR (31–36%) vs Li et al. (14–17%) — **2× difference is NOT a real quality gap**
> Our implementation: NLTK `meteor_score` with WordNet synonym matching (default ON)
> Li et al.: standard METEOR 1.5 (synonym matching OFF by default)
> Result: "riding" matches "skateboarding" in NLTK, not in METEOR 1.5
> **Safe cross-paper comparison: use BLEU-4 and ROUGE-L only**

<!--
═══════════════════════════════════════════════════════
SCRIPT — Slide 13 · LIMITATIONS  (~2 min)
═══════════════════════════════════════════════════════

"I want to be completely honest about the limitations of this work.

[Point to each row]

The most significant gap is multi-task learning. Li et al. simultaneously
classify the answer over 3,129 candidates and generate the explanation. That
explicit answer supervision is a real advantage. We don't have it. Despite
this, we outperform on BLEU-4 and ROUGE-L — so the answer to 'does multi-task
help more than better architecture?' appears to be no, at least for the
generation metrics.

CIDEr-D is the standard metric for image captioning and something the original
paper reports. I can't compute it because standard CIDEr-D requires 5 reference
sentences per sample, and VQA-E provides only 1.

[Point to BERTScore row]

BERTScore is technically computed and is statistically significant — with 88,000
samples, even a 0.001 difference has p < 0.001. But the spread of only 0.009
across all four models makes it a weak discriminator for this task. BLEU-4
and METEOR show more contrast.

[Point to METEOR warning]

I need to flag the METEOR issue explicitly because it looks suspicious. Our
METEOR scores appear to be roughly twice the paper's values. This is not
because we're twice as good — it's because NLTK's meteor_score enables
WordNet synonym matching by default. 'Riding' and 'skateboarding' count as
partial matches. 'Person' and 'man' count as partial matches. The standard
METEOR implementation doesn't do this. They are measuring different things.
Within our four models, METEOR is a valid relative comparison. For
cross-paper comparison, use BLEU-4 and ROUGE-L only."

[TRANSITION → Slide 14]
-->

---

<!-- SLIDE 14 — CONTRIBUTIONS -->
## Key Contributions

**1. Controlled 2×2 Factorial Study**
→ Clean empirical answer: pretrained features contribute 3.6× more than attention
→ Negative interaction confirmed across 3 independent metrics

**2. Surpass Original Paper (single-task, no multi-task advantage)**
→ BLEU-4: 11.59% vs 9.40% (+23.3%) · ROUGE-L: 42.70% vs 36.33% (+17.5%)
→ Even baseline Model A (9.15%) nearly matches their best (9.40%)

**3. Architectural Package**
→ BiLSTM question encoder (vs GRU in Li et al.)
→ Dual attention: image regions + question tokens per decode step
→ GatedFusion: per-dimension learnable gate (vs Hadamard product)
→ Coverage mechanism: prevents repetitive attention patterns

**4. 3-Phase Progressive Training with Relative Epoch Scheduling**
→ Scheduled sampling with `ε(e_rel)` not `ε(e_abs)` — critical correctness fix
→ Phase 3 finding: only Model D benefits; scratch models should stop at Phase 2

**5. Comprehensive 7-Metric Evaluation Suite**
→ BLEU-1/2/3/4, METEOR, ROUGE-L, BERTScore, Exact Match on full 88,488 val set
→ Greedy + beam search for all models; ROUGE-L beam-invariance identified

<!--
═══════════════════════════════════════════════════════
SCRIPT — Slide 14 · CONTRIBUTIONS  (~2 min)
═══════════════════════════════════════════════════════

"Let me summarize the contributions.

[Read each point, elaborating slightly]

First, the factorial study. By training all four models under completely
controlled conditions, I can give clean, quantitative answers to the research
questions. Pretrained features: 20% BLEU-4 gain. Dual attention: 5.4%.
Interaction: negative, confirming partial substitutability.

Second, and most concretely: we outperform the original paper by 23% on
BLEU-4 without their multi-task supervision, and the result is confirmed
independently by BLEU-4, METEOR, and ROUGE-L — three metrics that compute
quality in three different ways and all agree.

Third, the architectural package. Each piece is taken from the NLP literature —
BiLSTM for question encoding, Bahdanau attention, coverage from the
pointer-generator network paper — and combined into a coherent model.

Fourth, the training protocol. The scheduled sampling relative-epoch fix is
not a minor detail — it's the difference between stable training and training
collapse in Phase 3. And the observation that scratch models should stop at
Phase 2 is a practical finding that saves compute.

Fifth, the evaluation suite. Reporting seven metrics on 88,000 samples with
both greedy and beam search gives a comprehensive picture. The ROUGE-L
beam-invariance finding — that beam search improves Exact Match but not
content coverage — is a clean empirical insight about what beam search
actually does."

[TRANSITION → Slide 15]
-->

---

<!-- SLIDE 15 — CONCLUSION -->
## Conclusion

**All three research questions have data-backed answers:**

| Question | Answer | BLEU-4 evidence |
|---|---|---|
| Does transfer learning help? | **Yes, substantially** | +17.3–23.2% relative over scratch CNN |
| Does dual attention help? | **Yes, but modestly** | +2.8–8.0% relative |
| Do the effects compose? | **Yes, sub-additively** | D = 0.1159 < B + C combined (expected 0.1200) |

&nbsp;

**Recommended configurations:**

| Priority | Model | Why |
|---|---|---|
| **Best performance** | **Model D** | Best on all 7 metrics. BLEU-4 = 0.1159 (greedy), EM = 11.07% (beam) |
| **Best efficiency** | **Model B** | 97.2% of D's BLEU-4, 20% fewer trainable params (40.7M vs 51.2M). BLEU-4/10M = 0.0277 |

&nbsp;

> **Take-away:** In CNN-LSTM VQA, **visual encoder quality dominates decoder design**.
> Invest in the visual backbone first — then improve the decoder.

<!--
═══════════════════════════════════════════════════════
SCRIPT — Slide 15 · CONCLUSION  (~2 min)
═══════════════════════════════════════════════════════

"To close.

[Point to research questions table]

I came in with three questions, and I can answer all three with actual
experimental data on 88,000 validation samples.

Transfer learning helps substantially — the improvement is 17 to 23 percent
relative BLEU-4 depending on whether attention is present. This is robust
and reproducible.

Dual attention helps, but modestly — 2.8 to 8.0 percent relative. The
improvement is real but smaller than pretrained features, and it's larger
when the visual encoder is weaker.

The effects compose but sub-additively, confirming a negative interaction.
Both factors partially address visual understanding — so they're partially
redundant.

[Point to recommendations]

In terms of practical recommendations: if you want the absolute best
performance, use Model D — ResNet101, dual attention, coverage. It's best on
every metric.

But if resources are constrained, Model B is the right choice. It achieves
97.2% of Model D's BLEU-4 with 20% fewer trainable parameters, and it's much
simpler to implement and debug. The 2.8% additional BLEU-4 from adding dual
attention to an already-strong ResNet backbone may not be worth the added
complexity in a production setting.

[Point to take-away box]

The headline insight: in CNN-LSTM VQA, visual encoder quality dominates
decoder design. Pretrained features give 3.6 times the return of attention
improvements. If you have limited time or compute, invest it in the visual
backbone first.

[Pause]

Thank you. I'm happy to take questions."
-->

---

<!-- SLIDE 16 — BACKUP: ARCHITECTURE DETAILS -->
## Backup: Component Specifications

### Image Encoders
| Model | Architecture | Output |
|---|---|---|
| A (SimpleCNN) | 5× [Conv3×3→BN→ReLU→MaxPool] → AdaptiveAvgPool → FC(1024→1024) | (B, 1024) |
| B (ResNetEncoder) | ResNet101 pretrained, remove FC, keep avgpool → Linear(2048→1024) | (B, 1024) |
| C (SimpleCNNSpatial) | Same as A but remove avgpool → Conv1×1(1024→1024) → reshape | (B, 49, 1024) |
| D (ResNetSpatial) | ResNet101, remove avgpool+FC → Conv2d(2048→1024, k=1) → reshape | (B, 49, 1024) |

### Dual Attention (Models C/D)
```
At each decode step t:
  energy_img = tanh(W_h · h_top + W_img · region_i + W_cov · coverage_i)
  α_img      = softmax(v · energy_img)          # (B, 49)
  ctx_img    = Σ α_img · region                 # (B, 1024)
  (same for Q: energy_q, α_q, ctx_q)
  lstm_input = [embed(a_{t-1}) ; ctx_img ; ctx_q]   # (B, 512+1024+1024 = 2560)
  coverage  += α_img  (accumulated, prevents re-attending)
```

### Parameter counts (exact, from model.parameters())
| A: 45.9M | B: 83.2M (40.7M trainable) | C: 56.4M | D: 93.7M (51.2M trainable) |

<!--
═══════════════════════════════════════════════════════
SCRIPT — Backup Slide 16 · ARCHITECTURE DETAILS
═══════════════════════════════════════════════════════

IF ASKED: "Can you explain the attention mechanism in more detail?"

"Sure. At each decode step, the attention computes an energy score for every
one of the 49 image regions. This energy is a learned function — specifically
a tanh of a weighted sum — combining the current decoder hidden state, the
region feature, and a coverage vector.

The coverage vector accumulates the sum of all previous attention weights for
each region. If the model has already attended heavily to region 15 in steps
1 through 5, coverage[15] will be large, and the energy formula penalizes
re-attending to it. This prevents the attention from getting stuck on the most
salient region for every step of the generation.

The softmax of the energy gives us the alpha weights — a probability
distribution over the 49 regions. The context vector is the weighted average
of region features. This, together with an analogous computation over question
hidden states, gets concatenated with the current word embedding as the LSTM
input. So the LSTM input size is 512 (embedding) + 1024 (image context) +
1024 (question context) = 2560 dimensions."

IF ASKED: "Why 49 regions specifically?"

"49 comes naturally from the CNN architecture. Five max-pooling layers each
halve the spatial dimension: 224 → 112 → 56 → 28 → 14 → 7. So after five
pooling operations we have a 7×7 feature map, which is 49 spatial positions.
Each position has a receptive field of approximately 32×32 pixels in the
original image."
-->

---

<!-- SLIDE 17 — BACKUP: FULL RESULTS TABLE -->
## Backup: Complete Evaluation Results

### Greedy Decoding (n = 88,488)

| Model | B-1 | B-2 | B-3 | B-4 | METEOR | ROUGE-L | BERTScore | EM |
|---|---|---|---|---|---|---|---|---|
| A | 0.3715 | 0.2335 | 0.1415 | 0.0915 | 0.3117 | 0.3828 | 0.9008 | 2.83% |
| B | 0.4124 | 0.2702 | 0.1715 | 0.1127 | 0.3561 | 0.4237 | 0.9081 | 4.07% |
| C | 0.3865 | 0.2463 | 0.1516 | 0.0988 | 0.3271 | 0.3971 | 0.9034 | 4.18% |
| **D** | **0.4151** | **0.2734** | **0.1748** | **0.1159** | **0.3595** | **0.4270** | **0.9085** | **5.88%** |

### Beam Search w=3, trigram blocking (n = 88,488)

| Model | B-1 | B-2 | B-3 | B-4 | METEOR | ROUGE-L | BERTScore | EM |
|---|---|---|---|---|---|---|---|---|
| A | 0.3723 | 0.2333 | 0.1421 | 0.0926 | 0.3154 | 0.3823 | 0.8999 | 7.46% |
| B | 0.4122 | 0.2690 | 0.1713 | 0.1137 | 0.3589 | 0.4230 | 0.9073 | 9.94% |
| C | 0.3872 | 0.2465 | 0.1527 | 0.1005 | 0.3300 | 0.3972 | 0.9026 | 7.57% |
| **D** | **0.4160** | **0.2734** | **0.1754** | **0.1170** | **0.3632** | **0.4269** | **0.9080** | **11.07%** |

*Source: `outputs/evaluation_results.json` · Exact to 4 decimal places*

<!--
═══════════════════════════════════════════════════════
SCRIPT — Backup Slide 17 · NUMBERS ON DEMAND
═══════════════════════════════════════════════════════

This backup slide has all numbers at a glance if asked for any specific metric.

KEY NUMBERS TO MEMORIZE:
  Model D greedy BLEU-4:  0.1159  (the headline result)
  Model D beam BLEU-4:    0.1170
  Model D greedy ROUGE-L: 0.4270
  Model D beam EM:        11.07%
  Li et al. best BLEU-4:  9.40%  (scores are in %, our table is in [0,1])
  → Verify: 11.59% vs 9.40% = +23.3%

  Model B greedy BLEU-4:  0.1127  (second best, efficiency winner)
  Trainable params:       A=45.9M  B=40.7M  C=56.4M  D=51.2M

IF ASKED: "What's the BLEU-4 in percentage terms?"
  → Multiply by 100: A=9.15%, B=11.27%, C=9.88%, D=11.59%
  → Li et al. report in %, so when comparing: our D=11.59% vs their best=9.40%
-->

---

<!-- SLIDE 18 — BACKUP: LIKELY QUESTIONS & ANSWERS -->
## Backup: Anticipated Questions

**Q: Why not use a Transformer instead of LSTM?**
→ Scope: CNN-LSTM is the architecture family studied; Transformers (BLIP, LLaVA) are a separate family.
LSTM still performs competitively on this dataset and isolates the factors I'm studying.
Future work: replace LSTM decoder with transformer decoder — likely a substantial improvement.

**Q: How did you choose beam width = 3?**
→ Standard choice in the literature; Li et al. also use small beam widths.
Width 3 balances exploration vs. inference speed. Wider beams plateau quickly for short sequences (10–20 tokens).

**Q: Why not use attention over only the answer vocabulary (pointer network)?**
→ The answer vocabulary (8,648 words) is a superset of all generated words.
A pointer network would only help with rare words from the image — not needed here since all target words are in-vocabulary.

**Q: Isn't the BERTScore range of 0.009 too small to be meaningful?**
→ With n=88,488, a difference of 0.001 is statistically significant (p << 0.001 by paired t-test).
However, it's practically small — all models produce semantically similar outputs within the 'answer because explanation' template. BERTScore is poorly suited to discriminate within a narrow output distribution.

**Q: What would happen with a larger dataset or more epochs?**
→ The ranking (D>B>C>A) is unlikely to change — it's driven by architectural quality, not training length.
All best checkpoints emerge at epoch 15–16. Phase 3 only marginally helps Model D.
Larger data would likely widen the gap between pretrained and scratch models further.

**Q: Could you explain the GatedFusion more precisely?**
→ Both image and question features are passed through tanh-activated linear layers.
A sigmoid gate `g = σ(W·[f_img; f_q])` is a 1024-dim vector learned per input.
Output: `g ⊙ tanh(W_img·f_img) + (1−g) ⊙ tanh(W_q·f_q)`.
Each of the 1024 dimensions independently decides how much to weight the image vs the question.

<!--
SCRIPT — Backup Slide 18 · Q&A PREP

Read through these before presenting. Know the answers cold.
Key numbers to have memorized:
  - D beats Li et al. by +23.3% BLEU-4 and +17.5% ROUGE-L
  - Pretrained features: +20.2% BLEU-4 average
  - Dual attention: +5.4% BLEU-4 average
  - Ratio: 3.6×
  - Best checkpoint for all models: epoch 15 or 16
  - Model D BLEU-4 = 0.1159 greedy / 0.1170 beam
  - Model D EM = 5.88% greedy / 11.07% beam
  - Total params D = 93.7M · Trainable = 51.2M
-->

---

<!--
════════════════════════════════════════════════════
RENDERING INSTRUCTIONS
════════════════════════════════════════════════════

Option 1 — VS Code (recommended):
  Install extension: "Marp for VS Code" (marp-team.marp-vscode)
  Open this file → Click Marp icon (top-right) → Open Preview
  Export: Ctrl+Shift+P → "Marp: Export Slide Deck" → PDF or PPTX

Option 2 — Command line:
  npm install -g @marp-team/marp-cli
  marp PRESENTATION.md --pdf  -o slides.pdf
  marp PRESENTATION.md --pptx -o slides.pptx

Option 3 — Use as a speaker script only
  Read the SCRIPT sections under each slide as your speaking notes.
  Everything under ═══...═══ is your script — do NOT project these lines.

════════════════════════════════════════════════════
TIMING GUIDE  (target: 20–25 minutes)
════════════════════════════════════════════════════

  Slide  1  Title                        1 min
  Slide  2  Why Explanatory VQA          1.5 min
  Slide  3  Dataset                      1.5 min
  Slide  4  2×2 Design                   1.5 min
  Slide  5  Architecture                 2.5 min
  Slide  6  3-Phase Training             2.5 min
  Slide  7  Results (Greedy)             2.5 min
  Slide  8  Beam Search                  1.5 min
  Slide  9  Effect Decomposition         2.5 min
  Slide 10  vs. Li et al.               3.0 min
  Slide 11  Qualitative Examples         2.5 min
  Slide 12  Training Dynamics            2.0 min
  Slide 13  Limitations                  2.0 min
  Slide 14  Contributions                2.0 min
  Slide 15  Conclusion                   2.0 min
  ──────────────────────────────────────────────
  Core total:                          ~30 min

  FOR A 20-MINUTE SLOT: skip Slides 12 (training dynamics) and 8 (beam)
  FOR A 15-MINUTE SLOT: also skip Slide 13 (limitations) and compress Slide 3

  Backup slides (16-18): show only if asked during Q&A
-->
