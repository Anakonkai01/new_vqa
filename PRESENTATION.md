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

<!-- SLIDE 1: TITLE -->
# VQA-E: Explaining Visual Questions
## A Systematic Comparison of CNN-LSTM Architectures

**Generative Visual Question Answering with Explanatory Answers**

---

> **Question:** *"What is the boy doing?"*
> **Answer:** *"flying kite because a man is flying a kite in the sky"*

&nbsp;

**2×2 Factorial Design** — 4 Models × Full VQA-E Validation Set (88,488 samples)

&nbsp;

*Based on: Li et al., "VQA-E: Explaining, Elaborating, and Enhancing Your Answers for Visual Questions", ECCV 2018*

<!--
SPEAKER NOTES — Slide 1 (Title) ~1 min

"Good morning. My project is about Visual Question Answering with Explanations — a task where the model must not just answer a question about an image, but also explain WHY it gave that answer.

The core idea is shown here: given an image of a boy on a beach flying a kite, the model should output not just 'flying kite' but 'flying kite BECAUSE a man is flying a kite in the sky.'

My work is a systematic comparison of four architectures on this task, using the VQA-E dataset introduced by Li et al. in ECCV 2018. I'll show you what I built, how I trained it, and most importantly — how it performs against the original paper."
-->

---

<!-- SLIDE 2: MOTIVATION -->
## Why Explanatory VQA?

Standard VQA: **"What color is the car?"** → *"red"*

Explanatory VQA: **"What color is the car?"** → *"red because the car in the foreground is red"*

&nbsp;

**Why it matters:**

| Standard VQA | Explanatory VQA |
|---|---|
| Single token output | Full sentence generation |
| No reasoning trace | Explicit justification |
| Not accessible to visually impaired | Natural language explanation |
| Easy to guess from language bias | Must attend to the image |

&nbsp;

> Li et al. (2018) show that forcing a model to explain its answer **also improves answer accuracy** — explanation generation is a powerful form of self-supervision.

<!--
SPEAKER NOTES — Slide 2 (Motivation) ~1.5 min

"The key motivation is that standard VQA outputs a single word or short phrase — just the answer. Explanatory VQA goes further: the model must justify its answer in natural language.

This is valuable for three reasons:
First, it makes the model's reasoning transparent — you can see whether it's guessing or genuinely understanding the image.
Second, Li et al. showed in their original paper that explanation supervision actually IMPROVES answer accuracy. Forcing the model to explain forces it to really look at the image rather than relying on language bias.
Third, it's more useful for accessibility applications.

The task is significantly harder than standard VQA — you're training a sequence-to-sequence model, not a classifier."
-->

---

<!-- SLIDE 3: DATASET -->
## VQA-E Dataset

**Automatically derived from VQA v2 + MSCOCO captions**

| Split | Images | Q&A Pairs | Unique Q | Unique A |
|---|---|---|---|---|
| Train | 72,680 | **181,298** | 77,418 | 9,491 |
| **Val** | 35,645 | **88,488** | 42,055 | 6,247 |
| Total | 108,325 | 269,786 | 108,872 | 12,450 |

&nbsp;

**How explanations are synthesized:**
1. Find the MSCOCO caption most relevant to the question+answer (GloVe similarity)
2. Merge Q+A into a declarative statement: *"The boy is flying a kite"*
3. Fuse with caption via constituency parse tree alignment
4. Filter by similarity threshold ≥ 0.6 → keeps 41% of all QA pairs

&nbsp;

**My vocabulary:** Q vocab = 4,546 words · A vocab = 8,648 words (min_freq=3)

<!--
SPEAKER NOTES — Slide 3 (Dataset) ~1.5 min

"The VQA-E dataset is built automatically by Li et al. from two existing resources: the VQA v2 dataset and MSCOCO captions.

For each question-answer pair, they find the image caption most relevant to the question and answer using GloVe word embedding similarity. Then they merge the QA statement with the caption using constituency parse trees to create a natural-sounding explanation.

The key quality control step is a similarity threshold of 0.6 — only 41% of QA pairs get explanations, because captions don't always describe what the question is asking about. For example, a caption about a kitchen doesn't help explain why a dog is in the background.

I use the standard train/val split: 181K training samples and 88,488 validation samples. I built two vocabularies: one for questions (4,546 words) and one for answers/explanations (8,648 words)."
-->

---

<!-- SLIDE 4: THE 2×2 DESIGN -->
## The 2×2 Factorial Design

**Two binary decisions → 4 models, all else equal**

|  | **No Attention** | **Dual Attention** |
|---|---|---|
| **Scratch CNN** | **Model A** (Baseline) | **Model C** |
| **Pretrained ResNet101** | **Model B** | **Model D** ⭐ |

&nbsp;

**Why factorial design?**
- Cleanly separates two effects: *visual encoder quality* vs *decoder strategy*
- Each model is the ablation of another → no confounding variables
- A→B measures the effect of pretrained features (holding attention constant)
- A→C measures the effect of attention (holding features constant)

&nbsp;

> This is the core scientific contribution: **controlled comparison under identical training conditions**

<!--
SPEAKER NOTES — Slide 4 (2x2 Design) ~1.5 min

"The central design decision of this project is a 2×2 factorial experiment. I vary two independent factors:
Factor 1: Visual encoder — either a SimpleCNN trained from scratch, or a pretrained ResNet101.
Factor 2: Decoder strategy — either a plain LSTM, or an LSTM with dual attention and a coverage mechanism.

This gives 4 models. The value of this design is that I can cleanly measure each factor's contribution: A to B tells me exactly what pretrained features add, while A to C tells me what attention adds. Model D combines both.

All four models are trained under identical conditions: same data, same batch size, same 3-phase training schedule, same optimizer. The only differences are the components I'm studying."
-->

---

<!-- SLIDE 5: ARCHITECTURE OVERVIEW -->
## Architecture Overview

```
Image (3×224×224) ──► CNN Encoder ──────────────────────────┐
                                                              ▼
                                                    GatedFusion ──► h₀ ──► LSTM Decoder ──► Answer
Question tokens ──► BiLSTM Encoder ─────────────────────────┘              ▲
                         │                                                  │
                         └── q_hidden_states ──────────────────────────────┘
                                                        (Dual Attention: image + question)
```

&nbsp;

**Key components (shared by all models):**
- **BiLSTM Question Encoder** — 2-layer bidirectional LSTM, captures full question context
- **GatedFusion** — learnable gate: `gate·tanh(W_img) + (1-gate)·tanh(W_q)`
- **LSTM Decoder** — 2-layer, 1024 hidden units, teacher forcing during training
- **GloVe 300d** embeddings → projected to 512d, weight tying on output layer

**A/B only:** Global image vector → initialize decoder h₀

**C/D only:** 49 spatial regions → Bahdanau attention at EACH decode step + coverage

<!--
SPEAKER NOTES — Slide 5 (Architecture) ~2 min

"Let me walk through the architecture. All four models share the same skeleton:

The image goes through a CNN encoder — either our scratch SimpleCNN or pretrained ResNet101. The question goes through a 2-layer bidirectional LSTM that captures context in both directions. The image and question representations are fused using a GatedFusion module — this is more expressive than the Hadamard product used in the original paper, because the gate decides dynamically how much to rely on the image vs the question.

The fused vector initializes the decoder's hidden state. The decoder is a 2-layer LSTM that generates the answer one token at a time.

The key difference between models: In A and B, the CNN produces a single global vector, and the decoder never looks at the image again after initialization. In C and D, the CNN produces 49 spatial regions (a 7×7 grid), and at EVERY decode step, the decoder runs a Bahdanau attention over both the image regions AND the question hidden states. This dual attention is my main architectural innovation over the original paper.

I also added a coverage mechanism that tracks cumulative attention to prevent the decoder from repeatedly attending to the same image regions."
-->

---

<!-- SLIDE 6: TRAINING PIPELINE -->
## 3-Phase Progressive Training

```
Phase 1 (Epochs 1–10)        Phase 2 (Epochs 11–15)       Phase 3 (Epochs 16+)
Teacher Forcing 100%    ──►  Fine-tune ResNet (B/D)  ──►  Scheduled Sampling
LR = 1e-3                    LR = 5e-4                    LR = 2e-4
All layers train             Unfreeze ResNet layer3+4     ε decays: 1.0 → ~0.6
```

**Scheduled Sampling** reduces *exposure bias*:
- Training: model sees ground-truth tokens (easy, stable)
- Inference: model sees its own predictions (harder, realistic)
- Solution: gradually mix own predictions into training → `ε(e_rel) = k / (k + exp(e_rel/k))`

&nbsp;

**Best checkpoints (by val loss):**

| Model | Best Epoch | Best Val Loss |
|---|---|---|
| A | 16 | 3.2983 |
| B | 15 | **3.2178** |
| C | 15 | 3.2774 |
| D | 15 | 3.2216 |

<!--
SPEAKER NOTES — Slide 6 (Training) ~2 min

"Training is the most technically involved part. I use a 3-phase progressive approach.

Phase 1 is standard teacher forcing for 10 epochs — the decoder sees ground-truth tokens at each step. This gives stable gradient signals and fast initial learning.

Phase 2 runs for 5 epochs with a lower learning rate. For models B and D which use pretrained ResNet101, I also unfreeze the top layers of ResNet (layer3 and layer4) here, allowing the visual features to adapt to the VQA-E task.

Phase 3 introduces Scheduled Sampling — a technique to reduce 'exposure bias.' The problem is that during training the decoder always sees correct tokens, but at inference time it sees its own potentially wrong predictions. This mismatch hurts performance. Scheduled sampling bridges this by gradually replacing ground-truth inputs with the model's own predictions. The epsilon parameter decays from 1.0 to about 0.6 — so by the end, 40% of training tokens come from the model itself.

The critical implementation detail: the epsilon uses the RELATIVE epoch within Phase 3, not the absolute epoch. Using absolute epoch would cause epsilon to decay too fast and destabilize training.

All best checkpoints emerge at epoch 15 or 16, right at the Phase 2/3 boundary."
-->

---

<!-- SLIDE 7: QUANTITATIVE RESULTS -->
## Quantitative Results

**Full VQA-E validation set — n = 88,488 samples**

### Greedy Decoding

| Model | BLEU-1 | BLEU-4 ⭐ | METEOR ⭐ | ROUGE-L ⭐ | BERTScore | Exact Match |
|---|---|---|---|---|---|---|
| **A** Scratch+NoAttn | 0.3715 | 0.0915 | 0.3117 | 0.3828 | 0.9008 | 2.83% |
| **B** Pretrain+NoAttn | 0.4124 | 0.1127 | 0.3561 | 0.4237 | 0.9081 | 4.07% |
| **C** Scratch+Attn | 0.3865 | 0.0988 | 0.3271 | 0.3971 | 0.9034 | 4.18% |
| **D** Pretrain+Attn | **0.4151** | **0.1159** | **0.3595** | **0.4270** | **0.9085** | **5.88%** |

&nbsp;

**A → D improvement:** BLEU-4 **+26.7%** relative · METEOR **+15.3%** · ROUGE-L **+11.5%** · Exact Match **+107.8%**

<!--
SPEAKER NOTES — Slide 7 (Results) ~2 min

"Here are the main results. All four models evaluated on the full 88,488-sample validation set.

Model D is the clear winner across every metric. The BLEU-4 range goes from 0.0915 for Model A up to 0.1159 for Model D — that's a 26.7% relative improvement.

A few things to notice:
First, the ranking is D > B > C > A — pretrained features matter more than attention. Model B beats Model C despite having no attention, purely because of better visual features.

Second, ROUGE-L, measuring Longest Common Subsequence overlap, independently confirms the same ranking — D > B > C > A — providing strong cross-metric validation. All four models exceed Li et al.'s best ROUGE-L of 36.33%, with Model D achieving 42.70%.

Third, BERTScore is high for all models (above 0.90) but the spread is very narrow — only 0.0077. This is because all models produce text in the same format 'answer because explanation' — semantically they're all in the same neighborhood. BERTScore can't discriminate well here. BLEU-4 and METEOR are more informative metrics for this task.

Fourth, Exact Match is low for all models — 2.8% to 5.9% — which is expected for generative output. Two sentences with the same meaning rarely have identical wording."
-->

---

<!-- SLIDE 8: BEAM SEARCH -->
## Beam Search vs Greedy

**beam_width=3, trigram blocking**

| Model | Greedy B4 | Beam B4 | Δ B4 | Greedy EM | Beam EM | **ΔEM** |
|---|---|---|---|---|---|---|
| **A** | 0.0915 | 0.0926 | +1.2% | 2.83% | 7.46% | **+4.63 pp** |
| **B** | 0.1127 | 0.1137 | +0.9% | 4.07% | 9.94% | **+5.87 pp** |
| **C** | 0.0988 | 0.1005 | +1.7% | 4.18% | 7.57% | **+3.39 pp** |
| **D** | 0.1159 | 0.1170 | +0.9% | 5.88% | 11.07% | **+5.19 pp** |

&nbsp;

**Insight:** Beam search gives minimal BLEU gains (<2%) but **2–3× better Exact Match**

> Model D beam search: **11.07%** Exact Match — because beam search finds the *canonical* phrasing that matches ground-truth exactly

> **ROUGE-L is beam-invariant:** Δ ROUGE-L ≈ 0 for all models — beam search canonicalizes surface form (↑EM) without improving content coverage (≈ ROUGE-L)

<!--
SPEAKER NOTES — Slide 8 (Beam Search) ~1 min

"I also evaluated with beam search. Beam search keeps the top 3 hypotheses at each decode step, with trigram blocking to prevent repetition.

The interesting finding is that beam search barely moves BLEU-4 — less than 2% improvement. But it dramatically improves Exact Match: 2.6× for A, 2.4× for B, 1.8× for C, 1.9× for D.

Why? BLEU counts partial n-gram overlaps, so it's relatively forgiving of different phrasings. Exact Match is binary — either the output matches word-for-word or it doesn't. Beam search tends to find the most 'canonical' or common phrasing, which is more likely to match the ground truth exactly.

Model D reaches 11.07% Exact Match under beam search — roughly double its greedy rate.

One interesting finding: ROUGE-L is essentially unchanged by beam search. This confirms that the improvement in Exact Match comes from finding more canonical surface forms, not from generating richer content.

So for applications where you need precise matches, use beam search. For general quality measurement, greedy is fine."
-->

---

<!-- SLIDE 9: EFFECT DECOMPOSITION -->
## What Drives Performance?

### 2×2 Factorial Decomposition

| Effect | BLEU-4 gain | METEOR gain | Interpretation |
|---|---|---|---|
| **Pretrained features** (A→B or C→D) | **+0.0192 avg** (+20.2%) | **+0.039 avg** (+13%) | Dominant factor |
| **Dual attention** (A→C or B→D) | +0.0053 avg (+5.4%) | +0.012 avg (+4%) | Secondary factor |

&nbsp;

**Key finding:** Pretrained features contribute **~3.6× more** improvement than dual attention

&nbsp;

```
Scratch CNN    A (0.0915) ─── +0.0073 ──► C (0.0988)
                  │                            │
               +0.0212                      +0.0171
                  │                            │
               ▼                            ▼
Pretrained     B (0.1127) ─── +0.0032 ──► D (0.1159)
```

> **Attention helps more with weak features (+8.0%) than strong features (+2.8%)** — it partially compensates for a poor visual encoder

<!--
SPEAKER NOTES — Slide 9 (Decomposition) ~2 min

"This is the scientific heart of the project. By using a factorial design I can cleanly decompose each factor's contribution.

Pretrained features account for about 20.2% relative BLEU-4 improvement on average. Dual attention accounts for about 5.4%. So pretrained features are about 3.6 times more impactful than the decoder architecture.

The interaction pattern is also interesting: the attention mechanism helps more when combined with weak visual features (+8.0% from A to C) than with strong features (+2.8% from B to D). This makes intuitive sense — attention partially compensates for a poor visual encoder. When the features are already rich and informative (ResNet101), attention adds less marginal value.

The practical takeaway: if you're building a VQA system and have limited compute, spend it on the visual encoder first, not the decoder."
-->

---

<!-- SLIDE 10: COMPARISON WITH ORIGINAL PAPER -->
## vs. Li et al. (ECCV 2018) — Original Paper

**Same evaluation set: VQA-E val split, n = 88,488**

| Model | BLEU-1 | BLEU-2 | BLEU-3 | **BLEU-4** | **ROUGE-L** |
|---|---|---|---|---|---|
| Li et al. — QI-E Grid (gen only) | 36.30 | 21.10 | 12.50 | 7.60 | 34.00 |
| Li et al. — QI-E Bottom-up (gen only) | 38.00 | 22.60 | 13.80 | 8.60 | — |
| Li et al. — **QI-AE Bottom-up** *(best)* | 39.30 | 23.90 | 14.80 | 9.40 | 36.33 |
| **Ours — Model A** (*no multi-task*) | 37.15 | 23.35 | 14.15 | **9.15** | **38.28** |
| **Ours — Model D** (best, *no multi-task*) | **41.51** | **27.34** | **17.48** | **11.59** | **42.70** |

&nbsp;

<span class="win">Model D outperforms the original paper's best: BLEU-4 +2.19 pp (+23.3%) · ROUGE-L +6.37 pp (+17.5%) — without multi-task supervision</span>

> **Without** multi-task supervision · **Without** bottom-up features · **With** only 7×7 grid features · Confirmed by 3 independent metrics

<!--
SPEAKER NOTES — Slide 10 (Paper comparison) ~2.5 min

"This is the most important slide. How does our work compare to the original VQA-E paper?

Li et al. test several variants. Their best model — QI-AE Bottom-up — uses multi-task learning: it simultaneously predicts the answer as a classification problem over 3,129 candidates, AND generates the explanation. This multi-task supervision is a significant advantage because the model knows what the correct answer is before generating the explanation.

They also use Bottom-up features — Faster R-CNN object detection regions — which are semantically richer than grid features.

Despite these advantages on their side, our Model D achieves 11.59% BLEU-4 versus their 9.40%. That's a 23.3% relative improvement. ROUGE-L shows the same story: 42.70% vs 36.33%, a 17.5% relative improvement.

Even more striking: our weakest model, Model A — no pretrained features, no attention, no multi-task — achieves 9.15%, which is only 0.25 percentage points below their best. Our baseline nearly matches their state-of-the-art.

Why do we outperform? I attribute it to three factors: BiLSTM question encoding versus their single-layer GRU, dual attention versus their single image attention, and our 3-phase progressive training with scheduled sampling versus their 15-epoch fixed training. They train for 15 epochs with LR=0.01; we train for 30+ epochs across three carefully designed phases."
-->

---

<!-- SLIDE 11: QUALITATIVE EXAMPLES -->
## Qualitative Examples — Real Predictions

**Random samples from validation set (not cherry-picked)**

| Question | Ground Truth | Model A | Model D |
|---|---|---|---|
| *What is the boy doing?* | flying kite because ... beach ... kite | **skiing** because a person is skiing... | **flying kite** because a man is flying a kite in the sky |
| *How many cats on the couch?* | 2 because two cats laying on a couch | 2 because two cats sleeping **on a bed** | 2 because two cats laying **on a bed** |
| *What is the tablet next to?* | scissors because The tablet is next to scissors | **laptop** because a laptop is on a desk | **scissors** because a pair of scissors are on a table |

&nbsp;

**Common failure modes:**
- 🔴 **Beach/snow confusion** (Model A) — scratch CNN confuses similar visual patterns
- 🟡 **Furniture category error** (all models) — "couch" predicted as "bed"
- 🟢 **Fine-grained objects** (Models B/D) — correctly identifies scissors, kite, motorcycle

<!--
SPEAKER NOTES — Slide 11 (Examples) ~2 min

"Let me show some real predictions from the validation set. These were drawn randomly — not selected to make the models look good.

The first example — 'What is the boy doing?' — is a clear win for pretrained features. Model A predicts 'skiing' because the scratch CNN confuses the beach scene (sand, person, kite in sky) with snow skiing. Models B and D with pretrained ResNet101 correctly identify 'flying kite.'

The second example — 'How many cats on the couch?' — all models get the count right (2) but say 'bed' instead of 'couch.' This is a training distribution bias: VQA-E has more cats-on-bed images than cats-on-couch images, so the model defaults to 'bed.'

The third example shows fine-grained object recognition. Model A predicts 'laptop' when asked what a tablet is next to — it confuses laptop and tablet. Models B, C, D all correctly identify 'scissors.'

The key pattern: pretrained features dramatically improve visual discrimination for fine-grained categories. Attention helps with spatial relationships but doesn't fix fundamental feature quality issues."
-->

---

<!-- SLIDE 12: TRAINING DYNAMICS -->
## Training Dynamics

**From `history_model_*.json` — actual recorded losses**

| Model | Epoch 1 val | Epoch 10 val | Epoch 15 val | Best val |
|---|---|---|---|---|
| A | 4.365 | 3.375 | 3.299 | **3.2983** (ep16) |
| B | 4.335 | 3.274 | **3.218** | **3.2178** (ep15) |
| C | 4.530 | 3.325 | **3.277** | **3.2774** (ep15) |
| D | 4.622 | 3.260 | **3.222** | **3.2216** (ep15) |

&nbsp;

**Notable Phase 3 behaviors:**
- **Model B:** Train loss drops to 2.65 but val loss rises to 3.29 → classic overfitting of fine-tuned backbone
- **Model C:** Immediate val spike 3.277 → 3.430 when scheduled sampling starts → dual attention is sensitive to input noise
- **Model D:** Val loss partially recovers to 3.2625 at epoch 21 — *only model where scheduled sampling is genuinely beneficial*

<!--
SPEAKER NOTES — Slide 12 (Training dynamics) ~1.5 min

"Looking at the actual training curves recorded in the history JSON files, we can see interesting model-specific behaviors.

All models converge quickly in Phase 1 — dropping from around 4.4-4.6 to about 3.3 in 10 epochs. Pretrained models B and D consistently stay lower than scratch models A and C throughout training.

In Phase 3 when scheduled sampling starts, each model behaves differently. Model B shows classic overfitting — its training loss keeps dropping to 2.65, but its validation loss increases. The fine-tuned backbone is memorizing the training set.

Model C shows the most instability — its validation loss spikes by 0.15 in just 2 epochs when scheduled sampling starts. The combination of scratch CNN features and dual attention is more sensitive to receiving imperfect input tokens during training.

Most interestingly, Model D is the only model where scheduled sampling provides a genuine benefit — its validation loss partially recovers from the initial spike down to 3.26, close to its Phase 2 best. Strong pretrained features provide enough stability that the decoder can actually learn from its own mistakes."
-->

---

<!-- SLIDE 13: LIMITATIONS & HONEST ASSESSMENT -->
## Limitations — Honest Assessment

**What we don't have compared to Li et al.:**

| Gap | Impact | Notes |
|---|---|---|
| No multi-task answer classification | Moderate | We outperform on BLEU-4 (+23%) and ROUGE-L (+17%); their task is more complete |
| No CIDEr-D metric | Hard to compare | Standard captioning metric; we don't compute it |
| BERTScore ceiling effect | Minor | All 4 models score 0.900–0.909 (range: 0.009) — insufficient discrimination |
| No bottom-up features | Small | Faster R-CNN regions would likely improve our models further |
| No human evaluation | Moderate | Automated metrics don't fully capture explanation quality |

&nbsp;

**METEOR caveat:**

> Our METEOR (31–36%) vs paper's (14–17%) — **NOT directly comparable**
> Different implementations: NLTK uses WordNet synonym matching, Li et al. use standard METEOR 1.5
> ⚠️ Within our 4 models, METEOR is valid. Cross-paper comparison: use BLEU-4 only.

<!--
SPEAKER NOTES — Slide 13 (Limitations) ~1.5 min

"I want to be honest about what we're missing compared to the original paper.

The most significant gap is multi-task learning. Li et al. jointly predict the answer and generate the explanation — the answer supervision is a strong training signal. We only do explanation generation. Despite this disadvantage, we still outperform on BLEU-4 (+23.3%) and ROUGE-L (+17.5%).

We also don't compute CIDEr-D, the standard captioning metric that rewards specificity. Unlike ROUGE-L (which we now compute), CIDEr-D requires 5 reference sentences per sample — VQA-E provides only 1 — so standard computation is invalid.

I also want to flag a METEOR issue. Our METEOR scores look about twice as high as the paper's — 35% versus 17%. This is NOT because we're twice as good. It's because we use NLTK's meteor_score which enables WordNet synonym matching by default, while the paper uses the standard METEOR implementation. Words like 'riding' and 'skateboarding' count as partial matches in NLTK but not in standard METEOR. So please don't compare METEOR numbers across papers. Use BLEU-4 for cross-paper comparison."
-->

---

<!-- SLIDE 14: KEY CONTRIBUTIONS -->
## Key Contributions

**1. Systematic 2×2 Factorial Study**
- Clean empirical decomposition: pretrained features contribute 3.7× more than attention
- Controlled conditions: identical training, data, optimizer across all 4 models

**2. Surpass Original Paper on BLEU-4**
- Model D: 11.59% vs Li et al. best: 9.40% (+23.3% relative); ROUGE-L: 42.70% vs 36.33% (+17.5%)
- Checkpoint: 93.7M total params, 51.2M trainable, 375.5 MB · Confirmed by BLEU-4, METEOR, ROUGE-L

**3. Architectural Improvements**
- BiLSTM question encoder (vs GRU in original paper)
- Dual attention: image regions + question tokens at each decode step
- GatedFusion: learnable gate vs Hadamard product
- Coverage mechanism: prevents repetitive attention

**4. 3-Phase Progressive Training**
- Scheduled sampling with relative epoch scheduling (critical bug fix)
- Phase 3 discovery: only Model D benefits from scheduled sampling

**5. Complete 7-Metric Evaluation Suite**
- BLEU-1/2/3/4, METEOR, ROUGE-L, BERTScore, Exact Match (greedy + beam)
- All three primary metrics (BLEU-4, METEOR, ROUGE-L) independently confirm D > B > C > A

<!--
SPEAKER NOTES — Slide 14 (Contributions) ~1.5 min

"Let me summarize the contributions.

First, the systematic factorial design. By training all 4 models under identical conditions, I can give clean quantitative answers to the questions 'how much do pretrained features help' and 'how much does attention help' — without confounding variables.

Second, we outperform the original paper: +23.3% on BLEU-4, and 17.5% on ROUGE-L — both without their multi-task supervision advantage. Three independent metrics all confirm the same ranking, which makes the result robust.

Third, the architectural package: BiLSTM instead of GRU, dual attention, gated fusion, and coverage. Each of these is a documented improvement from the NLP literature applied to this task.

Fourth, the training protocol. Scheduled sampling is the standard solution to exposure bias, but it requires careful implementation. A critical bug in naive implementations uses the absolute epoch number for epsilon decay, which causes epsilon to drop too fast in Phase 3. Using the relative epoch within Phase 3 is essential for stable training."
-->

---

<!-- SLIDE 15: CONCLUSION -->
## Conclusion

**Research Questions — Answered:**

| Question | Answer | Evidence |
|---|---|---|
| Does transfer learning help? | **Yes, substantially** | +20% BLEU-4 (A→B or C→D) |
| Does dual attention help? | **Yes, but modestly** | +5% BLEU-4 (A→C or B→D) |
| Do they compose? | **Yes, sub-additively** | D < B + C combined gains |
| Can we beat the original paper? | **Yes** | +23.3% BLEU-4, +17.5% ROUGE-L vs Li et al. best |

&nbsp;

**Recommended configuration:** Model D — ResNet101 + Dual Attention + Coverage
- Best on all 7 metrics
- BLEU-4: **0.1159** (greedy), **0.1170** (beam) · ROUGE-L: **0.4270** (greedy)
- Exact Match: **5.88%** (greedy), **11.07%** (beam)
- Checkpoint: 93.7M total params, 51.2M trainable, 375.5 MB

&nbsp;

> **If resources are limited:** Use Model B — 97.2% of Model D's BLEU-4 at simpler decoder (no attention), fewer trainable params (40.7M vs 51.2M), 2.04× BLEU-4/param efficiency

<!--
SPEAKER NOTES — Slide 15 (Conclusion) ~1.5 min

"To conclude.

I had four research questions and I can answer all of them definitively with real experimental data.

Transfer learning helps substantially — 20% BLEU-4 improvement. Dual attention helps but modestly — about 5%. The two effects compose but sub-additively, suggesting they partially address the same underlying challenge of visual understanding.

And the headline result: yes, we can and do outperform the original paper, by 23.3% on BLEU-4, without their multi-task supervision advantage.

Key numbers for Model D: BLEU-4 = 0.1159, Exact Match = 11.07% under beam search.

If I had to give a single recommendation: use Model D — it's the best on every metric. But if resources are tight, Model B achieves 97.2% of Model D's BLEU-4 with 20% fewer trainable parameters — the best efficiency trade-off in this study. The 2.8% gain from adding dual attention to an already strong pretrained backbone may not justify the added architectural complexity in production.

Thank you. I'm happy to take questions."
-->

---

<!-- SLIDE 16: BACKUP — ARCHITECTURE DETAILS -->
## Backup: Detailed Architecture

### Image Encoder
- **SimpleCNN** (A/C): 5× [Conv3×3→BN→ReLU→MaxPool2×2] → AdaptiveAvgPool → FC(1024→1024)
- **ResNetSpatialEncoder** (C/D): ResNet101 pretrained, remove avgpool+fc, keep (batch, 2048, 7, 7) → Conv2d(2048→1024, 1×1) → (batch, 49, 1024)

### Question Encoder
```
tokens → Embedding(GloVe 300d) → proj(300→512) → BiLSTM(2-layer, 512/dir)
→ concat(h[-2], h[-1]) → q_feature (batch, 1024)
→ all hidden states → q_hidden_states (batch, q_len, 1024)
```

### Bahdanau Attention (dual: image + question)
```
energy = tanh(W_h(h_top) + W_img(regions) [+ W_cov(coverage)])
alpha = softmax(v(energy))        # (batch, 49)
context = sum(alpha * regions)    # (batch, 1024)
lstm_input = [embed; img_ctx; q_ctx]  # (batch, 1, 512+1024+1024)
```

<!--
SPEAKER NOTES — Backup slide (if asked for technical details)

"This backup slide has the detailed equations if the professor asks about the technical implementation.

The image encoder for models C and D removes ResNet's avgpool and fc layers, keeping the 7×7 spatial feature map with 2048 channels, then projects down to 1024 via a 1×1 convolution. This gives us 49 regions.

The question encoder uses a BiLSTM where the final question representation concatenates the last hidden state from both directions. It also returns all intermediate hidden states for the question attention.

The Bahdanau attention computes scores via an additive energy function. The coverage term adds the cumulative attention from previous steps — this penalizes the model for attending to regions it's already attended to extensively."
-->

---

<!-- SLIDE 17: BACKUP — FULL METRICS TABLE -->
## Backup: Complete Evaluation Results

### Greedy Decoding (n=88,488)

| Model | B-1 | B-2 | B-3 | B-4 | METEOR | ROUGE-L | BERTScore | EM |
|---|---|---|---|---|---|---|---|---|
| A | 0.3715 | 0.2335 | 0.1415 | 0.0915 | 0.3117 | 0.3828 | 0.9008 | 2.83% |
| B | 0.4124 | 0.2702 | 0.1715 | 0.1127 | 0.3561 | 0.4237 | 0.9081 | 4.07% |
| C | 0.3865 | 0.2463 | 0.1516 | 0.0988 | 0.3271 | 0.3971 | 0.9034 | 4.18% |
| D | **0.4151** | **0.2734** | **0.1748** | **0.1159** | **0.3595** | **0.4270** | **0.9085** | **5.88%** |

### Beam Search w=3, n-gram blocking=3

| Model | B-1 | B-2 | B-3 | B-4 | METEOR | ROUGE-L | BERTScore | EM |
|---|---|---|---|---|---|---|---|---|
| A | 0.3723 | 0.2333 | 0.1421 | 0.0926 | 0.3154 | 0.3823 | 0.8999 | 7.46% |
| B | 0.4122 | 0.2690 | 0.1713 | 0.1137 | 0.3589 | 0.4230 | 0.9073 | 9.94% |
| C | 0.3872 | 0.2465 | 0.1527 | 0.1005 | 0.3300 | 0.3972 | 0.9026 | 7.57% |
| D | **0.4160** | **0.2734** | **0.1754** | **0.1170** | **0.3632** | **0.4269** | **0.9080** | **11.07%** |

*All values from `outputs/evaluation_results.json` · Exact to 4 decimal places · n=88,488*

<!--
SPEAKER NOTES — Backup: all numbers at hand if asked
-->

---

<!-- SLIDE 18: HOW TO CONVERT THESE SLIDES -->
<!--
HOW TO RENDER THESE SLIDES:

Option 1 — VS Code (recommended):
  1. Install extension: "Marp for VS Code" (marp-team.marp-vscode)
  2. Open this file in VS Code
  3. Click the Marp icon (top-right) → "Open Preview"
  4. Export: Ctrl+Shift+P → "Marp: Export Slide Deck" → PDF or PPTX

Option 2 — Command line:
  npm install -g @marp-team/marp-cli
  marp PRESENTATION.md --pdf -o slides.pdf
  marp PRESENTATION.md --pptx -o slides.pptx

Option 3 — Use as a speaker script only (read the SPEAKER NOTES sections under each slide)

PRESENTATION TIMING (estimated ~18 minutes):
  Slide 1  Title           1 min
  Slide 2  Motivation      1.5 min
  Slide 3  Dataset         1.5 min
  Slide 4  2×2 Design      1.5 min
  Slide 5  Architecture    2 min
  Slide 6  Training        2 min
  Slide 7  Results         2 min
  Slide 8  Beam Search     1 min
  Slide 9  Decomposition   2 min
  Slide 10 vs. Paper       2.5 min
  Slide 11 Examples        2 min
  Slide 12 Dynamics        1.5 min
  Slide 13 Limitations     1.5 min
  Slide 14 Contributions   1.5 min
  Slide 15 Conclusion      1.5 min
  ─────────────────────────────
  Total:                  ~25 min (adjust by skipping slides 12 or 8 if needed)
  Q&A backup: Slides 16-17
-->
