---
title: "Model G Data Strategy: A Zero-Cost Data Engineering Blueprint for Long-Form Generative Visual Question Answering"
subtitle: "Technical Specification · Data Pipeline Design · Training Curriculum · Evaluation Protocol"
author: "VQA-E Research Team"
date: "March 2026"
abstract: |
  This document presents a comprehensive, zero-monetary-cost data engineering strategy
  for training Model G — a CNN-LSTM generative VQA system that produces explanatory
  answers of 15–30 words. We identify data scarcity and quality as the primary
  bottleneck limiting performance, superseding architectural improvements in marginal
  return. The strategy consolidates four free data sources (VQA-E filtered, VQA-X,
  A-OKVQA rationales, and COCO-caption-derived synthetic QA) into a unified
  ~225K-sample training pool with 73K human-written explanations as the quality
  backbone. We introduce three critical training innovations: (1) a length-conditioned
  decoding mechanism that prevents short-answer collapse when mixing heterogeneous
  data sources, (2) a five-stage visual grounding filter that reduces hallucination
  noise in VQA-E by ~50%, and (3) a four-phase curriculum that progressively
  transitions the LSTM decoder from short-form alignment to long-form explanation
  generation with SCST reinforcement learning. The document provides complete
  preprocessing specifications, format normalization rules, dataset compatibility
  analysis, training hyperparameters, and a multi-metric evaluation protocol
  including CIDEr-D, SPICE, and LAVE. Total engineering effort is estimated at
  25–35 hours with zero external API costs.
geometry: margin=1in
fontsize: 11pt
toc: true
toc-depth: 3
numbersections: true
---

\newpage

# The Data Bottleneck in Generative VQA

## Why Architecture Alone Cannot Reach SOTA

Model G's architecture — BUTD Faster R-CNN features, MUTAN Tucker fusion,
LayerNorm BiLSTM with Highway connections, dual MHCA, three-way Pointer-Generator,
and SCST reinforcement learning — represents the near-ceiling of what the CNN-LSTM
paradigm can achieve structurally. Adding further architectural components yields
diminishing returns: the transition from Model D (BLEU-4: 0.1159) to Model E/F
involved 8 architectural tiers but each incremental tier contributed smaller gains.

The primary remaining lever is **data quality and diversity**. Three fundamental
data problems constrain the current system:

**Problem 1: Generative ≠ Classification.** The project's objective is *Generative
VQA* — the model must produce natural language answers, not select from a fixed
set. A model that outputs "yes" or "blue" is performing classification with extra
computation. The target output is a complete explanatory sentence of 15–30 words,
e.g., *"The man is riding a bicycle because he appears to be commuting along a
city street during the daytime."*

**Problem 2: Short-Answer Collapse.** The current training pipeline uses VQA v2.0
(~660K samples, answers averaging 1.1 words) for Phase 1 warm-up. This creates a
powerful *early termination prior*: the LSTM decoder learns to emit `<end>` after
1–3 tokens. Subsequent phases on VQA-E partially correct this, but the bias
persists — the decoder favors short, safe outputs over long explanatory ones.

**Problem 3: VQA-E Quality Ceiling.** VQA-E's ~210K explanations were
auto-generated from COCO captions, not written by humans. The e-ViL benchmark
(Kayser et al., ICCV 2021) found that only 66.5% of VQA-E ground-truth
explanations were rated acceptable by human evaluators. Training on 70K+ noisy
samples teaches the model to hallucinate and produce generic, caption-like text
rather than question-specific explanations.

## The Core Insight: Quality Over Quantity

Our data strategy is guided by a single empirical finding: **73K human-written
explanation samples (VQA-X + A-OKVQA) provide more training signal than 210K
auto-generated VQA-E samples.** The A-OKVQA rationales, in particular, require
genuine commonsense reasoning and average 15–20 words — precisely matching our
target output length. By combining these high-quality sources with aggressively
filtered VQA-E and rule-based synthetic augmentation, we construct a training
pool that is both larger and cleaner than the current setup.


\newpage

# Dataset Audit: Available Free Resources

This section catalogs every publicly available dataset relevant to long-form
generative VQA over natural images, with compatibility analysis for our
COCO-based pipeline.

## Tier 1: Primary Explanation Datasets (COCO-based, Free)

### VQA-E (Li et al., ECCV 2018) — Currently in Use

| Property | Value |
|:---------|:------|
| Size | ~210K train / ~88K val |
| Images | COCO 2014 (train2014 + val2014) |
| Format | `{answer} because {explanation}` |
| Explanation source | Auto-generated from COCO captions |
| Avg explanation length | ~11 words |
| Human quality rating | 66.5% acceptable (e-ViL benchmark) |
| Download | Already in project (`data/vqa_e/`) |

**Assessment**: Large but noisy. The auto-generation process often produces
explanations that restate the caption rather than answering the question.
Example failure: Q: *"What sport is being played?"* → VQA-E: *"baseball because
there are players on a field"* (generic caption, not explanatory). After
aggressive filtering, ~80–100K samples are usable.

### VQA-X (Park et al., CVPR 2018) — Highest Priority Addition

| Property | Value |
|:---------|:------|
| Size | 29,459 train / 1,459 val / 1,968 test |
| Images | COCO 2014 (~28K unique images) |
| Format | Free-form human explanation per QA pair |
| Explanation source | **Human-written** (AMT workers) |
| Avg explanation length | ~11 words |
| Human quality rating | **91.4% acceptable** (e-ViL benchmark) |
| Download | `github.com/Seth-Park/MultimodalExplanations` |

**Assessment**: Gold standard for VQA explanations. Despite smaller size, the
quality differential is massive: 91.4% vs 66.5% acceptance rate. Every sample
is a COCO image with zero domain gap. The explanation format naturally aligns
with our target: workers wrote explanations starting with "I can tell [answer]
because..." which maps directly to our `{answer} because {explanation}` format.

**Compatibility**: Perfect. Same COCO images, same question sources (derived from
VQA v2.0), same format. Integration requires only JSON reformatting.

### A-OKVQA (Schwenk et al., ECCV 2022) — Second Priority Addition

| Property | Value |
|:---------|:------|
| Size | 17,056 train / 1,145 val (×3 rationales each) |
| Images | COCO 2017 |
| Format | Question + direct answer + 3 human rationales |
| Explanation source | **Human-written** (AMT workers) |
| Avg rationale length | **15–20 words** |
| Human quality rating | High (curated, reviewed) |
| Download | `github.com/allenai/aokvqa` |

**Assessment**: The most underutilized resource for this project. A-OKVQA
rationales are *longer* and *more reasoning-heavy* than VQA-X, because the
questions require world knowledge and commonsense inference. Example:
Q: *"What would happen if you touched the animal in the center?"*
Rationale: *"The animal is a porcupine which has sharp quills that would prick
your hand if you tried to touch it."* (22 words)

This is exactly the kind of explanatory reasoning we want Model G to produce.

**Compatibility**: COCO 2017 images overlap ~95% with COCO 2014. The ~5% unique
to 2017 require downloading COCO 2017 val images (~1GB) or mapping to 2014
equivalents. Rationale format needs conversion: extract rationales → prepend
answer → format as `{answer} because {rationale}`.

**Key advantage**: 3 rationales per question → 51K explanation instances from
17K questions. This provides *natural data augmentation* — the model sees three
different ways to explain the same answer, learning paraphrasing and diverse
reasoning patterns.

## Tier 2: Supplementary Sources (Free, Some Preprocessing Required)

### COCO Captions → Synthetic QA Pairs (Self-Generated)

| Property | Value |
|:---------|:------|
| Size | ~120K COCO images × 5 captions each |
| Images | COCO 2014 (same as VQA-E) |
| Format | Raw captions → template-based QA conversion |
| Explanation source | **Rule-based** from human captions |
| Quality | Medium (template-dependent) |
| Cost | Free (CPU time only) |

**Assessment**: The project already has `generate_synthetic_qa.py` with
existence/count/location templates. Extending this to 8–10 template families
(action, attribute, spatial, comparison, material, counting, yes/no, what/who)
can yield ~40–60K additional QA-explanation pairs.

The quality ceiling is limited by template rigidity, but these samples serve
a critical role: they provide *vocabulary breadth* and *visual grounding
practice* for the Pointer-Generator, teaching it to copy object names from
visual labels into generated text.

### Local VLM Generation (Optional, GPU Time Only)

| Property | Value |
|:---------|:------|
| Models | LLaVA-1.5-7B (4-bit), Qwen2-VL-2B, or InternVL2-2B |
| VRAM | 6–8 GB at 4-bit quantization |
| Speed | ~20–30 images/minute on RTX 5070 Ti |
| Output | ~10–20K explanation samples (after filtering) |
| Cost | Free (overnight GPU run, ~8–12 hours) |

**Assessment**: Optional but valuable. Running a small open-source VLM locally
produces more diverse and natural explanations than templates. The 7B LLaVA at
4-bit quantization fits in 6–8 GB VRAM, leaving enough room for batch inference.

**Critical**: All VLM-generated samples must pass the same visual grounding
filter as VQA-E (Section 4) to remove hallucinations. Expected pass rate: ~65%.

## Tier 3: Evaluated and Rejected

| Dataset | Reason for Rejection |
|:--------|:--------------------|
| **VCR** (290K, movie stills) | Domain gap: movie frames with `[person1]` tags ≠ COCO natural images. Would require Faster R-CNN re-extraction on non-COCO images. |
| **e-SNLI-VE** (430K, Flickr30K) | Task mismatch: visual entailment explanations, not QA. Format conversion lossy. |
| **ScienceQA** (21K, educational) | Domain mismatch: scientific diagrams ≠ natural images. |
| **VisualCOMET** (1.4M, movies) | Task mismatch: temporal inference, not QA. Movie stills. |
| **TextVQA** (45K) | No explanations — short OCR-based answers only. |
| **OK-VQA** (14K) | Superseded by A-OKVQA. No rationale annotations. |
| **GQA** (22M) | 1-word answers from scene graphs. Would massively amplify short-answer collapse. |
| **LLaVA-Instruct-150K** | Requires API cost for generation. Multi-sentence responses incompatible with LSTM max_len. Distribution mismatch with COCO-trained Faster R-CNN. |


\newpage

# Data Quality: The Five-Stage Grounding Filter

Raw data from any source contains noise. For generative VQA, the most dangerous
form of noise is *visual hallucination* — explanations that mention objects not
present in the image. The Pointer-Generator Network amplifies this problem:
if trained on hallucinated text, the model learns to generate plausible-sounding
but visually ungrounded explanations.

## Filter Architecture

We apply a five-stage sequential filter to all explanation data. Each stage is
independent and produces a binary keep/discard decision. A sample must pass all
five stages to enter the training pool.

### Stage 1: Length Gate

$$\text{pass}_1(y) = \begin{cases} \text{True} & \text{if } 5 \leq |y_{\text{explanation}}| \leq 35 \\ \text{False} & \text{otherwise} \end{cases}$$

where $|y_{\text{explanation}}|$ is the word count of the explanation portion
(after "because"). Explanations shorter than 5 words are too terse to be
meaningful ("because it is there"). Explanations longer than 35 words exceed
the LSTM decoder's practical generation capacity and will be truncated during
training, causing incomplete-sentence artifacts.

**Expected rejection rate**: ~8% of VQA-E, ~2% of VQA-X, ~5% of A-OKVQA.

### Stage 2: Copy-of-Question Detector

Compute token-level Jaccard overlap between the question $Q$ and explanation $E$:

$$J(Q, E) = \frac{|\text{tokens}(Q) \cap \text{tokens}(E)|}{|\text{tokens}(Q) \cup \text{tokens}(E)|}$$

$$\text{pass}_2(y) = \begin{cases} \text{True} & \text{if } J(Q, E) < 0.6 \\ \text{False} & \text{otherwise} \end{cases}$$

Explanations that merely rephrase the question provide no additional information.
Example: Q: *"Is the man riding a horse?"* → E: *"yes because the man is riding
a horse"* → $J = 0.78$ → **DISCARD**.

**Expected rejection rate**: ~12% of VQA-E, ~3% of VQA-X, ~1% of A-OKVQA.

### Stage 3: Visual Grounding Check

This is the critical filter. Using the BUTD Faster R-CNN detected labels
$\mathcal{L}_V = \{l_1, \ldots, l_k\}$ already extracted for Model F:

**Step 3a**: Extract content nouns from explanation using spaCy lemmatization:

$$\mathcal{N}_E = \{w \in E \mid \text{POS}(w) \in \{\text{NOUN}, \text{PROPN}\} \text{ and } w \notin \mathcal{S}\}$$

where $\mathcal{S}$ is a stopword set including generic nouns
("thing", "stuff", "way", "time", "picture", "image", "photo").

**Step 3b**: Compute grounding ratio:

$$r_{\text{ground}} = \frac{|\{n \in \mathcal{N}_E \mid \exists\, l \in \mathcal{L}_V : \text{match}(n, l)\}|}{|\mathcal{N}_E|}$$

where $\text{match}(n, l)$ is true if $n$ equals $l$, or if $n$ is a WordNet
synonym/hypernym of $l$ within distance 2 (e.g., "canine" matches "dog").

$$\text{pass}_3(y) = \begin{cases} \text{True} & \text{if } |\mathcal{N}_E| < 2 \text{ (too few nouns to judge)} \\ \text{True} & \text{if } r_{\text{ground}} \geq 0.3 \\ \text{False} & \text{otherwise} \end{cases}$$

The threshold 0.3 is deliberately lenient: explanations may legitimately mention
abstract concepts ("sport", "fun", "danger") not detectable by Faster R-CNN.
The filter targets egregious hallucinations where *none* of the mentioned concrete
objects appear in the image.

**Expected rejection rate**: ~25% of VQA-E, ~5% of VQA-X, ~8% of A-OKVQA.

**Implementation**: The project already has `filter_hallucinations.py` (Tier D5)
which implements a similar pipeline. Extend it with the WordNet synonym matching.

### Stage 4: Answer Consistency Check

Verify that the explanation's answer component matches the annotated answer:

$$\text{pass}_4(y) = \begin{cases} \text{True} & \text{if } a_{\text{annotated}} \in \text{first\_5\_words}(y_{\text{full}}) \\ \text{False} & \text{otherwise} \end{cases}$$

This catches cases where the explanation contradicts or ignores the answer.
Example: Answer: *"tennis"*, Explanation: *"baseball because there are players
on a field"* → **DISCARD**.

**Expected rejection rate**: ~3% of VQA-E, ~1% of VQA-X, ~2% of A-OKVQA.

### Stage 5: Deduplication

Remove near-duplicate explanations within the same image using MinHash with
Jaccard threshold 0.85. This targets VQA-E's tendency to produce identical
caption-derived explanations for different questions about the same image.

**Expected rejection rate**: ~8% of VQA-E, ~1% of VQA-X, ~0% of A-OKVQA.

## Expected Yield After Filtering

| Dataset | Raw Input | After Filter | Pass Rate |
|:--------|:---------|:------------|:---------|
| VQA-E | 210,000 | ~95,000 | ~45% |
| VQA-X | 29,459 | ~27,000 | ~92% |
| A-OKVQA (×3 rationales) | 51,168 | ~45,000 | ~88% |
| Synthetic (templates) | ~60,000 | ~40,000 | ~67% |
| Local VLM (optional) | ~20,000 | ~12,000 | ~60% |
| **Total** | **~370,000** | **~219,000** | **~59%** |

The filtered pool contains **~72K human-written explanations** (VQA-X + A-OKVQA)
serving as the quality backbone, supplemented by ~95K filtered VQA-E and ~40K
synthetic samples for vocabulary breadth. If local VLM generation is included,
the pool reaches ~231K.


\newpage

# Format Normalization

All datasets must be normalized to a single JSON format compatible with the
existing `VQAEDataset` and `BUTDDataset` classes.

## Target Format

```json
{
  "img_id": 123456,
  "question": "What is the man doing?",
  "multiple_choice_answer": "riding a bicycle",
  "explanation": [
    "he appears to be commuting along a city street",
    "he is pedaling a bike down a road near buildings"
  ],
  "source": "vqa_x",
  "length_bin": "long"
}
```

## Per-Dataset Conversion Rules

### VQA-E → Unified Format

No structural change needed. Add `"source": "vqa_e"` and compute `"length_bin"`.

### VQA-X → Unified Format

VQA-X raw format:
```json
{
  "question_id": 262148000,
  "image_id": 262148,
  "question": "Where is he looking?",
  "answers": [{"answer": "down"}],
  "explanation": ["He is looking down at the laptop screen"]
}
```

Conversion:
- `img_id` ← `image_id`
- `multiple_choice_answer` ← `answers[0]["answer"]`
- `explanation` ← `[explanation[0]]` (wrap in list for consistency)
- If explanation does not start with answer, prepend: `"{answer} because {explanation}"`

### A-OKVQA → Unified Format

A-OKVQA raw format:
```json
{
  "question_id": "22MexNkBPpdZGX54...",
  "image_id": 299640,
  "question": "What is the device on the left side used for?",
  "choices": ["cleaning", "heating", "cooking", "entertainment"],
  "correct_choice_idx": 2,
  "direct_answers": ["cooking", "cooking", "cook", ...],
  "rationales": [
    "The device is a stove which is used for cooking food",
    "It is a gas stove commonly found in kitchens for cooking",
    "The left device appears to be a cooking stove or range"
  ]
}
```

Conversion:
- `img_id` ← `image_id`
- `multiple_choice_answer` ← mode of `direct_answers`
- For each rationale $r_i$ in `rationales`:
  - If $r_i$ does not contain the answer, prepend: `"{answer} because {r_i}"`
  - Else, insert "because" after the answer mention
- `explanation` ← list of all 3 converted rationales
- **Note**: COCO 2017 `image_id` values overlap with COCO 2014 for ~95% of
  images. For the remaining ~5%, maintain a mapping table `coco2017_to_2014.json`
  or download the additional COCO 2017 val images (~1GB).

### Synthetic (Template-Generated) → Unified Format

Already in correct format from `generate_synthetic_qa.py`. Add
`"source": "synthetic"` and `"_synthetic": true` flag.

## Length Bin Assignment

Every sample receives a `length_bin` label based on the full answer-explanation
length (in tokens after vocabulary numericalization):

$$\text{length\_bin}(y) = \begin{cases} \texttt{"short"} & \text{if } |y| \leq 5 \\ \texttt{"medium"} & \text{if } 6 \leq |y| \leq 14 \\ \texttt{"long"} & \text{if } |y| \geq 15 \end{cases}$$

These bins are used for length-conditioned decoding (Section 6).


\newpage

# Synthetic Data Generation: Template Expansion

## Current Template Coverage

The existing `generate_synthetic_qa.py` covers three question types:

| Type | Example Q | Example A+E |
|:-----|:----------|:-----------|
| Existence | "Is there a dog in the image?" | "yes because there is a dog visible in the scene" |
| Count | "How many cats are there?" | "two because there are two cats sitting together" |
| Location | "Where is the person?" | "on the sidewalk because the person is standing on the sidewalk near a building" |

## Proposed Template Expansion

We propose expanding to 10 template families, targeting diverse question types
that require explanatory answers:

### Family 1: Action (NEW)

**Source**: COCO captions contain verb phrases ("a man riding a bike",
"children playing in a park").

**Extraction**: Parse captions with spaCy dependency parser. Extract
`(subject, verb, object)` triples.

**Templates**:
- Q: "What is the {subject} doing?"
- A: "{verb-ing} {object} because the {subject} can be seen {verb-ing} {object} in the image"

**Example**: Caption: "A woman walking her dog along the beach"
→ Q: "What is the woman doing?" → A: "walking her dog because the woman can be
seen walking her dog along the beach"

### Family 2: Attribute (NEW)

**Source**: COCO captions mention colors, materials, sizes.

**Extraction**: POS-tag for adjectives modifying nouns.

**Templates**:
- Q: "What color is the {noun}?"
- A: "{adjective} because the {noun} in the image appears to be {adjective}"

### Family 3: Comparison (NEW)

**Source**: Images with multiple detected objects.

**Templates**:
- Q: "What is bigger, the {obj1} or the {obj2}?"
- A: "the {obj_larger} because the {obj_larger} takes up more area in the image than the {obj_smaller}"

**Note**: Use bounding box area from BUTD features to determine relative size.

### Family 4: Yes/No with Reasoning (NEW)

**Templates**:
- Q: "Is this photo taken indoors?"
- A: "no because the image shows {outdoor_objects} which are typically found outdoors"

**Object lists**: Classify COCO objects into indoor/outdoor categories.

### Family 5: Spatial Relationship (EXPANDED)

**Source**: BUTD bounding box coordinates.

**Templates**:
- Q: "What is to the left of the {obj1}?"
- A: "the {obj2} because the {obj2} is positioned to the left of the {obj1} in the image"

### Families 6–10: Activity, Weather, Scene Type, Emotion, Purpose

Similar template structures using COCO caption parsing and object category
mappings. Each family targets a specific reasoning skill.

## Quality Control for Synthetic Data

All template-generated samples pass through the same five-stage filter
(Section 3). Additionally:

- **Grammar check**: Run a simple rule-based grammar validator (subject-verb
  agreement, article usage).
- **Diversity enforcement**: No more than 3 synthetic samples per image.
- **Template diversity**: Each image's synthetic samples must come from at
  least 2 different template families.

## Expected Yield

| Template Family | Raw Generated | After Filter |
|:---------------|:-------------|:------------|
| Existence (existing) | ~15K | ~12K |
| Count (existing) | ~10K | ~7K |
| Location (existing) | ~8K | ~5K |
| Action (new) | ~12K | ~8K |
| Attribute (new) | ~6K | ~4K |
| Comparison (new) | ~4K | ~2K |
| Yes/No reasoning (new) | ~5K | ~3K |
| **Total** | **~60K** | **~41K** |


\newpage

# Length-Conditioned Decoding

## The Short-Answer Collapse Problem

When training on mixed data (VQA v2.0 short answers + VQA-E long explanations),
the LSTM decoder faces a **bimodal length distribution**. The cross-entropy loss
gradient from 660K short-answer samples overwhelms the gradient from 210K
explanation samples, causing the decoder to converge on the short-answer mode.

Empirically, this manifests as:
- Generated answers average 3–5 words when 15–25 is desired
- The `<end>` token probability spikes after 2–3 tokens
- Beam search produces repetitive short completions

## Solution: Learned Length Embeddings

We introduce a **length-conditioned generation** mechanism inspired by Kikuchi
et al. (EMNLP 2016), adapted for the VQA-E decoder.

### Architecture Modification

Add a learnable embedding layer with 3 bins:

$$E_{\text{len}} \in \mathbb{R}^{3 \times d_{\text{len}}}$$

where $d_{\text{len}} = 64$ and the 3 bins correspond to `SHORT` (1–5 words),
`MEDIUM` (6–14 words), `LONG` (15–35 words).

At each decode step $t$, the length embedding is concatenated to the decoder
input:

$$x_t^{\text{new}} = [e_t; c_t^{\text{img}}; c_t^{Q}; E_{\text{len}}[\text{bin}]] \in \mathbb{R}^{2560 + 64} = \mathbb{R}^{2624}$$

This requires updating the LSTM input dimension from 2560 to 2624 — a trivial
change in `LSTMDecoderWithAttention.__init__`.

### Training Protocol

During training, each sample's length bin is computed from the ground-truth
target sequence length:

- VQA v2.0 samples → `SHORT` (ground truth is 1–3 words)
- VQA-E/VQA-X/A-OKVQA samples → `LONG` (ground truth is 15–30 words)
- Template-generated samples → Determined by actual length

### Inference Protocol

At inference time, **always feed `LONG`**. This conditions the decoder to
produce explanatory-length outputs regardless of question type.

Additionally, apply a **minimum-length constraint** during beam search:

$$P(y_t = \texttt{<end>} \mid t < t_{\min}) = 0$$

where $t_{\min} = 8$ tokens. This prevents premature termination while still
allowing the model to produce natural endpoints after the minimum length.

### Per-Token Loss Normalization

A complementary fix: normalize the cross-entropy loss by sequence length $T$
rather than summing:

$$\mathcal{L}_{\text{CE}}^{\text{norm}} = -\frac{1}{T} \sum_{t=1}^{T} \log P(y_t^* \mid y_{<t}, I, Q)$$

Without this normalization, short sequences (VQA v2.0, $T \approx 3$) produce
lower total loss than long sequences (VQA-E, $T \approx 20$), biasing the
optimizer toward learning short-output patterns. Per-token normalization
equalizes the gradient contribution regardless of sequence length.

### Impact Analysis

| Mechanism | Prevents | Implementation Effort |
|:----------|:---------|:---------------------|
| Length embedding | Short-answer mode collapse | 2 hours (1 embedding layer + concat) |
| Min-length constraint | Premature `<end>` at inference | 30 minutes (beam search modification) |
| Per-token loss norm | Short-sequence gradient dominance | 15 minutes (divide loss by $T$) |

All three mechanisms are orthogonal and should be applied simultaneously.


\newpage

# Training Pipeline: Four-Phase Curriculum

## Design Rationale

The curriculum follows a **progressive complexity** principle: the model first
learns visual-linguistic alignment on simple QA, then transitions to explanation
generation, then self-corrects via scheduled sampling, and finally optimizes
directly for human-correlated metrics via RL. Each phase uses a different data
mixture optimized for its learning objective.

## Phase 1: Alignment Warm-Up (15 epochs)

**Objective**: Learn visual grounding, basic QA, and vocabulary alignment.

| Parameter | Value |
|:----------|:------|
| Data mix | 40% VQA v2.0 (`SHORT`) + 30% VQA-E filtered (`LONG`) + 30% A-OKVQA (`LONG`) |
| Loss | SequenceFocalLoss ($\gamma=2.0$) + InfoNCE ($\beta=0.1$), per-token normalized |
| Length conditioning | Active — VQA v2.0 gets `SHORT`, explanations get `LONG` |
| Learning rate | $1 \times 10^{-3}$ → cosine decay |
| Warmup | 2 epochs linear |
| Scheduled sampling | OFF (pure teacher forcing) |
| Curriculum order | Epochs 1–5: easy questions (yes/no, color). Epochs 6–15: all types |
| Batch size | 192 |

**Data loading**: Use `build_mixed_sampler` with three-way weighting:

$$w_{\text{vqa2}} = \frac{0.40}{N_{\text{vqa2}}}, \quad w_{\text{vqae}} = \frac{0.30}{N_{\text{vqae}}}, \quad w_{\text{aokvqa}} = \frac{0.30}{N_{\text{aokvqa}}}$$

**Rationale for 40/30/30 split**: VQA v2.0 provides vocabulary breadth (3,129
unique answers) and alignment training. But at 40% (not 70% as in Model F),
the short-answer bias is significantly reduced while still providing enough
alignment signal.

## Phase 2: Explanation Mastery (10 epochs)

**Objective**: Master long-form explanation generation.

| Parameter | Value |
|:----------|:------|
| Data mix | 100% explanation data: VQA-E filtered + VQA-X + A-OKVQA + synthetic |
| Loss | SequenceFocalLoss + InfoNCE + Coverage ($\lambda_{\text{cov}}=0.5$) |
| Length conditioning | All samples get `LONG` |
| Learning rate | $5 \times 10^{-4}$ |
| Warmup | 0 (warm start from Phase 1) |
| VQA v2.0 replay | **20% experience replay** — randomly sample 20K VQA v2.0 into each epoch to prevent catastrophic forgetting of basic QA ability |
| Batch size | 192 |

**Experience replay implementation**: Maintain a fixed buffer of 20K randomly
selected VQA v2.0 samples. In each epoch, these are mixed into the explanation
dataloader at a 1:4 ratio (replay:explanation). The replay samples still receive
`SHORT` length conditioning.

## Phase 3: Scheduled Sampling (7 epochs)

**Objective**: Bridge the train-test gap by progressively feeding model's own
predictions instead of ground truth.

| Parameter | Value |
|:----------|:------|
| Data mix | Same as Phase 2 |
| Sampling probability | $\epsilon_t = k/(k + \exp(t/k))$, $k=5$ |
| Loss | Same as Phase 2 |
| Learning rate | $2 \times 10^{-4}$ |
| Early stopping | Patience 3 on validation loss |
| Batch size | 192 |

At epoch $t$ within Phase 3, the decoder uses the ground-truth token with
probability $\epsilon_t$ and its own previous prediction with probability
$1 - \epsilon_t$. This decays from ~83% teacher forcing to ~50% over 7 epochs.

## Phase 4: SCST Reinforcement Learning (3 epochs)

**Objective**: Directly optimize human-correlated evaluation metrics.

| Parameter | Value |
|:----------|:------|
| Data mix | VQA-E filtered + VQA-X only (need reference explanations for CIDEr) |
| Loss | $(1 - \lambda)\mathcal{L}_{\text{CE}} + \lambda\mathcal{L}_{\text{SCST}}$, $\lambda=0.5$ |
| Reward | CIDEr-D + 0.5 × BLEU-4 + 0.5 × METEOR − 0.3 × OHP |
| Min-length beam | $t_{\min} = 8$ tokens during greedy baseline |
| Length conditioning | `LONG` only |
| Learning rate | $5 \times 10^{-5}$ |
| No trigram blocking in sampling | Critical — blocking distorts REINFORCE probability space |
| Batch size | 64 (higher memory for sample + greedy sequences) |

**Important**: A-OKVQA is excluded from Phase 4 because CIDEr-D requires
multiple reference sentences per sample. VQA-E has 1 reference, VQA-X has 1.
A-OKVQA has 3 rationales but they explain the *reasoning* not the answer format,
causing CIDEr reward misalignment.

## Curriculum Summary

```
Phase 1 (15 ep)     Phase 2 (10 ep)     Phase 3 (7 ep)      Phase 4 (3 ep)
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ VQA v2.0 40% │    │ VQA-E   40%  │    │ VQA-E   40%  │    │ VQA-E   60%  │
│ VQA-E    30% │ → │ VQA-X   20%  │ → │ VQA-X   20%  │ → │ VQA-X   40%  │
│ A-OKVQA  30% │    │ A-OKVQA 20%  │    │ A-OKVQA 20%  │    │              │
│              │    │ Synthetic 20%│    │ Synthetic 20%│    │              │
│ SHORT+LONG   │    │ LONG + 20%   │    │ + Sched.     │    │ + SCST RL    │
│              │    │   VQA replay │    │   Sampling   │    │ + OHP reward │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
   Alignment           Mastery           Self-Correction       Optimization
```


\newpage

# Evaluation Protocol for Generative VQA

## Why Standard VQA Accuracy Is Insufficient

The standard VQA v2.0 accuracy metric (`min(#humans_who_agree / 3, 1)`) is
designed for 1–3 word classification answers. It fails catastrophically for
15–30 word explanations: *"The man is riding a bicycle because he appears to
be commuting"* would score 0% accuracy despite being correct, because no human
annotator wrote that exact string.

Generative VQA requires metrics that evaluate **semantic correctness**,
**textual quality**, and **visual grounding** independently.

## Recommended Metric Stack

### Primary Metrics (Always Report)

| Metric | What It Measures | Why It Matters |
|:-------|:----------------|:--------------|
| **CIDEr-D** | TF-IDF consensus with references | Primary SCST reward; rewards image-specific vocabulary |
| **METEOR** | Synonym-aware unigram alignment | Best traditional metric for paraphrase handling |
| **SPICE** | Scene graph tuple F-score | Captures object-attribute-relation semantics; 0.88 human correlation |
| **ROUGE-L** | Longest common subsequence F1 | Structural coherence of long sequences |

### Secondary Metrics (Recommended)

| Metric | What It Measures | Why It Matters |
|:-------|:----------------|:--------------|
| **BERTScore F1** | Contextual embedding similarity | Captures semantic similarity beyond n-grams |
| **BLEU-4** | 4-gram precision | Standard NLG baseline; enables comparison with prior work |
| **Avg. Length** | Mean output token count | Verifies model produces target 15–30 word range |
| **OOV Rate** | % of `<unk>` in outputs | Validates Pointer-Generator copy effectiveness |
| **Copy Ratio** | $\mathbb{E}[p_{\text{copy\_Q}} + p_{\text{copy\_V}}]$ | PGN analysis metric |

### Advanced Metrics (For Paper Submission)

| Metric | What It Measures | Implementation |
|:-------|:----------------|:--------------|
| **LAVE** | LLM-judged answer quality | Run Flan-T5-XXL locally (free) |
| **CHAIR$_i$** | Object hallucination rate | Cross-reference with COCO object annotations |
| **Length Distribution** | Histogram of output lengths | Verify bimodal collapse is resolved |

## SPICE Implementation

SPICE (Semantic Propositional Image Caption Evaluation) is notably absent from
the current `evaluate.py`. It parses both generated and reference text into
scene graphs using Stanford CoreNLP, then computes F-score over semantic tuples
(object, attribute, relation).

**Installation**:
```bash
pip install pycocoevalcap  # includes SPICE with Stanford CoreNLP
java -version  # requires Java 8+
```

SPICE is the single most informative metric for explanatory text because it
directly measures whether the generated explanation captures the correct objects,
attributes, and relationships — exactly what "generative VQA" requires.

## Evaluation Protocol

1. **Validation during training**: CIDEr-D + METEOR (fast, differentiable proxy)
2. **Full evaluation**: All primary + secondary metrics on val set ($n = 88,488$)
3. **Ablation comparison**: Primary metrics only (CIDEr-D, METEOR, SPICE, ROUGE-L)
4. **Paper reporting**: All primary + secondary + length analysis + LAVE on 500-sample subset


\newpage

# Implementation Roadmap

## Task Breakdown

| Task | Files | Est. Hours | Priority |
|:-----|:------|:-----------|:---------|
| **Download VQA-X** | New: `scripts/download_vqa_x.sh` | 0.5h | P0 |
| **Download A-OKVQA** | New: `scripts/download_aokvqa.sh` | 0.5h | P0 |
| **VQA-X preprocessor** | New: `scripts/preprocess_vqa_x.py` | 2h | P0 |
| **A-OKVQA preprocessor** | New: `scripts/preprocess_aokvqa.py` | 3h | P0 |
| **COCO 2017→2014 image mapper** | New: `scripts/coco_id_mapper.py` | 1.5h | P1 |
| **Extend filter_hallucinations.py** | Edit: `scripts/filter_hallucinations.py` | 3h | P0 |
| **WordNet synonym matching** | Edit: `scripts/filter_hallucinations.py` | 2h | P1 |
| **Expand synthetic templates** | Edit: `scripts/generate_synthetic_qa.py` | 4h | P1 |
| **Length-bin embedding** | Edit: `decoder_attention.py` | 2h | P0 |
| **Per-token loss normalization** | Edit: `train.py` | 0.5h | P0 |
| **Min-length beam constraint** | Edit: `inference.py` | 0.5h | P0 |
| **Multi-source DataLoader** | Edit: `dataset.py`, `train.py` | 3h | P0 |
| **Experience replay buffer** | Edit: `dataset.py` | 2h | P1 |
| **Unified vocab rebuild** | Edit: `scripts/1_build_vocab.py` | 1.5h | P0 |
| **Add SPICE metric** | Edit: `evaluate.py` | 2h | P1 |
| **Add LAVE metric (Flan-T5)** | New: `evaluate_lave.py` | 3h | P2 |
| **Integration testing** | New: `tests/test_data_pipeline.py` | 3h | P1 |
| **Training script (Model G)** | New: `train_model_g.sh` | 2h | P0 |
| **Total** | | **~35h** | |

## Recommended Execution Order

**Week 1 (Critical Path — P0 tasks)**:
1. Download VQA-X + A-OKVQA (1h)
2. Write preprocessors, normalize to unified format (5h)
3. Extend filter, run on all data (5h)
4. Rebuild vocabulary from merged pool (1.5h)
5. Implement length-conditioned decoding + per-token loss norm (3h)
6. Update DataLoader for multi-source + length bins (3h)

**Week 2 (Enhancement — P1 tasks)**:
7. Expand synthetic templates (4h)
8. WordNet synonym matching in filter (2h)
9. Experience replay buffer (2h)
10. SPICE metric integration (2h)
11. COCO 2017→2014 ID mapping (1.5h)
12. Integration tests (3h)

**Week 3 (Training)**:
13. Phase 1: 15 epochs (~12h GPU)
14. Phase 2: 10 epochs (~8h GPU)
15. Phase 3: 7 epochs (~6h GPU)
16. Phase 4: 3 epochs (~4h GPU)

**Week 4 (Evaluation + Ablation)**:
17. Full evaluation on val set
18. Ablation: with/without VQA-X, with/without A-OKVQA, with/without length conditioning

## Critical Dependencies

```
Download VQA-X + A-OKVQA ──→ Preprocess ──→ Filter ──→ Rebuild Vocab
                                                              │
                                                              ▼
Length Embedding ──→ Update DataLoader ──→ train_model_g.sh ──→ Train
                                              │
Per-token Loss Norm ──────────────────────────┘
                                              │
Min-length Beam ──────────────────────────────┘ (inference only)
```

## Vocabulary Rebuild Specification

The merged vocabulary must include tokens from all sources:

```python
# In scripts/1_build_vocab.py, extend to:
sources = [
    vqa_e_annotations,       # existing
    vqa_v2_questions,        # existing
    vqa_v2_answers,          # existing
    vqa_x_explanations,      # NEW
    aokvqa_rationales,       # NEW
    synthetic_explanations,  # NEW
]
```

Expected vocabulary size increase: ~15–25% for answer vocab (from ~8,648 to
~10,000–11,000 tokens). Question vocab change is minimal.

**Threshold adjustment**: With more data, increase `threshold` from 3 to 5
occurrences to keep vocabulary manageable and reduce long-tail noise.


\newpage

# Risk Analysis and Mitigation

## Risk 1: Domain Gap (A-OKVQA COCO 2017 vs COCO 2014)

**Severity**: Low. ~95% of COCO 2017 val images are identical to COCO 2014.
For the remaining ~5%, the BUTD features must be extracted separately.

**Mitigation**: Run `extract_butd_features.py` on the additional COCO 2017
images. Alternatively, discard the ~5% non-overlapping samples (losing ~2.5K
out of 51K — negligible).

## Risk 2: Format Inconsistency Across Sources

**Severity**: Medium. VQA-X uses "I can tell... because..." while VQA-E uses
"{answer} because {explanation}". A-OKVQA rationales may not contain the answer.

**Mitigation**: Strict format normalization (Section 4). Post-normalization
validation: verify every sample matches regex
`^[a-zA-Z].+ because [a-zA-Z].+$` after preprocessing.

## Risk 3: Catastrophic Forgetting Between Phases

**Severity**: Medium. Phase 1→2 transition (dropping VQA v2.0 from 40% to 0%)
can cause sudden loss of alignment capability.

**Mitigation**: Experience replay (20% VQA v2.0 buffer in Phases 2–3). Monitor
VQA v2.0 validation accuracy alongside explanation metrics. If VQA v2.0
accuracy drops >10% between phases, increase replay fraction to 30%.

## Risk 4: Length Embedding Doesn't Prevent Collapse

**Severity**: Low (unlikely given published results, but worth monitoring).

**Mitigation**: Monitor mean output length per epoch during training. If mean
length drops below 10 tokens after Phase 1, aggressively increase the `LONG`
data fraction and decrease VQA v2.0 fraction. Fallback: add an explicit length
reward $r_{\text{len}} = \min(|y|/15, 1.0)$ to the SCST reward in Phase 4.

## Risk 5: Filter Is Too Aggressive

**Severity**: Low-Medium. The 45% pass rate on VQA-E means discarding 115K samples.

**Mitigation**: Run ablation with relaxed filter (60% pass rate, threshold
adjustments) vs strict filter. If relaxed filter yields better metrics, adopt
the more lenient settings. The filter stages are independently tunable —
Stage 3 (visual grounding) threshold can be lowered from 0.3 to 0.2.


\newpage

# Conclusion

The data bottleneck in generative VQA is not a shortage of *samples* but a
shortage of *high-quality explanatory text* paired with *visual grounding*.
The current 210K VQA-E dataset, with its 66.5% human acceptance rate, provides
a noisy foundation that limits the ceiling of any architectural improvement.

This strategy addresses the bottleneck through four complementary actions:

1. **Addition of 73K human-written explanations** (VQA-X + A-OKVQA) that provide
   the quality backbone the model needs to learn genuine explanatory reasoning
   rather than caption rephrasing. This single action is estimated to yield the
   largest performance improvement of any intervention in the project.

2. **Aggressive quality filtering** reducing VQA-E from 210K to ~95K samples
   using a five-stage pipeline that eliminates hallucinations, question copies,
   length outliers, answer inconsistencies, and duplicates.

3. **Length-conditioned decoding** with learned bin embeddings, per-token loss
   normalization, and minimum-length beam constraints — three orthogonal
   mechanisms that collectively prevent short-answer collapse when training on
   heterogeneous data sources.

4. **A four-phase curriculum** that progressively transitions from alignment
   warm-up (mixed short+long) to explanation mastery (long only) to
   self-correction (scheduled sampling) to metric optimization (SCST RL),
   with experience replay preventing catastrophic forgetting between phases.

The total cost is **zero dollars** and approximately **35 hours of engineering
effort** plus **30 hours of GPU training time**. The expected impact on primary
metrics (CIDEr-D, METEOR, SPICE, ROUGE-L) is estimated at **+5–15%** over the
current VQA-E-only training baseline — a larger gain than any architectural
enhancement in Model G.


\newpage

# References

1. Anderson, P., et al. (2016). SPICE: Semantic Propositional Image Caption Evaluation. *ECCV 2016*.
2. Anderson, P., et al. (2018). Bottom-Up and Top-Down Attention for Image Captioning and VQA. *CVPR 2018*.
3. Hsieh, C.-Y., et al. (2023). Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes. *ACL 2023 Findings*.
4. Kayser, M., et al. (2021). e-ViL: A Dataset and Benchmark for Natural Language Explanations in Vision-Language Tasks. *ICCV 2021*.
5. Kikuchi, Y., et al. (2016). Controlling Output Length in Neural Encoder-Decoders. *EMNLP 2016*.
6. Li, Q., et al. (2018). VQA-E: Explaining, Elaborating, and Enhancing Your Answers for Visual Questions. *ECCV 2018*.
7. Li, Z. & Hoiem, D. (2018). Learning without Forgetting. *IEEE TPAMI*.
8. Mañas, O., et al. (2024). LAVE: LLM-Assisted VQA Evaluation. *AAAI 2024*.
9. Park, D. H., et al. (2018). Multimodal Explanations: Justifying Decisions and Pointing to the Evidence. *CVPR 2018*. (VQA-X)
10. Rennie, S. J., et al. (2017). Self-Critical Sequence Training for Image Captioning. *CVPR 2017*.
11. Schwenk, D., et al. (2022). A-OKVQA: A Benchmark for Visual Question Answering using World Knowledge. *ECCV 2022*.
12. Zellers, R., et al. (2019). From Recognition to Cognition: Visual Commonsense Reasoning. *CVPR 2019*. (VCR)
13. Zhang, T., et al. (2020). BERTScore: Evaluating Text Generation with BERT. *ICLR 2020*.
14. Sammani, F., et al. (2022). NLX-GPT: A Model for Natural Language Explanations in Vision and Vision-Language Tasks. *CVPR 2022*.
15. Buettner, F., et al. (2024). Towards Efficient and Robust VQA-NLE Data Generation with Large Vision-Language Models. *arXiv:2409.14785*.
