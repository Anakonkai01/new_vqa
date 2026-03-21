# Model G — Data Phase Technical Report

**Date**: 2026-03-20
**Author**: VQA-E Research Team
**Status**: Complete ✅
**Branch**: `vqa-e`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Data Sources](#2-data-sources)
3. [Preprocessing: Format Discoveries and Decisions](#3-preprocessing-format-discoveries-and-decisions)
   - 3.1 [VQA-E](#31-vqa-e)
   - 3.2 [VQA-X](#32-vqa-x)
   - 3.3 [A-OKVQA](#33-a-okvqa)
4. [Merge Statistics](#4-merge-statistics)
5. [Five-Stage Quality Filter](#5-five-stage-quality-filter)
   - 5.1 [Filter Architecture](#51-filter-architecture)
   - 5.2 [Debugging History: Stage 4 Failures](#52-debugging-history-stage-4-failures)
   - 5.3 [Final Filter Results](#53-final-filter-results)
6. [Vocabulary Rebuild](#6-vocabulary-rebuild)
7. [Verification Results](#7-verification-results)
8. [Sequence Length Analysis](#8-sequence-length-analysis)
9. [Known Issues and Limitations](#9-known-issues-and-limitations)
10. [File Inventory](#10-file-inventory)
11. [Script Reference](#11-script-reference)

---

## 1. Executive Summary

This report documents the complete data engineering pipeline for Model G — a long-form Generative VQA system targeting 15–30 word explanatory answers. The pipeline consolidates three free data sources (VQA-E, VQA-X, A-OKVQA) into a unified, quality-filtered training pool.

**Key outcomes:**

| Metric | Value |
|:-------|------:|
| Total input samples (3 sources) | 227,813 |
| After quality filtering | **184,954** |
| Overall pass rate | 81.2% |
| Human-written explanation samples (VQA-X + A-OKVQA) | **45,876** |
| Answer vocabulary size | **11,271 tokens** |
| Question vocabulary size | **9,937 tokens** |
| Missing images (after verify) | **0** |
| All verification checks passed | **✅ Yes** |

The pipeline is reproducible end-to-end via 6 sequential scripts. Total runtime: ~20 minutes (dominated by spaCy NLP processing of 227K samples).

---

## 2. Data Sources

### Actual Data Layout

> **Note**: The actual directory layout differs from the `Data_Phase_Implementation_Guide.md` specification. All annotation files reside under `data/annotations/` (not `data/raw/` or `data/vqa_e/`).

```
data/
├── annotations/
│   ├── vqa_e/
│   │   ├── VQA-E_train_set.json        ← 181,298 samples
│   │   └── VQA-E_val_set.json          ← 88,488 samples
│   ├── vqa_x/
│   │   ├── train_x.json                ← 29,459 samples
│   │   ├── val_x.json                  ← 1,459 samples
│   │   └── test_x.json
│   ├── aokvqa/
│   │   ├── aokvqa_v1p0_train.json      ← 17,056 samples
│   │   ├── aokvqa_v1p0_val.json        ← 1,145 samples
│   │   └── aokvqa_v1p0_test.json
│   └── vqa_v2/
│       ├── instances_train2014.json    ← COCO annotations (needed for Stage 3)
│       ├── instances_val2014.json
│       ├── v2_OpenEnded_mscoco_train2014_questions.json
│       └── v2_mscoco_train2014_annotations.json
└── images/
    ├── train2014/                      ← ~82K COCO images
    └── val2014/                        ← ~40K COCO images
```

> The guide originally indicated `instances_train2014.json` needed to be downloaded. It was already present at `data/annotations/vqa_v2/`. No additional downloads were required.

### Source Properties

| Source | Size (train) | Explanation Type | Avg Length | Human-Written? | e-ViL Accept Rate |
|:-------|------------:|:----------------|----------:|:--------------|:-----------------:|
| VQA-E | 181,298 | Auto-generated from COCO captions | ~11 words | No | 66.5% |
| VQA-X | 29,459 | Human-written (AMT workers) | ~9 words | **Yes** | **91.4%** |
| A-OKVQA | 17,056 (×3 rationales) | Human-written (AMT workers) | ~11 words | **Yes** | High |

---

## 3. Preprocessing: Format Discoveries and Decisions

The "peek" step (Step 1 in the guide) revealed four non-obvious format discrepancies between the documented spec and actual files. These required targeted preprocessing adjustments.

### 3.1 VQA-E

**Actual format** (181,298 samples):

```json
{
  "img_id": 9,
  "question": "What is the green stuff?",
  "multiple_choice_answer": "broccoli",
  "explanation": ["Closeup of bins of food that include broccoli and bread.", 0.6791],
  "answers": ["broccoli", "broccoli", ...],
  "answer_type": "other",
  "question_type": "what is the"
}
```

**Key discovery**: `explanation` is a `[string, float]` pair — index `[0]` is the text, index `[1]` is a confidence score from the auto-generation process. The spec documented `explanation` as a plain string.

**Preprocessing decision**: Extract `explanation[0]`; discard confidence float. No image-existence check needed (all VQA-E images are COCO 2014).

**Script**: `src/scripts/preprocess_vqae.py`
**Output**: `data/annotations/vqa_e/vqa_e_train_unified.json` (181,298 samples, 100% pass)

---

### 3.2 VQA-X

**Actual format** (29,459 samples):

```json
{
  "img_id": "COCO_train2014_000000262146",
  "sent": "What is the person doing?",
  "label": {"skiing": 1},
  "explanation": ["they are wearing skis going down the hill."],
  "answer_type": "other",
  "question_id": 262146001
}
```

**Key discoveries** (3 discrepancies):

1. `img_id` is a string (`"COCO_train2014_000000262146"`) — must parse to integer `262146` via regex `_(\d+)$`
2. Question is in field `sent`, **not** `question`
3. Answer is in `label` dict `{word: weight}` — take `argmax(weight)`. Some entries have multiple candidates (e.g., `{"no": 0.6, "yes": 1}`)

**Preprocessing decisions**:
- Parse `img_id` with regex; handle both `train2014` and `val2014` prefixes
- `multiple_choice_answer` ← `argmax(label)`
- `explanation` ← wrap `explanation[0]` in list for format consistency

**Script**: `src/scripts/preprocess_vqa_x.py`
**Output**: `data/annotations/vqa_x/vqa_x_train_unified.json` (29,459 samples, 100% pass — all images found in `data/images/`)

---

### 3.3 A-OKVQA

**Actual format** (17,056 samples):

```json
{
  "image_id": 299207,
  "question_id": "22MexNkBPpdZGX6sxbxVBH",
  "question": "What is the man by the bags awaiting?",
  "choices": ["skateboarder", "train", "delivery", "cab"],
  "correct_choice_idx": 3,
  "direct_answers": ["ride", "ride", "bus", "taxi", "travelling", "taxi", "cab", "cab", "his ride"],
  "rationales": [
    "A train would not be on the street, he would not have luggage waiting for a delivery, and the skateboarder is there and not paying attention to him so a cab is the only possible answer.",
    "He has bags as if he is going somewhere, and he is on a road waiting for a vehicle.",
    "He looks to be waiting for a paid ride to pick him up."
  ]
}
```

**Preprocessing decisions**:
- `multiple_choice_answer` ← `choices[correct_choice_idx]` (more reliable than mode of `direct_answers`)
- `explanation` ← all 3 rationales as a list (natural data augmentation: training randomly picks 1 of 3 per epoch)
- 100% of image IDs mapped to COCO 2014 images (A-OKVQA uses COCO 2017, ~95%+ overlap with 2014)

**Script**: `src/scripts/preprocess_aokvqa.py`
**Output**: `data/annotations/aokvqa/aokvqa_train_unified.json` (17,056 samples, 100% pass)

### Unified Format

All three sources are normalized to:

```json
{
  "img_id": 262146,
  "question": "What is the person doing?",
  "multiple_choice_answer": "skiing",
  "explanation": ["they are wearing skis going down the hill."],
  "source": "vqa_x",
  "split": "train"
}
```

---

## 4. Merge Statistics

**Script**: `src/scripts/merge_sources.py`
**Output**: `data/processed/merged_train_raw.json`

| Source | Count | % | Avg Expl Length |
|:-------|------:|--:|----------------:|
| VQA-E | 181,298 | 79.6% | 11.1 words |
| VQA-X | 29,459 | 12.9% | 8.6 words |
| A-OKVQA | 17,056 | 7.5% | 11.1 words |
| **Total** | **227,813** | **100%** | **10.8 words** |

**Length bin distribution (explanation-only, pre-filter)**:

| Bin | Range | Count | % |
|:----|:------|------:|--:|
| short | ≤5 words | 7,665 | 3.4% |
| medium | 6–14 words | 196,403 | 86.2% |
| long | ≥15 words | 23,745 | 10.4% |

**Top answers**: `yes` (43,659), `no` (33,433), `2` (6,806), `white` (4,114), `3` (3,494). Yes/no answers represent ~34% of the pool — relevant for Phase 1 curriculum design.

**Duplicate Q flags**: 16,291 near-duplicate question pairs (Jaccard ≥ 0.9) within same image flagged for Stage 5 deduplication.

---

## 5. Five-Stage Quality Filter

**Script**: `src/scripts/filter_quality.py`
**Runtime**: ~15 minutes on CPU (spaCy NLP on 227K samples using `nlp.pipe()`)

### 5.1 Filter Architecture

Each sample receives a quality score starting at 5. Failures deduct points:

| Stage | Description | Penalty | Tool |
|:------|:-----------|:-------:|:-----|
| S1 — Length gate | 5 ≤ `len(explanation.split())` ≤ 35 (40 for A-OKVQA) | −1 | Python |
| S2 — Copy-of-question | Jaccard(Q tokens, E tokens) < 0.6 | −1 | Python |
| S3 — Visual grounding | ≥30% content nouns match COCO category names | −2 | spaCy + COCO instances |
| S4 — Answer consistency | Answer word(s) appear in explanation | −1 | Python |
| S5 — Deduplication | Explanation Jaccard < 0.85 within same image | −1 | Python |

**Retention threshold**: score ≥ 3 → kept.

**Stage 3 implementation details**:
- Content nouns extracted via spaCy POS tagging (`NOUN`, `PROPN`), excluding a hardcoded generic-noun stoplist (`thing`, `stuff`, `image`, `photo`, etc.)
- COCO object labels from `instances_train2014.json` + `instances_val2014.json` (80 categories, 604,907 annotations, 122,218 images)
- Samples with fewer than 2 content nouns pass automatically (benefit of the doubt)
- Samples with no COCO annotation for their `img_id` pass automatically

**Stage 3 limitation**: COCO's 80-category taxonomy is narrow. Nouns like `"hill"`, `"crowd"`, `"building"`, `"street"` are visually grounded but absent from COCO labels → high false-fail rate (~55% of VQA-E). However, Stage 3 alone does **not** reject a sample (score drops from 5 to 3, still ≥ threshold). Only when Stage 3 co-fails with another stage does a sample get rejected.

---

### 5.2 Debugging History: Stage 4 Failures

> This section documents a significant issue discovered during pipeline development. It is preserved for reproducibility and future debugging.

**Initial Stage 4 implementation**: Check if `answer` appears in the first 5 words of explanation. This was derived directly from the Data Strategy spec.

**Problem discovered**: Running with this implementation produced:
- 181,696 Stage 4 failures out of 227,813 samples **(79.7% false-fail rate)**
- VQA-X pass rate: 51.4% (expected ~92%)
- A-OKVQA pass rate: 45.1% (expected ~88%)

**Root cause analysis** (by categorizing 32,512 VQA-E Stage 4 failures):

| Failure category | Count | % of S4 fails | Should reject? |
|:-----------------|------:|:-------------:|:--------------:|
| Number word mismatch (`ans='2'` ↔ expl: `'Two'`) | 10,180 | 31.3% | **No** |
| Morphological variant (`ans='skiing'` ↔ expl: `'skis'`) | 5,376 | 16.5% | **No** |
| Genuine contradiction (zero token overlap) | 13,116 | 40.3% | Yes |
| Partial overlap (borderline) | 3,840 | 11.8% | Kept as fail (conservative) |

For VQA-X and A-OKVQA, the issue was different: workers wrote explanations that describe the visual evidence *without stating the answer word*, using paraphrases:
- `ans='skiing'` → expl: `"they are wearing skis going down the hill"` (valid!)
- `ans='macintosh'` → expl: `"The computer is a mac"` (valid!)
- `ans='barbeque'` → expl: `"The gathering is a bbq"` (valid!)

These are correct human explanations that the original Stage 4 logic incorrectly rejects.

**Fix applied** (3 targeted changes):

1. **VQA-X and A-OKVQA**: Skip Stage 4 entirely. These are human-curated; Stage 3 already provides grounding check.

2. **VQA-E — number normalization**: Build a bidirectional digit↔word lookup (`"two"` ↔ `"2"`, etc.) and apply before matching.

3. **VQA-E — stem matching**: Check if the first 5 characters of each answer token match any explanation token (catches `skiing/skis`, `grazing/graze`, `herding/herd`).

4. **A-OKVQA Stage 1**: Increase `MAX_EXPL_WORDS` from 35 to 40 (A-OKVQA rationales are longer by design and high quality).

**Result after fix**:

| Source | Pass rate (before fix) | Pass rate (after fix) |
|:-------|:---------------------:|:--------------------:|
| VQA-E | 52.8% | **76.7%** |
| VQA-X | 51.4% | **99.7%** |
| A-OKVQA | 45.1% | **96.8%** |

---

### 5.3 Final Filter Results

| | VQA-E | VQA-X | A-OKVQA | Total |
|:--|------:|------:|--------:|------:|
| **Input** | 181,298 | 29,459 | 17,056 | **227,813** |
| S1 fail (length) | 213 | 1,600 | 1,816 | 3,629 |
| S2 fail (copy-Q) | 231 | 30 | 92 | 353 |
| S3 fail (grounding) | 100,851 | 15,750 | 10,887 | 127,488 |
| S4 fail (answer) | 25,001 | 0 | 0 | 25,001 |
| S5 fail (dedup) | 64,190 | 7 | 0 | 64,197 |
| | | | | |
| GOLD (score 5) | 42,608 | 12,173 | 4,809 | **59,590** |
| GOOD (score 4) | 33,188 | 1,526 | 1,360 | **36,074** |
| OK (score 3) | 63,282 | 15,669 | 10,339 | **89,290** |
| REJECT (score ≤2) | 42,220 | 91 | 548 | **42,859** |
| | | | | |
| **Kept (score ≥3)** | **139,078** | **29,368** | **16,508** | **184,954** |
| **Pass rate** | **76.7%** | **99.7%** | **96.8%** | **81.2%** |

**VQA-E rejection breakdown**: The 42,220 rejected VQA-E samples are primarily those failing both Stage 3 (visual grounding) AND Stage 5 (duplicate captions). The Stage 5 rejection is legitimate: 31,145 of 72,680 VQA-E images have multiple questions with identical caption-derived explanations (the same COCO caption used for 4–6 different questions about the same image).

---

## 6. Vocabulary Rebuild

**Script**: `src/scripts/1_build_vocab_v2.py`

**Sources used**:
- `merged_train_filtered.json`: all questions + explanations + answers
- VQA v2.0 questions + answers (for Phase 1 warm-up compatibility)

**Threshold**: 3 (token must appear ≥3× to enter vocabulary)

**Threshold rationale** (analysis of all thresholds):

| Threshold | Answer vocab | Coverage | Decision |
|----------:|------------:|:--------:|:---------|
| 2 | 14,199 | 99.68% | Too large — includes typos, hapax noise |
| **3** | **11,271** | **99.45%** | **Selected — best balance** |
| 4 | 9,534 | 99.25% | Cuts meaningful domain words (freq 3–4) |
| 5 | 8,441 | 99.08% | Similar to old vocab, too aggressive |
| 10 | 5,865 | 98.42% | Too small for generative task |

Tokens cut at threshold=3 (appearing only 1–2×) cannot yield meaningful embeddings; tokens kept at threshold=3 but cut at threshold=5 (appearing 3–4×) include domain-specific words critical for explanation generation (`tortellini`, `parachutes`, `croquet`, `afghan`, etc.).

**Results**:

| | Old Vocab (backup) | New Vocab | Delta |
|:--|------------------:|----------:|------:|
| Question vocab | 4,558 | **9,937** | +5,379 (+118%) |
| Answer vocab | 8,813 | **11,271** | +2,458 (+28%) |

New answer tokens by source contribution vs. old vocab:
- VQA-X: +1,350 new tokens
- A-OKVQA: +2,132 new tokens
- VQA-E (new filtered samples): +1,916 new tokens

**File format**: Saved as `{"word2idx": {...}, "idx2word": {...}, "idx": N}` — compatible with existing `Vocabulary.load()` API.

**Special token indices** (fixed, verified):

| Token | Index |
|:------|------:|
| `<pad>` | 0 |
| `<start>` | 1 |
| `<end>` | 2 |
| `<unk>` | 3 |

---

## 7. Verification Results

All 7 checks passed on 2026-03-20:

| Check | Criterion | Result | Value |
|:------|:----------|:------:|------:|
| Total samples | 140K–220K | ✅ | 184,954 |
| VQA-X coverage | ≥25K samples | ✅ | 29,368 |
| A-OKVQA coverage | ≥13K samples | ✅ | 16,508 |
| Avg explanation length | ≥8 words | ✅ | 10.5 words |
| 15–30 word samples | ≥8% | ✅ | 8.9% |
| Question vocab | 8.5K–15K | ✅ | 9,937 |
| Answer vocab | 9K–15K | ✅ | 11,271 |
| Missing images (5K sample) | 0 | ✅ | 0 |

---

## 8. Sequence Length Analysis

> **Important distinction**: Two different length measurements are relevant for different purposes.

### Explanation-only length (for filter Stage 1, quality analysis)

| Source | Avg | Min | Max | ≥15 words |
|:-------|----:|----:|----:|----------:|
| VQA-E | 10.8w | 6 | 44 | 8.0% |
| VQA-X | 8.6w | 1 | 35 | 5.9% |
| A-OKVQA | 11.2w | 1 | 58 | 24.1% |
| **Overall** | **10.5w** | | | **8.9%** |

### Full sequence length — `{answer} because {explanation}` (for training curriculum)

This is what the LSTM decoder actually generates. Adding the answer prefix (+2–4 words) shifts the distribution rightward.

| Bin | Range | Count | % |
|:----|:------|------:|--:|
| short | ≤5 words | 701 | 0.4% |
| medium | 6–14 words | 147,274 | 79.6% |
| long | ≥15 words | **36,979** | **20.0%** |

| Source | Avg full seq | ≥15 words |
|:-------|------------:|----------:|
| VQA-E | 13.0w | 20.0% |
| VQA-X | 10.6w | 12.0% |
| A-OKVQA | 13.5w | **34.3%** |
| **Overall** | **12.6w** | **20.0%** |

**Known discrepancy**: The `length_bin` field currently stored in `merged_train_filtered.json` is based on explanation-only length (not full sequence length). This means the `LONG` bin is under-represented (9.1% vs. 20.0% actual). For training, this field should be recomputed using full sequence length before assigning to the length-conditioned decoder (G5). This is a `TODO` for the DataLoader implementation.

---

## 9. Known Issues and Limitations

### Issue 1: `length_bin` Uses Explanation-Only Length (Minor)
**File**: `data/processed/merged_train_filtered.json`
**Field**: `length_bin`
**Problem**: Computed from `len(explanation[0].split())`, not from the full `{answer} because {explanation}` sequence.
**Impact**: `long` bin underrepresented (9.1% instead of 20.0%). Length-conditioned decoder (G5) may receive incorrect bin signals during training.
**Fix**: Recompute `length_bin` in the DataLoader using `len(answer.split()) + 2 + len(explanation.split())` before passing to the model. No need to re-filter.

### Issue 2: Stage 3 Visual Grounding Uses 80-Category COCO Labels Only (By Design)
**Problem**: COCO's 80-category taxonomy misses many valid visual nouns (`hill`, `crowd`, `building`, `street`). Stage 3 false-fail rate ~55% for VQA-E.
**Impact**: Minimal — Stage 3 alone does not reject a sample (score 3 = kept). Only co-failing with another stage causes rejection.
**Mitigation per spec**: WordNet synonym matching (distance ≤ 2) would reduce false-fail rate. This was descoped to keep runtime reasonable.

### Issue 3: A-OKVQA Rationale Diversity Not Preserved in `length_bin`
**Problem**: A-OKVQA stores 3 rationales per question; `length_bin` is computed from `explanation[0]` only. The other 2 rationales may have different lengths.
**Impact**: Negligible at this stage. During training, when a random rationale is picked, the length bin may not match the selected rationale's actual length.

### Issue 4: Stage 4 Stem Matching May Have False Positives
**Problem**: The 5-character stem match (`ans[:5] == w[:5]`) used for VQA-E can produce false positives for short words (e.g., `ans='bike'` matches `'biker'`, `'bikes'` — acceptable; but `ans='ban'` might match `'band'` — borderline).
**Impact**: Very low — Stage 4 false-positives add noise to the GOLD/GOOD score but do not cause incorrect rejection.

---

## 10. File Inventory

### Input Files (unchanged)

| File | Path | Size | Notes |
|:-----|:-----|-----:|:------|
| VQA-E train | `data/annotations/vqa_e/VQA-E_train_set.json` | ~181K | explanation = [str, float] |
| VQA-E val | `data/annotations/vqa_e/VQA-E_val_set.json` | ~88K | |
| VQA-X train | `data/annotations/vqa_x/train_x.json` | 29,459 | img_id is string |
| VQA-X val | `data/annotations/vqa_x/val_x.json` | 1,459 | |
| A-OKVQA train | `data/annotations/aokvqa/aokvqa_v1p0_train.json` | 17,056 | 3 rationales each |
| A-OKVQA val | `data/annotations/aokvqa/aokvqa_v1p0_val.json` | 1,145 | |
| COCO instances train | `data/annotations/vqa_v2/instances_train2014.json` | 80 cats / 605K ann | |
| COCO instances val | `data/annotations/vqa_v2/instances_val2014.json` | | |

### Intermediate Files (generated)

| File | Path | Samples | Notes |
|:-----|:-----|--------:|:------|
| VQA-E unified | `data/annotations/vqa_e/vqa_e_train_unified.json` | 181,298 | source tag added |
| VQA-X unified | `data/annotations/vqa_x/vqa_x_train_unified.json` | 29,459 | img_id parsed to int |
| A-OKVQA unified | `data/annotations/aokvqa/aokvqa_train_unified.json` | 17,056 | 3 rationales in list |
| Merged raw | `data/processed/merged_train_raw.json` | 227,813 | length_bin assigned |
| Merged scored | `data/processed/merged_train_scored.json` | 227,813 | quality_score added |

### Final Output Files

| File | Path | Samples | Notes |
|:-----|:-----|--------:|:------|
| **Filtered data** | `data/processed/merged_train_filtered.json` | **184,954** | score ≥ 3, training-ready |
| Filter report | `data/processed/filter_report.json` | — | per-source stage stats |
| Question vocab | `data/processed/vocab_questions.json` | 9,937 | threshold=3 |
| Answer vocab | `data/processed/vocab_answers.json` | 11,271 | threshold=3 |

### Backup Files

| File | Path | Notes |
|:-----|:-----|:------|
| Old answer vocab | `data/processed/backup/vocab_answers_8813.json` | Pre-pipeline vocab |
| Old question vocab | `data/processed/backup/vocab_questions_4558.json` | Pre-pipeline vocab |

---

## 11. Script Reference

All scripts are in `src/scripts/`. Run from project root. All use `conda run -n d2l python`.

| Step | Script | Runtime | Input → Output |
|:-----|:-------|:-------:|:---------------|
| 1 | *(inline peek)* | <1 min | Raw JSONs → Format confirmed |
| 2 | `preprocess_vqa_x.py` | <1 min | `train_x.json` → `vqa_x_train_unified.json` |
| 3 | `preprocess_aokvqa.py` | <1 min | `aokvqa_v1p0_train.json` → `aokvqa_train_unified.json` |
| 4 | `preprocess_vqae.py` | <1 min | `VQA-E_train_set.json` → `vqa_e_train_unified.json` |
| 5 | `merge_sources.py` | ~1 min | 3 unified JSONs → `merged_train_raw.json` |
| 6 | `filter_quality.py` | ~15 min | `merged_train_raw.json` → `merged_train_filtered.json` |
| 7 | `1_build_vocab_v2.py` | ~2 min | filtered + VQA v2.0 → `vocab_*.json` |
| 8 | *(inline verify)* | ~3 min | All outputs → verification report |

**Run all in sequence**:
```bash
conda run -n d2l python src/scripts/preprocess_vqa_x.py
conda run -n d2l python src/scripts/preprocess_aokvqa.py
conda run -n d2l python src/scripts/preprocess_vqae.py
conda run -n d2l python src/scripts/merge_sources.py
conda run -n d2l python src/scripts/filter_quality.py
conda run -n d2l python src/scripts/1_build_vocab_v2.py
```

**Dependencies**: `spacy`, `en_core_web_sm` (for filter Stage 3 only)
```bash
pip install spacy && python -m spacy download en_core_web_sm
```
