# VQA-E Strategic Analysis Log

**Date:** 2026-03-19
**Status:** Model E Phase 4 (SCST) complete. Full evaluation done.

---

## Official Metrics (Model E P4 best — 88,488 val samples, beam=3)

| Metric    | Score  |
|-----------|--------|
| BLEU-4    | 0.1244 |
| METEOR    | 0.3735 |
| ROUGE-L   | 0.4393 |
| BERTScore | 0.9092 |
| Exact     | 0.14%  |

### Quick Eval Comparison (2000 samples, beam=3)
| Checkpoint        | BLEU-4 | METEOR | ROUGE-L | BERTScore |
|-------------------|--------|--------|---------|-----------|
| E P2 epoch45      | 0.1288 | 0.3796 | 0.4437  | 0.9090    |
| E P3 epoch51      | 0.1300 | 0.3793 | 0.4447  | 0.9093    |
| **E P4 best**     | **0.1295** | **0.3834** | **0.4458** | **0.9091** |

SCST gave modest improvement. Production checkpoint: `checkpoints/model_e_best.pth`

---

## SOTA Gap Analysis

| System                       | BLEU-4    | Gap vs Ours |
|------------------------------|-----------|-------------|
| **Our current (Model E P4)** | **0.1244**| —           |
| CNN-LSTM specialist (lit.)   | 0.17–0.19 | +0.046–0.066|
| CNN-LSTM+BUTD+PGN (lit.)     | 0.18–0.20 | +0.056–0.076|
| Transformer-based (ViT+BERT) | 0.22–0.25 | +0.096–0.126|

**~37% below CNN-LSTM+BUTD+PGN SOTA.** Achievable within constraints.

---

## Hard Constraints (NEVER VIOLATE)

1. LSTM must remain the core temporal decoder
2. NO Transformers (no ViT, no BERT, no GPT-style decoders)
3. Cross-attention LEGAL (Q = LSTM h_t, K/V = encoder memory)
4. Self-attention FORBIDDEN (Q = K = V = same sequence)
5. ConvNeXt OK (pure CNN despite transformer-era design)

---

## Root Cause Analysis: Why 0.1244?

### What's Working
- ConvNeXt-Base spatial encoder: strong visual features
- MUTAN fusion: improvement over concat/gated
- MHCA dual cross-attention: correct, no self-attention violation
- SCST REINFORCE: negative loss (−0.07→−0.77) is mathematically correct
- Composite BLEU-4+METEOR reward: appropriate for VQA-E

### Critical Bugs Found

**Bug 1: PGN disabled in ALL beam search (HIGH SEVERITY)**
- `true_batched_beam_search_with_attention` calls `model.decoder.decode_step(cur, (h,c), img_feats_k, q_hid_k, coverage)` — no `q_token_ids` passed
- `decode_step` logic: `if self.use_pgn and q_token_ids is not None` → False → PGN never fires
- All eval metrics produced WITHOUT PGN. Full improvement potential unrealized.
- **Fix:** Add `q_tok_k = qs.repeat_interleave(K, dim=0)` and pass `q_token_ids=q_tok_k`

**Bug 2: SCST greedy/sample asymmetry (MEDIUM SEVERITY)**
- `_sampling_decode` passes `q_token_ids=questions` → PGN ACTIVE
- `_greedy_decode` uses `batch_greedy_decode_with_attention` → PGN INACTIVE
- Would artificially inflate advantage during `--pgn` training
- **Fix:** Thread `q_token_ids` through greedy path too

**Bug 3: Model E never trained with --pgn (CONFIRMED)**
- PGN weights randomly initialized in checkpoint, never optimized
- `--pgn` flag never passed in any training phase

### What Was Not Yet Attempted
| Feature       | Code State          | Training State | Expected BLEU-4 Gain |
|---------------|---------------------|----------------|----------------------|
| PGN (Tier 5)  | Fully implemented   | Never trained  | +0.008–0.015         |
| BUTD Model F  | Architecture ready  | Never trained  | +0.020–0.035         |
| ConceptGNN T9 | Coded, not wired    | Never          | +0.010–0.020         |
| Extended SCST | 3 epochs done       | More available | +0.005–0.010         |
| VG Pre-train  | Not implemented     | Never          | +0.020–0.040         |

---

## Roadmap to CNN-LSTM SOTA

### Realistic Projection (cumulative, non-additive)
```
Current:                              0.1244 BLEU-4
After Phase 5 (PGN fix + train):     +0.010–0.015  →  ~0.135–0.140
After Phase 6 (extended SCST):       +0.005–0.010  →  ~0.140–0.148
After Phase 7 (BUTD Model F):        +0.020–0.030  →  ~0.160–0.175
After Phase 8 (ensemble E+F):        +0.005–0.010  →  ~0.165–0.183
After Phase 9 (ConceptGNN):          +0.010–0.020  →  ~0.175–0.200 ← CNN-LSTM SOTA
After Phase 10 (VG pre-train):       +0.020–0.040  →  ~0.190–0.220
```

CNN-LSTM SOTA ceiling (within constraints): **~0.18–0.20 BLEU-4**

---

## Phase 5: PGN Fix + Training (Immediate)

### Code Fixes Required

**Fix A — `src/inference.py`: `true_batched_beam_search_with_attention`**
After expanding encodings to B*K, add:
```python
q_tok_k = qs.repeat_interleave(K, dim=0)  # (B*K, q_len)
```
In decode loop, change:
```python
# BEFORE (bug):
logit, (h, c), _, coverage = model.decoder.decode_step(
    cur, (h, c), img_feats_k, q_hid_k, coverage
)
# AFTER:
logit, (h, c), _, coverage = model.decoder.decode_step(
    cur, (h, c), img_feats_k, q_hid_k, coverage,
    q_token_ids=q_tok_k
)
```

**Fix B — `src/inference.py`: `batch_greedy_decode_with_attention`**
Pass `q_token_ids=questions` to `decode_step` in the greedy decode path.

**Fix C — `src/training/scst.py`: `_greedy_decode`**
Thread `q_token_ids` through so PGN is active in greedy baseline (SCST symmetry).

### Training Plan

```bash
# Phase 3 continuation with PGN from best P3 checkpoint
python src/train.py --model E --epochs 5 --lr 2e-4 --batch_size 256 \
    --resume checkpoints/model_e_epoch51.pth \
    --scheduled_sampling --ss_k 5 --augment --weight_decay 1e-5 \
    --pgn --coverage --layer_norm --focal --curriculum --wandb

# Phase 4 SCST with PGN
python src/train.py --model E --epochs 5 --lr 5e-5 --batch_size 128 \
    --resume checkpoints/model_e_pgn_resume.pth \
    --scst --scst_lambda 0.5 --no_compile \
    --pgn --coverage --focal --curriculum --wandb
```

### Eval (after Phase 5)
```bash
python src/evaluate.py --model_type E \
    --checkpoint checkpoints/model_e_pgn_best.pth \
    --beam_width 3 --batch_size 512
```

**Target: BLEU-4 ≥ 0.130**

---

## Phase 6: Extended SCST + Reward Tuning

```python
# Proposed reward rebalancing for VQA-E (recall-heavy explanations)
bleu_weight=0.3, meteor_weight=0.6, length_bonus_weight=0.02
```
Add `--scst_bleu_w`, `--scst_meteor_w`, `--scst_len_bonus` to `train.py` argparse.

**Target: BLEU-4 ≈ 0.138–0.148**

---

## Phase 7: BUTD Model F (High ROI)

Fastest R-CNN RoI features (object-specific, 10-100 regions vs 49 uniform grid).

```bash
python src/scripts/extract_butd_features.py \
    --splits train2014 val2014 \
    --output_dir data/butd_features \
    --min_boxes 10 --max_boxes 100
```
Then run Model F through all 4 training phases. **Target: BLEU-4 ≈ 0.160–0.175**

---

## Phase 8: Ensemble E+F

Logit averaging at inference — no extra training. Add `--ensemble_checkpoint` to `evaluate.py`.
**Target: BLEU-4 ≈ 0.165–0.183**

---

## Phase 9: ConceptGNN Integration

Wire `src/models/concept_gnn.py` into decoder as a third cross-attention head (concept MHCA).
Entity extraction from question → ConceptNet subgraph → GCN node features → MHCA.
**Target: BLEU-4 ≈ 0.175–0.200**

---

## Phase 10: Visual Genome Pre-training (Optional)

Pre-train ConvNeXt on 108K VG images × 5.4M region descriptions.
**Target: BLEU-4 ≈ 0.190–0.220**

---

## Priority Matrix

| Phase | Action              | ROI  | Dev Effort | Compute | Prerequisite    |
|-------|---------------------|------|------------|---------|-----------------|
| **5** | PGN fix + train     | High | 4h         | ~3h     | Fix 3 bugs      |
| **6** | SCST reward tune    | Med  | 1h         | ~5h     | Phase 5 done    |
| **7** | BUTD Model F        | High | 6h         | ~40h    | detectron2      |
| **8** | Ensemble E+F        | Med  | 2h         | ~0h     | Phase 7 done    |
| **9** | ConceptGNN          | Med  | 20h        | ~10h    | ConceptNet data |
| **10**| VG pre-training     | High | 10h        | ~80h    | VG dataset      |

**Recommended order:** 5 → 7 (parallel prep) → 6 → 8 → 9 → 10

---

## Files to Modify

### Phase 5
- `src/inference.py` — `true_batched_beam_search_with_attention`: add `q_tok_k`, pass to decode_step
- `src/inference.py` — `batch_greedy_decode_with_attention`: pass `q_token_ids`
- `src/training/scst.py` — `_greedy_decode`: thread `q_token_ids`
- `train_model_e.sh` — add `--pgn --coverage` to Phase 3 and Phase 4 blocks

### Phase 6
- `src/training/scst.py` — reward weight passthrough
- `src/train.py` — add `--scst_bleu_w`, `--scst_meteor_w`, `--scst_len_bonus` argparse

### Phase 7
- `src/scripts/extract_butd_features.py` — run as-is (already implemented)
- No code changes for Model F training (VQAModelF + BUTDDataset already complete)

### Phase 8
- `src/evaluate.py` — `--ensemble_checkpoint` flag + logit averaging

### Phase 9
- `src/models/decoder_attention.py` — add c_mhca (concept cross-attention head)
- `src/models/vqa_models.py` — thread `concept_features` through VQAModelE.forward()
- `src/dataset.py` — concept feature loading
- `src/models/concept_gnn.py` — wire into training pipeline
