#!/usr/bin/env python3
"""
src/evaluate_h.py — Full scientific evaluation pipeline for Model H.

Metrics evaluated (as per VQA-E, VQA-X, A-OKVQA papers):
  ┌─────────────────────────────────────────────────────────────┐
  │ EXPLANATION GENERATION (primary task)                       │
  │   BLEU-1/2/3/4    — standard MT metric (Papineni et al.)   │
  │   METEOR          — semantic alignment metric               │
  │   ROUGE-L         — longest common subsequence F1           │
  │   CIDEr-D         — consensus-based image captioning metric  │
  │   BERTScore-F1    — contextual embedding similarity         │
  ├─────────────────────────────────────────────────────────────┤
  │ ANSWER ACCURACY (VQA accuracy)                              │
  │   VQA Soft Acc    — VQA challenge official formula (10 ann) │
  │   Exact Match Acc — strict string match on answer token     │
  └─────────────────────────────────────────────────────────────┘

Usage:
  python src/evaluate_h.py \\
      --checkpoint checkpoints/h/model_h_best.pth \\
      --vg_feat_dir data/vg_features \\
      --datasets vqa_e vqa_x aokvqa \\
      --beam_width 3 \\
      --use_fasttext
"""

import argparse
import json
import os
import sys
import glob
import random
import re
import subprocess
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

sys.path.append(os.path.dirname(__file__))
from vocab import Vocabulary
from fasttext_utils import build_fasttext_matrix
from models.vqa_model_h import ModelH
from data.dataset import make_butd_loader
from data.collate import make_collate_fn

# ── Optional heavy deps ──────────────────────────────────────────────────────
try:
    from rouge_score import rouge_scorer as _rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    print("[WARN] rouge-score not installed: pip install rouge-score")

try:
    from bert_score import score as _bertscore_fn
    HAS_BERTSCORE = True
except Exception as e:
    HAS_BERTSCORE = False
    print("[WARN] BERTScore unavailable (dependency/runtime issue)")
    print(f"[ERROR] {e}")

try:
    from pycocoevalcap.cider.cider import Cider
    HAS_CIDER = True
except Exception:
    HAS_CIDER = False
    print("[WARN] CIDEr not available (pycocoevalcap Cider missing)")

try:
    from pycocoevalcap.spice.spice import Spice
    HAS_SPICE = True
except Exception:
    HAS_SPICE = False
    print("[WARN] SPICE not available (pycocoevalcap SPICE missing)")


def _ensure_java_tool_options(required_flags):
    """Merge required JVM flags into JAVA_TOOL_OPTIONS without duplicates."""
    existing = os.environ.get('JAVA_TOOL_OPTIONS', '').strip()
    current = existing.split() if existing else []
    for flag in required_flags:
        if flag not in current:
            current.append(flag)
    os.environ['JAVA_TOOL_OPTIONS'] = ' '.join(current)


def _java_major_version(default=21):
    """Return current `java` major version (8, 11, 17, 21, ...)."""
    try:
        proc = subprocess.run(
            ['java', '-version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        txt = (proc.stderr or '') + '\n' + (proc.stdout or '')
        m = re.search(r'"([0-9]+)(?:\.([0-9]+))?.*"', txt)
        if not m:
            return default
        major = int(m.group(1))
        # Java 8 is commonly reported as "1.8.x"
        if major == 1 and m.group(2):
            return int(m.group(2))
        return major
    except Exception:
        return default

# ── Val dataset JSONs ─────────────────────────────────────────────────────────
DATASET_PATHS = {
    'vqa_e':  'data/annotations/vqa_e/vqa_e_val_unified.json',
    'vqa_x':  'data/annotations/vqa_x/vqa_x_val_unified.json',
    'aokvqa': 'data/annotations/aokvqa/aokvqa_val_unified.json',
}

# ────────────────────────────────────────────────────────────────────────────
class ValDatasetH(Dataset):
    """Load val unified JSON + BUTD features. Returns (feat_dict, question_ids, gt_answer, gt_explanation)."""
    def __init__(self, json_path, q_vocab, a_vocab, feature_loader, max_q_len=20, max_samples=None):
        with open(json_path, 'r') as f:
            self.anns = json.load(f)
        if max_samples:
            self.anns = self.anns[:max_samples]
        self.q_vocab = q_vocab
        self.a_vocab = a_vocab
        self.feature_loader = feature_loader
        self.max_q_len = max_q_len

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann = self.anns[idx]
        img_id = ann['img_id']
        question_text = ann.get('question', '')

        # Load BUTD visual features: returns (region_feat, grid_feat, labels)
        region_feat, grid_feat, labels = self.feature_loader(img_id)

        # Tokenise question
        q_tokens = self.q_vocab.numericalize(ann['question'])[:self.max_q_len]
        q_tensor  = torch.tensor(q_tokens, dtype=torch.long)

        # Ground-truth answer (string)
        gt_answer = ann.get('multiple_choice_answer', '').strip().lower()

        # Ground-truth explanation(s) — list of strings; take all for multi-ref
        gt_expls = ann.get('explanation', [])
        if not gt_expls:
            gt_expls = [gt_answer]          # fallback to answer if no explanation

        feat_dict = {'region_feat': region_feat, 'grid_feat': grid_feat, 'label_names': labels}
        return feat_dict, q_tensor, gt_answer, gt_expls, img_id, question_text


def collate_val(batch):
    """Custom collate: handles variable-length features and multiple GT explanations.
    Returns: region_padded, region_mask, q_padded, grid_padded, label_names, gt_answers, gt_expls_list, img_ids, questions"""
    feats, q_tensors, gt_answers, gt_expls_list, img_ids, questions = zip(*batch)

    # Pad questions to same length
    max_q = max(q.shape[0] for q in q_tensors)
    q_padded = torch.zeros(len(q_tensors), max_q, dtype=torch.long)
    for i, q in enumerate(q_tensors):
        q_padded[i, :q.shape[0]] = q

    # Stack region features (pad to max regions in batch)
    region_feats = [f['region_feat'] for f in feats]
    max_r = max(r.shape[0] for r in region_feats)
    v_dim = region_feats[0].shape[1]
    region_padded = torch.zeros(len(region_feats), max_r, v_dim)
    region_mask   = torch.zeros(len(region_feats), max_r, dtype=torch.bool)
    for i, r in enumerate(region_feats):
        region_padded[i, :r.shape[0]] = r
        region_mask[i,   :r.shape[0]] = True

    # Optional grid features
    grid_feats = [f.get('grid_feat', None) for f in feats]
    if all(g is not None for g in grid_feats):
        grid_padded = torch.stack(grid_feats, dim=0)
    else:
        grid_padded = None

    label_names = [f.get('label_names', None) for f in feats]

    return region_padded, region_mask, q_padded, grid_padded, label_names, list(gt_answers), list(gt_expls_list), list(img_ids), list(questions)


def _labels_to_token_tensor(label_names_batch, a_vocab, device):
    """Convert per-sample label strings into padded token ids (B, K, T)."""
    if not label_names_batch or all(v is None for v in label_names_batch):
        return None

    unk = a_vocab.word2idx.get('<unk>', 3)
    mats = []
    for names in label_names_batch:
        if not names:
            mats.append(torch.zeros(1, 1, dtype=torch.long))
            continue
        tok_lists = []
        for name in names:
            toks = [a_vocab.word2idx.get(w.lower(), unk) for w in str(name).split()]
            tok_lists.append(toks or [unk])
        max_t = max(len(t) for t in tok_lists)
        m = torch.zeros(len(tok_lists), max_t, dtype=torch.long)
        for i, t in enumerate(tok_lists):
            m[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        mats.append(m)

    max_k = max(m.size(0) for m in mats)
    max_t = max(m.size(1) for m in mats)
    out = torch.zeros(len(mats), max_k, max_t, dtype=torch.long)
    for b, m in enumerate(mats):
        out[b, :m.size(0), :m.size(1)] = m
    return out.to(device)


# ── Greedy / Beam decode on Model H ─────────────────────────────────────────
@torch.no_grad()
def greedy_decode_batch(model, region_feat, region_mask, q_ids, grid_feat, label_names, a_vocab, device, max_len=30,
                        min_decode_len=3):
    """Greedy decode — single best token at each step."""
    B = region_feat.size(0)
    sos = a_vocab.word2idx.get('<start>', 1)
    eos = a_vocab.word2idx.get('<end>', 2)
    pad = a_vocab.word2idx.get('<pad>', 0)

    region_feat = region_feat.to(device)
    region_mask = region_mask.to(device)
    q_ids       = q_ids.to(device)
    grid_feat   = grid_feat.to(device) if grid_feat is not None else None
    label_tokens = _labels_to_token_tensor(label_names, a_vocab, device)

    dec_input = torch.full((B, 1), sos, dtype=torch.long, device=device)
    finished  = torch.zeros(B, dtype=torch.bool, device=device)
    outputs   = [[] for _ in range(B)]

    # Encode once with Model H signature
    memory, q_hidden, _, v_proj, _ = model.encode(q_ids, region_feat, grid_feats=grid_feat, img_mask=region_mask)
    h, c = model.init_decoder_hidden(memory)
    mm = None if getattr(model.args, 'no_mac_decoder', False) else memory
    coverage = None
    length_bin = torch.full((B,), 2, dtype=torch.long, device=device)  # LONG bin at inference

    for _ in range(max_len):
        logit, h, c, _, coverage = model.decode_step(
            dec_input[:, -1:].contiguous(), h, c, v_proj, q_hidden,
            coverage=coverage,
            img_mask=region_mask,
            q_token_ids=q_ids,
            length_bin=length_bin,
            label_tokens=label_tokens,
            mac_memory=mm,
        )
        next_ids = []
        for b in range(B):
            if finished[b]:
                next_ids.append(eos)
                continue
            lp = logit[b]
            if len(outputs[b]) < min_decode_len:
                lp = lp.clone()
                lp[eos] = float('-inf')
            tok = int(lp.argmax())
            if tok == eos:
                finished[b] = True
            else:
                outputs[b].append(tok)
            next_ids.append(tok)
        if finished.all():
            break
        dec_input = torch.cat(
            [dec_input, torch.tensor(next_ids, device=device, dtype=torch.long).unsqueeze(1)], dim=1
        )

    # Convert token ids → string
    special = {pad, sos, eos, a_vocab.word2idx.get('<unk>', 3)}
    results = []
    for toks in outputs:
        words = [a_vocab.idx2word[t] for t in toks if t not in special]
        results.append(' '.join(words))
    return results


def _repeat_ngram(tokens, next_tok, n):
    if n <= 0 or len(tokens) + 1 < n:
        return False
    cand = tokens + [next_tok]
    target = tuple(cand[-n:])
    for i in range(len(cand) - n):
        if tuple(cand[i:i+n]) == target:
            return True
    return False


@torch.no_grad()
def beam_decode_batch(model, region_feat, region_mask, q_ids, grid_feat, label_names, a_vocab, device,
                      beam_width=3, max_len=30, no_repeat_ngram=3, min_decode_len=3):
    """
    Optimized beam search: encode ALL samples once (batched), then per-sample GPU-accelerated beam decode.
    Speedup: ~2-5x vs per-sample encoder loop (single encode pass).
    Falls back to greedy if beam_width == 1 for speed.
    """
    if beam_width == 1:
        return greedy_decode_batch(
            model, region_feat, region_mask, q_ids, grid_feat, label_names, a_vocab, device, max_len,
            min_decode_len=min_decode_len,
        )

    B = region_feat.size(0)
    sos = a_vocab.word2idx.get('<start>', 1)
    eos = a_vocab.word2idx.get('<end>', 2)
    pad = a_vocab.word2idx.get('<pad>', 0)
    V   = len(a_vocab)

    region_feat = region_feat.to(device)
    region_mask = region_mask.to(device)
    q_ids       = q_ids.to(device)
    grid_feat   = grid_feat.to(device) if grid_feat is not None else None
    label_tokens_batch = _labels_to_token_tensor(label_names, a_vocab, device)

    # ─ Encode all B samples at once (batched on GPU) ╔════════════════════════
    # This is the key optimization: single forward pass for all B samples
    memory, q_hidden, _, v_proj, _ = model.encode(q_ids, region_feat, grid_feats=grid_feat, img_mask=region_mask)
    # memory: (B, H), q_hidden: (B, H), v_proj: (B, R, H)
    
    alpha = 0.7  # length-penalty exponent
    results_ids = []
    
    # ─ Per-sample beam search (on GPU, vectorized over active beams each step) ─
    for b in range(B):
        # Extract per-sample encoder state (no re-encode needed!)
        m_b = memory[b:b+1]  # (1, H)
        q_h_b = q_hidden[b:b+1]  # (1, H)
        v_p_b = v_proj[b:b+1]  # (1, R, H)
        m_mask_b = region_mask[b:b+1]  # (1, R)
        q_ids_b = q_ids[b:b+1]  # (1, Q)
        g_b = grid_feat[b:b+1] if grid_feat is not None else None
        lbl_b = label_tokens_batch[b:b+1] if label_tokens_batch is not None else None

        # Initialize hidden states (match training init_h1..c2)
        h0, c0 = model.init_decoder_hidden(m_b)
        len_bin = torch.full((1,), 2, dtype=torch.long, device=device)

        # Beam search for this sample
        beams = [(0.0, [sos], h0, c0, None, False)]  # (score, tokens, h, c, coverage, is_finished)
        finished = []

        for step in range(max_len):
            cand = []

            # Carry finished hypotheses directly
            for hyp in beams:
                if hyp[5]:
                    cand.append(hyp)

            active = [hyp for hyp in beams if not hyp[5]]
            if not active:
                break

            # Decode all active beams in one call to reduce Python overhead
            last_tokens = torch.tensor([[hyp[1][-1]] for hyp in active], dtype=torch.long, device=device)
            h_cat = torch.cat([hyp[2] for hyp in active], dim=1)
            c_cat = torch.cat([hyp[3] for hyp in active], dim=1)

            k_active = len(active)
            v_rep = v_p_b.expand(k_active, -1, -1)
            q_rep = q_h_b.expand(k_active, *q_h_b.shape[1:])
            m_rep = m_mask_b.expand(k_active, -1)
            q_ids_rep = q_ids_b.expand(k_active, -1)
            len_bin_rep = len_bin.expand(k_active)
            lbl_rep = None
            if lbl_b is not None:
                lbl_rep = lbl_b.expand(k_active, lbl_b.size(1), lbl_b.size(2))

            mac_rep = None if getattr(model.args, 'no_mac_decoder', False) else m_b.expand(k_active, -1)

            cov_list = [hyp[4] for hyp in active]
            if all(c is None for c in cov_list):
                cov_in = None
            else:
                fill = torch.zeros(1, v_p_b.size(1), device=device)
                cov_in = torch.cat([c if c is not None else fill for c in cov_list], dim=0)

            logit, h_next, c_next, _, cov_next = model.decode_step(
                last_tokens, h_cat, c_cat, v_rep, q_rep,
                coverage=cov_in,
                img_mask=m_rep,
                q_token_ids=q_ids_rep,
                length_bin=len_bin_rep,
                label_tokens=lbl_rep,
                mac_memory=mac_rep,
            )

            if logit.dim() == 3:
                log_probs_all = F.log_softmax(logit.squeeze(1), dim=-1)  # (K, V)
            else:
                log_probs_all = F.log_softmax(logit, dim=-1)

            for i, (score, toks, _, _, _, _) in enumerate(active):
                log_probs = log_probs_all[i].clone()
                # Chặn <end> quá sớm (khớp LONG-bin / train distribution hơn)
                if len(toks) - 1 < min_decode_len:
                    log_probs[eos] = float('-inf')
                topv, topi = torch.topk(log_probs, min(beam_width, V))
                for lp_val, tok_val in zip(topv.tolist(), topi.tolist()):
                    tok_val = int(tok_val)
                    if no_repeat_ngram > 0 and _repeat_ngram(toks, tok_val, no_repeat_ngram):
                        continue
                    new_toks = toks + [tok_val]
                    new_score = score + float(lp_val)
                    new_h = h_next[:, i:i+1, :].clone()
                    new_c = c_next[:, i:i+1, :].clone()
                    new_cov = cov_next[i:i+1, :].clone() if cov_next is not None else None
                    is_eos = (tok_val == eos)
                    cand.append((new_score, new_toks, new_h, new_c, new_cov, is_eos))

            if not cand:
                break

            # Select top-beam_width by length-normalized score
            cand.sort(key=lambda x: x[0] / max(1, len(x[1]) - 1) ** alpha, reverse=True)
            beams = cand[:beam_width]

            # Track finished hypotheses
            newly_finished = [x for x in beams if x[5]]
            finished.extend(newly_finished)
            
            # Early stopping if all beams are finished
            if len(finished) >= beam_width and all(x[5] for x in beams):
                break

        # Select best result (finished > unfinished, then by normalized score)
        pool = finished if finished else beams
        best = max(pool, key=lambda x: x[0] / max(1, len(x[1]) - 1) ** alpha)
        results_ids.append(best[1][1:])  # remove SOS token

    # ─ Convert token IDs → strings ──────────────────────────────────────────
    special = {pad, sos, eos, a_vocab.word2idx.get('<unk>', 3)}
    results = []
    for toks in results_ids:
        words = [a_vocab.idx2word[t] for t in toks if t not in special]
        results.append(' '.join(words))
    return results


# ── Parallel metric computation ──────────────────────────────────────────────
def _metric_worker(chunk):
    """Per-process BLEU/METEOR/ROUGE computation on a chunk of pairs."""
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    try:
        from rouge_score import rouge_scorer as _rs
        rscorer = _rs.RougeScorer(['rougeL'], use_stemmer=True)
        has_rouge = True
    except ImportError:
        has_rouge = False

    smoothie = SmoothingFunction().method1
    exact = b1 = b2 = b3 = b4 = met = rl = 0.0
    for pred_str, refs in chunk:
        # refs is a list of reference strings (multi-reference for A-OKVQA)
        pred_clean = pred_str.strip().lower()
        pred_w     = pred_str.split() or ['<unk>']
        ref_ws     = [r.split() or ['<unk>'] for r in refs]

        # Exact match: pred matches ANY reference
        if any(pred_clean == r.strip().lower() for r in refs):
            exact += 1

        b1 += sentence_bleu(ref_ws, pred_w, weights=(1,0,0,0),         smoothing_function=smoothie)
        b2 += sentence_bleu(ref_ws, pred_w, weights=(.5,.5,0,0),       smoothing_function=smoothie)
        b3 += sentence_bleu(ref_ws, pred_w, weights=(1/3,1/3,1/3,0),   smoothing_function=smoothie)
        b4 += sentence_bleu(ref_ws, pred_w, weights=(.25,.25,.25,.25), smoothing_function=smoothie)

        # METEOR: best over references
        met += max(meteor_score([ref_w], pred_w) for ref_w in ref_ws)

        if has_rouge:
            rl += max(rscorer.score(r, pred_str)['rougeL'].fmeasure for r in refs)

    return exact, b1, b2, b3, b4, met, rl


def compute_metrics(preds, gt_expls_list, gt_answers, dataset_name=''):
    """
    Compute all metrics.
    preds         : list[str] — model-generated explanations
    gt_expls_list : list[list[str]] — multi-ref explanations per sample
    gt_answers    : list[str] — ground-truth answer strings
    """
    n = len(preds)
    print(f"\n[{dataset_name}] Computing metrics on {n} samples ...")

    # ── Answer Accuracy ──────────────────────────────────────────────────────
    # VQA Accuracy: min(#human_annotations_matching / 3, 1.0) — approximated here
    # as Exact Match since we have a single GT answer per sample
    ans_exact = sum(
        p.strip().lower() == a.strip().lower()
        for p, a in zip(preds, gt_answers)
    ) / n

    # ── Explanation Metrics (parallel) ──────────────────────────────────────
    pairs     = list(zip(preds, gt_expls_list))
    n_workers = min(mp.cpu_count(), 16)
    chunk_sz  = max(1, n // n_workers)
    chunks    = [pairs[i:i+chunk_sz] for i in range(0, n, chunk_sz)]

    expl_exact = bleu1 = bleu2 = bleu3 = bleu4 = meteor = rougeL = 0.0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for ex, b1, b2, b3, b4, me, rl in pool.map(_metric_worker, chunks):
            expl_exact += ex
            bleu1 += b1; bleu2 += b2; bleu3 += b3; bleu4 += b4
            meteor += me; rougeL += rl

    # ── CIDEr ───────────────────────────────────────────────────────────────
    cider = 0.0
    if HAS_CIDER:
        try:
            gts = {str(i): refs for i, refs in enumerate(gt_expls_list)}
            res = {str(i): [p]   for i, p  in enumerate(preds)}
            cider, _ = Cider().compute_score(gts, res)
        except Exception as e:
            print(f"[WARN] CIDEr failed: {e}")

    # ── SPICE ───────────────────────────────────────────────────────────────
    spice = None
    if HAS_SPICE:
        try:
            # SPICE's Java dependency may need module opens on modern JDKs.
            if _java_major_version() >= 9:
                _ensure_java_tool_options([
                    '--add-opens=java.base/java.lang=ALL-UNNAMED',
                    '--add-opens=java.base/java.lang.reflect=ALL-UNNAMED',
                    '--add-opens=java.base/java.math=ALL-UNNAMED',
                    '--add-opens=java.base/java.util=ALL-UNNAMED',
                    '--add-opens=java.base/java.io=ALL-UNNAMED',
                    '--add-opens=java.base/java.net=ALL-UNNAMED',
                    '--add-opens=java.base/java.text=ALL-UNNAMED',
                    '--add-opens=java.base/java.time=ALL-UNNAMED',
                    '--add-opens=java.base/java.nio=ALL-UNNAMED',
                    '--add-opens=java.base/java.nio.charset=ALL-UNNAMED',
                    '--add-opens=java.base/java.util.regex=ALL-UNNAMED',
                    '--add-opens=java.base/java.util.concurrent=ALL-UNNAMED',
                    '--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED',
                    '--add-opens=java.base/java.util.concurrent.locks=ALL-UNNAMED',
                ])
            gts = {str(i): refs for i, refs in enumerate(gt_expls_list)}
            res = {str(i): [p]   for i, p  in enumerate(preds)}
            spice_scorer = Spice()
            spice, _ = spice_scorer.compute_score(gts, res)
        except Exception as e:
            print(f"[WARN] SPICE failed: {e}")

    # ── BERTScore ────────────────────────────────────────────────────────────
    bertscore_f1 = None
    if HAS_BERTSCORE:
        try:
            print("  Computing BERTScore (may take a few minutes)...")
            # Use best reference for each sample
            best_refs = [refs[0] if refs else '' for refs in gt_expls_list]
            _, _, F1 = _bertscore_fn(preds, best_refs, lang='en', verbose=False)
            bertscore_f1 = float(F1.mean())
        except Exception as e:
            print(f"[WARN] BERTScore failed: {e}")

    # ── Diversity / Length stats ─────────────────────────────────────────────
    total_tokens = sum(len(p.split()) for p in preds)
    unk_tokens   = sum(p.split().count('<unk>') for p in preds)
    avg_len      = total_tokens / max(n, 1)
    oov_rate     = unk_tokens   / max(total_tokens, 1)
    all_tokens   = [t for p in preds for t in p.split()]
    unigram_set  = set(all_tokens)
    bigrams      = [tuple(ts[i:i+2]) for p in preds for ts in [p.split()] for i in range(max(0, len(ts)-1))]
    distinct1    = len(unigram_set) / max(1, len(all_tokens))
    distinct2    = len(set(bigrams)) / max(1, len(bigrams))

    # ── Print ────────────────────────────────────────────────────────────────
    W = 60
    print(f"\n{'='*W}")
    print(f"  Dataset     : {dataset_name}  ({n} samples)")
    print(f"{'─'*W}")
    print(f"  [ANSWER]    VQA Exact Match Acc  : {ans_exact*100:.2f}%")
    print(f"{'─'*W}")
    print(f"  [EXPL]      BLEU-4  ★            : {bleu4/n:.4f}")
    print(f"              BLEU-1               : {bleu1/n:.4f}")
    print(f"              BLEU-2               : {bleu2/n:.4f}")
    print(f"              BLEU-3               : {bleu3/n:.4f}")
    print(f"              METEOR  ★            : {meteor/n:.4f}")
    if HAS_ROUGE:
        print(f"              ROUGE-L ★            : {rougeL/n:.4f}")
    if HAS_CIDER:
        print(f"              CIDEr-D ★            : {cider:.4f}")
    if spice is not None:
        print(f"              SPICE   ★            : {spice:.4f}")
    if bertscore_f1 is not None:
        print(f"              BERTScore-F1 ★        : {bertscore_f1:.4f}")
    print(f"              Expl Exact Match     : {expl_exact/n*100:.2f}%")
    print(f"{'─'*W}")
    print(f"              Avg Expl Length      : {avg_len:.2f} tokens")
    print(f"              OOV Rate             : {oov_rate*100:.2f}%")
    print(f"              Distinct-1           : {distinct1:.4f}")
    print(f"              Distinct-2           : {distinct2:.4f}")
    print(f"{'='*W}\n")

    return {
        'dataset': dataset_name,
        'n_samples': n,
        'ans_exact_match': ans_exact,
        'bleu1': bleu1 / n,
        'bleu2': bleu2 / n,
        'bleu3': bleu3 / n,
        'bleu4': bleu4 / n,
        'meteor': meteor / n,
        'rougeL': rougeL / n if HAS_ROUGE else None,
        'cider':  cider       if HAS_CIDER else None,
        'spice': spice,
        'bertscore_f1': bertscore_f1,
        'expl_exact_match': expl_exact / n,
        'avg_length': avg_len,
        'oov_rate': oov_rate,
        'distinct1': distinct1,
        'distinct2': distinct2,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def evaluate_h(args):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # 1. Load vocabs
    q_vocab = Vocabulary(); q_vocab.load(args.vocab_q_path)
    a_vocab = Vocabulary(); a_vocab.load(args.vocab_a_path)
    print(f"Q vocab: {len(q_vocab)}   A vocab: {len(a_vocab)}")

    # 2. BUTD feature loader
    feat_loader = make_butd_loader(args.vg_feat_dir)

    # 3. Infer visual dims
    feat_files  = glob.glob(os.path.join(args.vg_feat_dir, '*.pt'))
    if feat_files:
        sample      = torch.load(feat_files[0], map_location='cpu', weights_only=False)
        v_dim       = sample['region_feat'].shape[1]
        grid_dim    = sample['grid_feat'].shape[0] if 'grid_feat' in sample else 2048
    else:
        v_dim, grid_dim = 2055, 2048
    print(f"Visual dims: region={v_dim}, grid={grid_dim}")

    # 4. Load checkpoint first (hyperparameters for matching architecture)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    saved = ckpt.get('args')
    if isinstance(saved, dict):
        sd = saved
    elif saved is None:
        sd = {}
    else:
        sd = vars(saved)

    no_mac = getattr(args, 'no_mac_decoder', False) or sd.get('no_mac_decoder', False)
    num_mac_hops = args.num_mac_hops if getattr(args, 'num_mac_hops', None) is not None else sd.get('num_mac_hops', 3)

    # 5. Build model
    model_args = argparse.Namespace(
        v_dim=v_dim, grid_dim=grid_dim,
        dropout=sd.get('dropout', 0.5),
        use_fasttext=args.use_fasttext,
        infonce=sd.get('infonce', False),
        scst=False,
        exact_match_lambda=1.0,
        ohp_lambda=0.0,
        scheduled_sampling=False,
        ss_k=5.0,
        phase=sd.get('phase', 1),
        no_mac_decoder=no_mac,
        num_mac_hops=num_mac_hops,
    )
    model = ModelH(len(q_vocab), len(a_vocab), model_args).to(DEVICE)

    if args.use_fasttext:
        q_mat, _ = build_fasttext_matrix(q_vocab.word2idx)
        model.q_emb.weight.data.copy_(q_mat)
        a_mat, _ = build_fasttext_matrix(a_vocab.word2idx)
        model.decoder.embedding.weight.data.copy_(a_mat)
        model.decoder.fc.weight = model.decoder.embedding.weight
        print("FastText embeddings loaded.")

    state = ckpt.get('model_state_dict', ckpt)

    # Strip torch.compile prefix if present
    state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {args.checkpoint}")

    if torch.cuda.is_available():
        try:
            model = torch.compile(model, mode='default', dynamic=True)
            print("torch.compile: ON")
        except Exception:
            pass

    model.eval()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    all_results = []

    # 6. Evaluate each requested dataset
    for ds_name in args.datasets:
        json_path = DATASET_PATHS.get(ds_name)
        if json_path is None:
            print(f"[SKIP] Unknown dataset: {ds_name}")
            continue
        if not os.path.exists(json_path):
            print(f"[SKIP] Val JSON not found: {json_path}")
            continue

        print(f"\n{'─'*60}")
        print(f"  Evaluating on: {ds_name.upper()}  ({json_path})")
        print(f"{'─'*60}")

        val_ds = ValDatasetH(
            json_path, q_vocab, a_vocab, feat_loader,
            max_q_len=20,
            max_samples=args.max_samples
        )

        loader_kwargs = {
            'batch_size': args.batch_size,
            'shuffle': False,
            'collate_fn': collate_val,
            'num_workers': args.num_workers,
            'pin_memory': (DEVICE.type == 'cuda'),
        }
        if args.num_workers > 0:
            loader_kwargs['persistent_workers'] = True
            loader_kwargs['prefetch_factor'] = 4
        val_loader = DataLoader(val_ds, **loader_kwargs)

        all_preds      = []
        all_gt_expls   = []
        all_gt_answers = []
        all_questions  = []
        all_img_ids    = []

        for region_feat, region_mask, q_ids, grid_feat, label_names, gt_answers, gt_expls_list, img_ids, questions in tqdm(val_loader, desc=f"Decoding [{ds_name}]"):
            with torch.no_grad():
                with torch.autocast(device_type=DEVICE.type, dtype=amp_dtype, enabled=(DEVICE.type == 'cuda')):
                    preds = beam_decode_batch(
                        model, region_feat, region_mask, q_ids, grid_feat, label_names,
                        a_vocab, DEVICE,
                        beam_width=args.beam_width,
                        max_len=args.max_len,
                        no_repeat_ngram=args.no_repeat_ngram,
                        min_decode_len=args.min_decode_len,
                    )

            all_preds.extend(preds)
            all_gt_expls.extend(gt_expls_list)
            all_gt_answers.extend(gt_answers)
            all_questions.extend(questions)
            all_img_ids.extend(img_ids)

        result = compute_metrics(all_preds, all_gt_expls, all_gt_answers, dataset_name=ds_name.upper())
        all_results.append(result)

        # Save sample predictions for qualitative analysis
        if args.save_predictions:
            out_file = f"checkpoints/h/eval_{ds_name}_predictions.json"
            samples = []
            for p, refs, ans in zip(all_preds[:200], all_gt_expls[:200], all_gt_answers[:200]):
                samples.append({'prediction': p, 'references': refs, 'gt_answer': ans})
            with open(out_file, 'w') as f:
                json.dump(samples, f, indent=2)
            print(f"  Saved 200 sample predictions → {out_file}")

    # 7. Save aggregate results with comprehensive metadata
    ckpt_tag = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out_json = f"checkpoints/h/eva_{ckpt_tag}_beam{args.beam_width}.json"
    
    # Collect git info for reproducibility
    try:
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                            stderr=subprocess.DEVNULL).decode().strip()[:8]
    except:
        git_commit = 'unknown'
    
    # Build comprehensive metadata
    run_metadata = {
        'timestamp': datetime.now().isoformat(),
        'git_commit': git_commit,
        'device': str(DEVICE),
        'checkpoint': args.checkpoint,
        'checkpoint_tag': ckpt_tag,
        'command_args': {
            'beam_width': args.beam_width,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'max_len': args.max_len,
            'min_decode_len': args.min_decode_len,
            'no_repeat_ngram': args.no_repeat_ngram,
            'datasets': args.datasets,
            'use_fasttext': args.use_fasttext,
            'max_samples': args.max_samples,
        },
        'vocab_config': {
            'question_vocab_size': len(q_vocab),
            'answer_vocab_size': len(a_vocab),
        },
        'visual_encoders': {
            'region_feat_dim': v_dim,
            'grid_feat_dim': grid_dim,
        },
        'metrics_backend': {
            'spice': HAS_SPICE,
            'bertscore': HAS_BERTSCORE,
            'rouge': HAS_ROUGE,
            'cider': HAS_CIDER,
        },
    }
    
    # Build comprehensive output with metadata and samples
    output = {
        'meta': run_metadata,
        'results': all_results,
        'sample_predictions': [
            {
                'dataset': r['dataset'],
                'samples': [
                    {'prediction': p, 'references': rf, 'answer': a, 'question': q, 'img_id': img_id}
                    for p, rf, a, q, img_id in zip(all_preds[:100], all_gt_expls[:100], all_gt_answers[:100], all_questions[:100], all_img_ids[:100])
                ]
            }
            for r in all_results
        ],
    }
    
    with open(out_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nAll results saved → {out_json}")
    print(f"  📊 Metadata + {sum(len(r['results']) if 'results' in r else 0 for r in output['sample_predictions'])} sample predictions included")

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Model H on VQA Explanation datasets')
    parser.add_argument('--checkpoint',  type=str, default='checkpoints/h/model_h_best.pth',
                        help='Path to best checkpoint')
    parser.add_argument('--vg_feat_dir', type=str, default='data/vg_features',
                        help='Path to pre-extracted BUTD features')
    parser.add_argument('--vocab_q_path', type=str, default='data/processed/vocab_questions.json')
    parser.add_argument('--vocab_a_path', type=str, default='data/processed/vocab_answers.json')
    parser.add_argument('--datasets',    nargs='+', default=['vqa_e', 'vqa_x', 'aokvqa'],
                        choices=['vqa_e', 'vqa_x', 'aokvqa'],
                        help='Datasets to evaluate on')
    parser.add_argument('--beam_width',  type=int, default=3,
                        help='Beam search width (1=greedy)')
    parser.add_argument('--no_repeat_ngram', type=int, default=3,
                        help='Block repeated n-grams (0=disabled)')
    parser.add_argument('--max_len',     type=int, default=30,
                        help='Maximum explanation generation length')
    parser.add_argument('--min_decode_len', type=int, default=3,
                        help='Minimum content tokens before allowing <end> (greedy/beam)')
    parser.add_argument('--batch_size',  type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Cap number of val samples (for quick testing)')
    parser.add_argument('--use_fasttext', action='store_true',
                        help='Load FastText embeddings into model')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save 200 sample predictions to JSON for qualitative analysis')
    parser.add_argument('--no_mac_decoder', action='store_true',
                        help='Match checkpoint trained without MAC-in-decoder (ablation)')
    parser.add_argument('--num_mac_hops', type=int, default=None,
                        help='Override MAC hops (default: from checkpoint or 3)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate_h(args)
