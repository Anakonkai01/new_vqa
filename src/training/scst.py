"""
Tier-8: Self-Critical Sequence Training (SCST)
================================================
REINFORCE with greedy-decode baseline (Rennie et al., 2017).

Core idea:
  - Greedy decode  → baseline reward  r_greedy  (no gradient needed)
  - Sampling decode → sample reward   r_sample  (need log-probabilities)
  - Advantage       = r_sample - r_greedy
  - Policy gradient loss = -mean(advantage * sum_t(log P(a_t | context)))

Why greedy as baseline: SCST uses the model's own greedy output as a
no-variance baseline.  Any sample that beats greedy gets positive reinforcement;
anything worse gets negative reinforcement.  This eliminates reward centering
ambiguity while keeping variance low.

Composite Reward (default)
--------------------------
R = bleu_w * BLEU-4  +  meteor_w * METEOR  [+  len_w * LengthBonus]

Rationale for BLEU-4 + METEOR composite:
  - BLEU-4 measures n-gram precision (4-gram level). Penalizes hallucination.
  - METEOR measures alignment-based recall with stemming + synonym matching.
    Example: "riding a bicycle" vs "cycling" → BLEU-4=0.0, METEOR≈0.5.
    VQA-E explanations are paraphrase-heavy — METEOR is critical.
  - Both are normalized to [0,1] — no scaling conflict.
  - Default weights 0.5/0.5 balance precision (BLEU) and recall (METEOR).

Length Bonus (optional, default weight=0.0)
  - Encourages outputs in the 8–20 word range (typical VQA-E explanation).
  - Use sparingly (weight≤0.05) — larger values cause verbose outputs.

Why NOT add CIDEr-D / ROUGE-L:
  - CIDEr-D requires corpus TF-IDF precomputation (one-time but nontrivial).
  - ROUGE-L (LCS) is already largely captured by BLEU-4 for sentence-length text.
  - Adding more metrics multiplies REINFORCE gradient variance.

Usage in train.py
-----------------
    from training.scst import scst_step

    loss = scst_step(
        model, model_type, imgs, questions, target_texts, vocab,
        device=DEVICE,
        bleu_weight=0.5, meteor_weight=0.5, length_bonus_weight=0.0,
    )
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# ── BLEU-4 ─────────────────────────────────────────────────────────────────────

def _bleu4(hyp_tokens: List[str], ref_tokens: List[str]) -> float:
    """BLEU-4 with Chen & Cherry smoothing to handle short sequences."""
    smooth = SmoothingFunction().method1
    return sentence_bleu(
        [ref_tokens], hyp_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smooth,
    )


def compute_bleu4_rewards(hypotheses: List[str], references: List[str]) -> torch.Tensor:
    """
    hypotheses : list[str] of space-tokenized decoded text
    references : list[str] of space-tokenized ground truth text
    returns    : (B,) float tensor of BLEU-4 scores in [0,1]
    """
    rewards = [
        _bleu4(h.split(), r.split())
        for h, r in zip(hypotheses, references)
    ]
    return torch.tensor(rewards, dtype=torch.float32)


# ── METEOR ──────────────────────────────────────────────────────────────────────

try:
    from nltk.translate.meteor_score import single_meteor_score as _nltk_meteor
    _METEOR_AVAILABLE = True
except ImportError:
    _METEOR_AVAILABLE = False


def _meteor(hyp_tokens: List[str], ref_tokens: List[str]) -> float:
    """
    METEOR score for a single hypothesis/reference pair.

    METEOR advantages over BLEU-4 for VQA-E:
      - Handles stemming: "cycling" matches "cycles"
      - Handles synonym alignment (WordNet, if available)
      - F-measure based: balances precision AND recall
      - Better correlation with human judgment on short explanatory text

    Falls back to a simple unigram F1 if NLTK meteor is unavailable.
    """
    if not hyp_tokens or not ref_tokens:
        return 0.0

    if _METEOR_AVAILABLE:
        # nltk >= 3.7: single_meteor_score expects Iterable[str] (token lists)
        return float(_nltk_meteor(ref_tokens, hyp_tokens))

    # Fallback: unigram F1 (no stemming/synonyms, but at least recall-aware)
    hyp_set = set(hyp_tokens)
    ref_set = set(ref_tokens)
    if not hyp_set or not ref_set:
        return 0.0
    p = len(hyp_set & ref_set) / len(hyp_set)
    r = len(hyp_set & ref_set) / len(ref_set)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)   # F1


def compute_meteor_rewards(hypotheses: List[str], references: List[str]) -> torch.Tensor:
    """
    hypotheses : list[str] of space-tokenized decoded text
    references : list[str] of space-tokenized ground truth text
    returns    : (B,) float tensor of METEOR scores in [0,1]
    """
    rewards = [
        _meteor(h.split(), r.split())
        for h, r in zip(hypotheses, references)
    ]
    return torch.tensor(rewards, dtype=torch.float32)


# ── Length Bonus ────────────────────────────────────────────────────────────────

def compute_length_bonus(
    hypotheses: List[str],
    target_min: int = 8,
    target_max: int = 20,
) -> torch.Tensor:
    """
    Small reward that peaks at 1.0 for outputs in [target_min, target_max] words.
    Ramps linearly below target_min; decays linearly above target_max.

    Typical VQA-E explanation: "yellow because bananas are naturally yellow"
    → 8 words → hits target_min.

    Use with weight ≤ 0.05 to avoid verbose generation.

           1.0 ┤    ████████████
               │   █            █
               │  █              █
               │ █                █
           0.0 ┤──────────────────────
               0   min           max  (words)
    """
    rewards = []
    span = target_max - target_min
    for h in hypotheses:
        n = len(h.split())
        if n == 0:
            rewards.append(0.0)
        elif n < target_min:
            rewards.append(n / target_min)
        elif n <= target_max:
            rewards.append(1.0)
        else:
            # Soft penalty above max — not a hard cliff
            penalty = (n - target_max) / span
            rewards.append(max(0.0, 1.0 - penalty))
    return torch.tensor(rewards, dtype=torch.float32)


# ── G4: Object Hallucination Penalty ───────────────────────────────────────────

_OHP_STOPWORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'on',
    'at', 'by', 'for', 'with', 'about', 'from', 'into', 'through',
    'it', 'its', 'this', 'that', 'and', 'or', 'but', 'not', 'no', 'so',
    'yet', 'both', 'he', 'she', 'they', 'we', 'you', 'i', 'me', 'him',
    'her', 'them', 'us', 'my', 'your', 'his', 'their', 'our', 'which',
    'who', 'what', 'there', 'here', 'because', 'since', 'as',
})


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two numpy vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-10:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_ohp_reward(
    hyp_tokens: List[str],
    visual_labels: List[str],
    glove_embed: dict,
    threshold: float = 0.5,
) -> float:
    """
    Object Hallucination Penalty for a single sample (G4, Eq 37-38).

    For each content word w in hypothesis:
        max_sim(w) = max over all label tokens l: cosine(glove[w], glove[l])
        penalty(w) = max(0, delta - max_sim(w))
    OHP = mean(penalty) over content words that appear in GloVe.

    A content word with max_sim >= delta gets 0 penalty (it matches a visible object).
    A content word absent from labels gets penalty = delta (maximum hallucination).

    Args:
        hyp_tokens    : list[str] — decoded tokens (already split)
        visual_labels : list[str] — BUTD label names (e.g. ["cat", "fire hydrant"])
        glove_embed   : dict[str, np.ndarray] — GloVe word vectors
        threshold     : delta in Eq 37 (default 0.5 per spec)

    Returns:
        OHP ∈ [0, delta] — higher = more hallucination
    """
    # Tokenize label names and build lookup set of label vecs
    label_vecs = []
    for label in visual_labels:
        for tok in label.lower().split():
            if tok in glove_embed:
                label_vecs.append(glove_embed[tok])

    content_words = [
        w for w in hyp_tokens
        if w not in _OHP_STOPWORDS
        and w not in ('<start>', '<end>', '<pad>', '<unk>')
        and w in glove_embed
    ]

    if not content_words:
        return 0.0

    penalties = []
    for word in content_words:
        w_vec = glove_embed[word]
        if label_vecs:
            max_sim = max(_cosine_sim(w_vec, lv) for lv in label_vecs)
        else:
            max_sim = 0.0
        penalties.append(max(0.0, threshold - max_sim))

    return sum(penalties) / len(penalties)


def compute_ohp_rewards(
    hypotheses: List[str],
    visual_labels_batch: List[List[str]],
    glove_embed: dict,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Batch OHP reward computation.

    Args:
        hypotheses          : list[str] — decoded texts (space-tokenized)
        visual_labels_batch : list[list[str]] — BUTD label names per sample
        glove_embed         : dict[str, np.ndarray]
        threshold           : delta (default 0.5)

    Returns:
        (B,) float tensor — OHP per sample ∈ [0, threshold]
    """
    rewards = [
        compute_ohp_reward(h.split(), labels, glove_embed, threshold)
        for h, labels in zip(hypotheses, visual_labels_batch)
    ]
    return torch.tensor(rewards, dtype=torch.float32)


# ── Composite reward ────────────────────────────────────────────────────────────

def compute_exact_match(hypotheses: List[str], references: List[str]) -> torch.Tensor:
    """Exact string match metric for generative VQA."""
    r = []
    for h, ref in zip(hypotheses, references):
        # normalize and compare
        h_norm = " ".join(h.strip().split())
        r_norm = " ".join(ref.strip().split())
        r.append(1.0 if h_norm == r_norm else 0.0)
    return torch.tensor(r, dtype=torch.float32)

def compute_rewards(
    hypotheses: List[str],
    references: List[str],
    bleu_weight: float = 0.5,
    meteor_weight: float = 0.5,
    length_bonus_weight: float = 0.0,
    ohp_weight: float = 0.0,
    ohp_tensor: Optional[torch.Tensor] = None,
    cider_weight: float = 0.0,
    exact_match_weight: float = 0.0,
) -> torch.Tensor:
    """
    Composite reward: bleu_w*BLEU4 + meteor_w*METEOR + len_w*LengthBonus - ohp_w*OHP.

    Weights do not need to sum to 1.0 — the advantage (r_sample - r_greedy)
    is what drives the gradient, not the absolute scale.

    G4: OHP is subtracted (penalty): R -= ohp_weight * OHP.
        Pass pre-computed ohp_tensor (B,) to avoid recomputing GloVe lookups.

    Args:
        hypotheses           : decoded texts (space-tokenized)
        references           : ground-truth texts (space-tokenized)
        bleu_weight          : BLEU-4 coefficient (default 0.5)
        meteor_weight        : METEOR coefficient (default 0.5)
        length_bonus_weight  : LengthBonus coefficient (default 0.0)
        ohp_weight           : G4 OHP coefficient (default 0.0 = disabled)
        ohp_tensor           : pre-computed (B,) OHP per sample; required when ohp_weight>0

    Returns:
        (B,) float tensor
    """
    r = torch.zeros(len(hypotheses))

    if bleu_weight > 0.0:
        r = r + bleu_weight * compute_bleu4_rewards(hypotheses, references)

    if meteor_weight > 0.0:
        r = r + meteor_weight * compute_meteor_rewards(hypotheses, references)

    if length_bonus_weight > 0.0:
        r = r + length_bonus_weight * compute_length_bonus(hypotheses)
        
    if exact_match_weight > 0.0:
        r = r + exact_match_weight * compute_exact_match(hypotheses, references)

    if cider_weight > 0.0:
        try:
            from .cider import compute_cider
            res = {str(i): [h] for i, h in enumerate(hypotheses)}
            gts = {str(i): [r] for i, r in enumerate(references)}
            _, scores = compute_cider(gts, res)
            r = r + cider_weight * torch.tensor(scores, dtype=torch.float32)
        except Exception as e:
            print(f"  [WARN] CIDEr calculation failed: {e}")

    if ohp_weight > 0.0 and ohp_tensor is not None:
        # r may be CPU (BLEU/METEOR/CIDEr); ohp_tensor is often CUDA from training
        r = r.to(device=ohp_tensor.device, dtype=torch.float32) - ohp_weight * ohp_tensor.float()

    return r


# ── Greedy decode helper ────────────────────────────────────────────────────────

def _greedy_decode(model, model_type, imgs, questions, vocab,
                   max_len: int, device, q_token_ids=None,
                   grid_feats: Optional[torch.Tensor] = None,
                   img_mask: Optional[torch.Tensor] = None,
                   label_tokens: Optional[torch.Tensor] = None) -> List[str]:
    """
    Greedy decode — used to produce the SCST baseline.
    Delegates to model's existing decode infrastructure.
    q_token_ids: pass questions tensor so PGN activates symmetrically with _sampling_decode.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from inference import batch_greedy_decode_with_attention, batch_greedy_decode

    model.eval()
    with torch.no_grad():
        if model_type == 'G':
            # VQAModel API — feats = imgs (BUTD features for Model G)
            B         = imgs.size(0)
            end_idx   = vocab.word2idx.get('<end>', 2)
            start_idx = vocab.word2idx.get('<start>', 1)

            V, q_feat, Q_H = model.encode(imgs, questions)
            h, c = model._fuse_and_init(V, q_feat, img_mask=None)
            # G5: always LONG bin (2) at inference
            length_bin = torch.full((B,), 2, dtype=torch.long, device=device)

            current_token = torch.full((B,), start_idx, dtype=torch.long, device=device)
            done          = torch.zeros(B, dtype=torch.bool, device=device)
            decoded_ids   = [[] for _ in range(B)]
            coverage      = None

            for _ in range(max_len):
                tok = current_token.unsqueeze(1)
                logit, h, c, _, coverage = model.decode_step(
                    tok, h, c, V, Q_H,
                    coverage=coverage,
                    q_token_ids=questions,
                    length_bin=length_bin,
                )
                pred = logit.argmax(dim=-1)   # (B,)
                for b in range(B):
                    if not done[b]:
                        decoded_ids[b].append(pred[b].item())
                done = done | (pred == end_idx)
                current_token = pred
                if done.all():
                    break

            texts = []
            for ids in decoded_ids:
                words = []
                for i in ids:
                    if i == end_idx:
                        break
                    w = vocab.idx2word.get(i, '<unk>') if hasattr(vocab, 'idx2word') \
                        else vocab.idx_to_word.get(i, '<unk>')
                    if w not in ('<pad>', '<start>'):
                        words.append(w)
                texts.append(' '.join(words))

        elif model_type == 'H':
            B         = imgs.size(0)
            end_idx   = vocab.word2idx.get('<end>', 2)
            start_idx = vocab.word2idx.get('<start>', 1)

            memory, Q_H, _, kb, _v_raw, _ = model.encode(
                questions, imgs, grid_feats=grid_feats, img_mask=img_mask)
            V = kb
            h, c = model.init_decoder_hidden(memory)
            mm = None if getattr(model.args, 'no_mac_decoder', False) else memory

            length_bin = torch.full((B,), 2, dtype=torch.long, device=questions.device)

            current_token = torch.full((B,), start_idx, dtype=torch.long, device=device)
            done          = torch.zeros(B, dtype=torch.bool, device=device)
            decoded_ids   = [[] for _ in range(B)]
            coverage      = None

            for _ in range(max_len):
                tok = current_token.unsqueeze(1)
                logit, h, c, _, coverage = model.decode_step(
                    tok, h, c, V, Q_H,
                    coverage=coverage,
                    q_token_ids=questions,
                    length_bin=length_bin,
                    label_tokens=label_tokens,
                    img_mask=img_mask,
                    mac_memory=mm,
                )
                pred = logit.argmax(dim=-1)
                for b in range(B):
                    if not done[b]:
                        decoded_ids[b].append(pred[b].item())
                done = done | (pred == end_idx)
                current_token = pred
                if done.all():
                    break

            texts = []
            for ids in decoded_ids:
                words = []
                for i in ids:
                    if i == end_idx:
                        break
                    w = vocab.idx2word.get(i, '<unk>') if hasattr(vocab, 'idx2word') \
                        else vocab.idx_to_word.get(i, '<unk>')
                    if w not in ('<pad>', '<start>'):
                        words.append(w)
                texts.append(' '.join(words))

        elif model_type in ('C', 'D', 'E', 'F'):
            texts = batch_greedy_decode_with_attention(
                model, imgs, questions, vocab,
                max_len=max_len, device=device)
            # PGN symmetry: q_token_ids is already passed inside
            # batch_greedy_decode_with_attention via its qs variable
        else:
            texts = batch_greedy_decode(
                model, imgs, questions, vocab,
                max_len=max_len, device=device)
    model.train()
    return texts


# ── Sampling decode ─────────────────────────────────────────────────────────────

def _sampling_decode(model, model_type, imgs, questions, vocab,
                     max_len: int, device, temperature: float = 1.0,
                     grid_feats: Optional[torch.Tensor] = None,
                     img_mask: Optional[torch.Tensor] = None,
                     label_tokens: Optional[torch.Tensor] = None,
                     ) -> Tuple[List[str], torch.Tensor]:
    """
    Temperature-sampling decode.  Records per-step log-probabilities so we can
    compute the REINFORCE gradient.

    Returns:
      texts     : list[str] — decoded text for each batch sample
      log_probs : (B,) — sum of log P(a_t) over all non-pad steps
    """
    model.train()
    end_idx   = vocab.word2idx.get('<end>', 2)
    start_idx = vocab.word2idx.get('<start>', 1)
    B = imgs.size(0)

    if model_type == 'G':
        # ── VQAModel API (Model G) ─────────────────────────────────────────
        with torch.no_grad():
            V, q_feat, Q_H = model.encode(imgs, questions)
            h, c = model._fuse_and_init(V, q_feat, img_mask=None)
        # G5: always LONG bin at inference
        length_bin    = torch.full((B,), 2, dtype=torch.long, device=device)
        current_token = torch.full((B,), start_idx, dtype=torch.long, device=device)
        done          = torch.zeros(B, dtype=torch.bool, device=device)
        log_prob_sums = torch.zeros(B, device=device)
        decoded_ids   = [[] for _ in range(B)]
        coverage      = None

        for _ in range(max_len):
            tok = current_token.unsqueeze(1)
            logit, h, c, _, coverage = model.decode_step(
                tok, h, c, V, Q_H,
                coverage=coverage,
                q_token_ids=questions,
                length_bin=length_bin,
            )
            if temperature != 1.0:
                logit = logit / temperature
            probs   = F.softmax(logit, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(1)
            log_p   = F.log_softmax(logit, dim=-1)
            step_lp = log_p.gather(1, sampled.unsqueeze(1)).squeeze(1)
            log_prob_sums = log_prob_sums + step_lp * (~done).float()
            for b in range(B):
                if not done[b]:
                    decoded_ids[b].append(sampled[b].item())
            done = done | (sampled == end_idx)
            current_token = sampled
            if done.all():
                break

        texts = []
        for ids in decoded_ids:
            words = []
            for i in ids:
                if i == end_idx:
                    break
                w = vocab.idx2word.get(i, '<unk>') if hasattr(vocab, 'idx2word') \
                    else vocab.idx_to_word.get(i, '<unk>')
                if w not in ('<pad>', '<start>'):
                    words.append(w)
            texts.append(' '.join(words))
        return texts, log_prob_sums

    elif model_type == 'H':
        memory, Q_H, _, kb, _v_raw, _ = model.encode(
            questions, imgs, grid_feats=grid_feats, img_mask=img_mask)
        V = kb
        h, c = model.init_decoder_hidden(memory)
        mm = None if getattr(model.args, 'no_mac_decoder', False) else memory

        length_bin    = torch.full((B,), 2, dtype=torch.long, device=questions.device)
        current_token = torch.full((B,), start_idx, dtype=torch.long, device=device)
        done          = torch.zeros(B, dtype=torch.bool, device=device)
        log_prob_sums = torch.zeros(B, device=device)
        decoded_ids   = [[] for _ in range(B)]
        coverage      = None

        for _ in range(max_len):
            tok = current_token.unsqueeze(1)
            logit, h, c, _, coverage = model.decode_step(
                tok, h, c, V, Q_H,
                coverage=coverage,
                q_token_ids=questions,
                length_bin=length_bin,
                label_tokens=label_tokens,
                img_mask=img_mask,
                mac_memory=mm,
            )
            if temperature != 1.0:
                logit = logit / temperature
            probs   = F.softmax(logit, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(1)
            log_p   = F.log_softmax(logit, dim=-1)
            step_lp = log_p.gather(1, sampled.unsqueeze(1)).squeeze(1)
            log_prob_sums = log_prob_sums + step_lp * (~done).float()
            for b in range(B):
                if not done[b]:
                    decoded_ids[b].append(sampled[b].item())
            done = done | (sampled == end_idx)
            current_token = sampled
            if done.all():
                break

        texts = []
        for ids in decoded_ids:
            words = []
            for i in ids:
                if i == end_idx:
                    break
                w = vocab.idx2word.get(i, '<unk>') if hasattr(vocab, 'idx2word') \
                    else vocab.idx_to_word.get(i, '<unk>')
                if w not in ('<pad>', '<start>'):
                    words.append(w)
            texts.append(' '.join(words))
        return texts, log_prob_sums

    # ── Encode image + question (models A–F) ──────────────────────────────
    with torch.no_grad():
        if model_type in ('C', 'D', 'E'):
            img_features = F.normalize(model.i_encoder(imgs), p=2, dim=-1)  # (B,N,H)
            img_mean     = img_features.mean(dim=1)
            q_feature, q_hidden = model.q_encoder(questions)
            if model_type == 'E':
                fused = model.fusion(q_feature, img_mean)
            else:
                fused = model.fusion(img_mean, q_feature)
        else:
            img_feat = F.normalize(model.i_encoder(imgs), p=2, dim=1)
            q_feature, _ = model.q_encoder(questions)
            fused = model.fusion(img_feat, q_feature)

    h = fused.unsqueeze(0).repeat(model.num_layers, 1, 1)
    c = torch.zeros_like(h)
    hidden = (h, c)

    # ── Step-by-step sampling ──────────────────────────────────────────────
    current_token = torch.full((B,), start_idx, dtype=torch.long, device=device)
    done          = torch.zeros(B, dtype=torch.bool, device=device)
    log_prob_sums = torch.zeros(B, device=device)
    decoded_ids   = [[] for _ in range(B)]

    for _ in range(max_len):
        tok = current_token.unsqueeze(1)  # (B, 1)

        if model_type in ('C', 'D', 'E'):
            logit, hidden, _, _ = model.decoder.decode_step(
                tok, hidden, img_features, q_hidden, q_token_ids=questions)
        else:
            emb     = model.decoder.dropout(model.decoder.embedding(tok))
            if model.decoder.embed_proj is not None:
                emb = model.decoder.embed_proj(emb)
            out, hidden = model.decoder.lstm(emb, hidden)
            logit = model.decoder.fc(model.decoder.out_proj(out.squeeze(1)))

        # Temperature scaling + sample
        if temperature != 1.0:
            logit = logit / temperature
        probs = F.softmax(logit, dim=-1)                     # (B, V)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(1)  # (B,)

        # Accumulate log prob for non-finished samples only
        log_p = F.log_softmax(logit, dim=-1)                 # (B, V)
        step_lp = log_p.gather(1, sampled.unsqueeze(1)).squeeze(1)  # (B,)
        log_prob_sums = log_prob_sums + step_lp * (~done).float()

        for b in range(B):
            if not done[b]:
                decoded_ids[b].append(sampled[b].item())

        done = done | (sampled == end_idx)
        current_token = sampled
        if done.all():
            break

    # Decode IDs → text
    texts = []
    for ids in decoded_ids:
        words = []
        for i in ids:
            if i == end_idx:
                break
            w = vocab.idx2word.get(i, '<unk>') if hasattr(vocab, 'idx2word') else \
                vocab.idx_to_word.get(i, '<unk>')
            if w not in ('<pad>', '<start>'):
                words.append(w)
        texts.append(' '.join(words))

    return texts, log_prob_sums   # (B,)


# ── Main SCST step ──────────────────────────────────────────────────────────────

def scst_step(model, model_type: str, imgs: torch.Tensor, questions: torch.Tensor,
              target_texts: List[str], vocab,
              max_len: int = 50, device='cpu',
              temperature: float = 1.0,
              bleu_weight: float = 0.5,
              meteor_weight: float = 0.5,
              length_bonus_weight: float = 0.0,
              ohp_weight: float = 0.0,
              cider_weight: float = 0.0,
              exact_match_weight: float = 0.0,
              visual_labels_batch: Optional[List[List[str]]] = None,
              glove_embed: Optional[dict] = None,
              ohp_threshold: float = 0.5,
              return_stats: bool = False,
              grid_feats: Optional[torch.Tensor] = None,
              img_mask: Optional[torch.Tensor] = None,
              label_tokens: Optional[torch.Tensor] = None,
              ) -> torch.Tensor:
    """
    Compute SCST policy-gradient loss for one batch.

    Args:
        model                : VQA model
        model_type           : 'A'/'B'/'C'/'D'/'E'/'F'/'G'
        imgs                 : (B, 3, H, W) or (B, k, feat_dim) for Model F/G
        questions            : (B, q_len)
        target_texts         : list[str] ground-truth answer texts
        vocab                : answer Vocabulary
        max_len              : max decode length
        device               : torch device
        temperature          : sampling temperature (1.0 = unbiased)
        bleu_weight          : BLEU-4 reward coefficient (default 0.5)
        meteor_weight        : METEOR reward coefficient (default 0.5)
        length_bonus_weight  : LengthBonus coefficient (default 0.0)
        ohp_weight           : G4 OHP penalty coefficient (default 0.0 = disabled)
        visual_labels_batch  : G4 list[list[str]] — BUTD label names per sample
        glove_embed          : G4 dict[str, np.ndarray] — GloVe vectors for OHP
        ohp_threshold        : G4 delta for max(0, delta - cos_sim) (default 0.5)

    Returns:
        loss : scalar — REINFORCE loss (to be .backward()-ed)
    """
    # 1. Greedy baseline (no gradient) — pass questions so PGN activates symmetrically
    greedy_texts = _greedy_decode(
        model, model_type, imgs, questions,
        vocab, max_len, device, q_token_ids=questions,
        grid_feats=grid_feats, img_mask=img_mask, label_tokens=label_tokens,
    )

    # 2. Sampling decode (with gradient)
    sample_texts, log_prob_sums = _sampling_decode(
        model, model_type, imgs, questions, vocab, max_len, device, temperature,
        grid_feats=grid_feats, img_mask=img_mask, label_tokens=label_tokens,
    )

    # 3. G4: pre-compute OHP for sample (greedy OHP not needed — same visual labels)
    ohp_sample = None
    if ohp_weight > 0.0 and visual_labels_batch is not None and glove_embed is not None:
        ohp_sample = compute_ohp_rewards(
            sample_texts, visual_labels_batch, glove_embed, ohp_threshold).to(device)

    # 4. Composite rewards
    r_greedy = compute_rewards(
        greedy_texts, target_texts,
        bleu_weight=bleu_weight,
        meteor_weight=meteor_weight,
        length_bonus_weight=length_bonus_weight,
        ohp_weight=ohp_weight,
        ohp_tensor=None,
        cider_weight=cider_weight,
        exact_match_weight=exact_match_weight,
    ).to(device)

    r_sample = compute_rewards(
        sample_texts, target_texts,
        bleu_weight=bleu_weight,
        meteor_weight=meteor_weight,
        length_bonus_weight=length_bonus_weight,
        ohp_weight=ohp_weight,
        ohp_tensor=ohp_sample,
        cider_weight=cider_weight,
        exact_match_weight=exact_match_weight,
    ).to(device)

    # 5. REINFORCE: L = -mean(advantage * log_prob_sum)
    advantage = (r_sample - r_greedy).detach()   # (B,) — no grad through reward
    loss = -(advantage * log_prob_sums).mean()

    if return_stats:
        stats = {
            'scst/reward_greedy':   r_greedy.mean().item(),
            'scst/reward_sample':   r_sample.mean().item(),
            'scst/advantage_mean':  advantage.mean().item(),
            'scst/advantage_std':   advantage.std().item(),
            'scst/reward_delta':    (r_sample - r_greedy).mean().item(),
        }
        if ohp_sample is not None:
            stats['scst/ohp_mean'] = ohp_sample.mean().item()
        return loss, stats
    return loss


if __name__ == "__main__":
    # Test all reward functions
    hyps = [
        "yes it is a cat",
        "the dog is riding a bicycle",   # synonym test: cycling vs bicycle
        "a red apple on the table",
        "short",                          # length penalty test
    ]
    refs = [
        "yes it is a cat",
        "the dog is cycling near the park",
        "a green apple",
        "a very long explanation about why the sky is blue because of rayleigh scattering",
    ]

    b4  = compute_bleu4_rewards(hyps, refs)
    met = compute_meteor_rewards(hyps, refs)
    lb  = compute_length_bonus(hyps)
    comp = compute_rewards(hyps, refs, bleu_weight=0.5, meteor_weight=0.5)

    print(f"{'Hypothesis':<45}  {'BLEU4':>6}  {'METEOR':>7}  {'LenBonus':>9}  {'Composite':>10}")
    print("─" * 85)
    for i, h in enumerate(hyps):
        print(f"{h:<45}  {b4[i]:>6.3f}  {met[i]:>7.3f}  {lb[i]:>9.3f}  {comp[i]:>10.3f}")

    # Verify METEOR > BLEU-4 for the synonym case
    assert met[1] > b4[1], "METEOR should score higher than BLEU-4 for synonym paraphrase"
    print("\nAll reward tests PASSED")
    print(f"METEOR available (NLTK): {_METEOR_AVAILABLE}")
