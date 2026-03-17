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
        model, model_type, imgs, questions, target_texts, vocab_a,
        device=DEVICE,
        bleu_weight=0.5, meteor_weight=0.5, length_bonus_weight=0.0,
    )
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple
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


# ── Composite reward ────────────────────────────────────────────────────────────

def compute_rewards(
    hypotheses: List[str],
    references: List[str],
    bleu_weight: float = 0.5,
    meteor_weight: float = 0.5,
    length_bonus_weight: float = 0.0,
) -> torch.Tensor:
    """
    Composite reward: bleu_w*BLEU4 + meteor_w*METEOR + len_w*LengthBonus.

    Weights do not need to sum to 1.0 — the advantage (r_sample - r_greedy)
    is what drives the gradient, not the absolute scale.

    Args:
        hypotheses           : decoded texts (space-tokenized)
        references           : ground-truth texts (space-tokenized)
        bleu_weight          : BLEU-4 coefficient (default 0.5)
        meteor_weight        : METEOR coefficient (default 0.5)
        length_bonus_weight  : LengthBonus coefficient (default 0.0)
                               Use ≤0.05; higher values cause verbose drift.

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

    return r


# ── Greedy decode helper ────────────────────────────────────────────────────────

def _greedy_decode(model, model_type, imgs, questions, vocab_a,
                   max_len: int, device) -> List[str]:
    """
    Greedy decode — used to produce the SCST baseline.
    Delegates to model's existing decode infrastructure.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from inference import batch_greedy_decode_with_attention, batch_greedy_decode

    model.eval()
    with torch.no_grad():
        if model_type in ('C', 'D', 'E'):
            texts = batch_greedy_decode_with_attention(
                model, imgs, questions, vocab_a,
                max_len=max_len, device=device)
        else:
            texts = batch_greedy_decode(
                model, imgs, questions, vocab_a,
                max_len=max_len, device=device)
    model.train()
    return texts


# ── Sampling decode ─────────────────────────────────────────────────────────────

def _sampling_decode(model, model_type, imgs, questions, vocab_a,
                     max_len: int, device, temperature: float = 1.0
                     ) -> Tuple[List[str], torch.Tensor]:
    """
    Temperature-sampling decode.  Records per-step log-probabilities so we can
    compute the REINFORCE gradient.

    Returns:
      texts     : list[str] — decoded text for each batch sample
      log_probs : (B,) — sum of log P(a_t) over all non-pad steps
    """
    model.train()
    end_idx   = vocab_a.word2idx.get('<end>', 2)
    start_idx = vocab_a.word2idx.get('<start>', 1)
    B = imgs.size(0)

    # ── Encode image + question ────────────────────────────────────────────
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
            w = vocab_a.idx2word.get(i, '<unk>') if hasattr(vocab_a, 'idx2word') else \
                vocab_a.idx_to_word.get(i, '<unk>')
            if w not in ('<pad>', '<start>'):
                words.append(w)
        texts.append(' '.join(words))

    return texts, log_prob_sums   # (B,)


# ── Main SCST step ──────────────────────────────────────────────────────────────

def scst_step(model, model_type: str, imgs: torch.Tensor, questions: torch.Tensor,
              target_texts: List[str], vocab_a,
              max_len: int = 50, device='cpu',
              temperature: float = 1.0,
              bleu_weight: float = 0.5,
              meteor_weight: float = 0.5,
              length_bonus_weight: float = 0.0,
              ) -> torch.Tensor:
    """
    Compute SCST policy-gradient loss for one batch.

    Args:
        model                : VQA model
        model_type           : 'A'/'B'/'C'/'D'/'E'/'F'
        imgs                 : (B, 3, H, W) or (B, k, feat_dim) for Model F
        questions            : (B, q_len)
        target_texts         : list[str] ground-truth answer texts
        vocab_a              : answer Vocabulary
        max_len              : max decode length
        device               : torch device
        temperature          : sampling temperature (1.0 = unbiased)
        bleu_weight          : BLEU-4 reward coefficient (default 0.5)
        meteor_weight        : METEOR reward coefficient (default 0.5)
        length_bonus_weight  : LengthBonus coefficient (default 0.0)

    Returns:
        loss : scalar — REINFORCE loss (to be .backward()-ed)
    """
    # 1. Greedy baseline (no gradient)
    greedy_texts = _greedy_decode(model, model_type, imgs, questions,
                                  vocab_a, max_len, device)

    # 2. Sampling decode (with gradient)
    sample_texts, log_prob_sums = _sampling_decode(
        model, model_type, imgs, questions, vocab_a, max_len, device, temperature)

    # 3. Composite rewards
    r_greedy = compute_rewards(
        greedy_texts, target_texts,
        bleu_weight=bleu_weight,
        meteor_weight=meteor_weight,
        length_bonus_weight=length_bonus_weight,
    ).to(device)

    r_sample = compute_rewards(
        sample_texts, target_texts,
        bleu_weight=bleu_weight,
        meteor_weight=meteor_weight,
        length_bonus_weight=length_bonus_weight,
    ).to(device)

    # 4. REINFORCE: L = -mean(advantage * log_prob_sum)
    advantage = (r_sample - r_greedy).detach()   # (B,) — no grad through reward
    loss = -(advantage * log_prob_sums).mean()

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
