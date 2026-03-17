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

Reward: BLEU-4 (simple, no corpus stats needed) — can swap to CIDEr-D later.

Usage in train.py
-----------------
    from training.scst import scst_step

    # Phase 4 training (after Phase 3 converges):
    # loss = scst_step(model, model_type, imgs, questions, target_texts,
    #                  vocab_a, device=DEVICE)
    # scaler.scale(loss).backward()
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# ── Reward ─────────────────────────────────────────────────────────────────────

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
    returns    : (B,) float tensor of BLEU-4 scores
    """
    rewards = [
        _bleu4(h.split(), r.split())
        for h, r in zip(hypotheses, references)
    ]
    return torch.tensor(rewards, dtype=torch.float32)


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
    pad_idx   = vocab_a.word2idx.get('<pad>', 0)
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
              temperature: float = 1.0) -> torch.Tensor:
    """
    Compute SCST policy-gradient loss for one batch.

    Args:
        model        : VQA model
        model_type   : 'A'/'B'/'C'/'D'/'E'
        imgs         : (B, 3, H, W)
        questions    : (B, q_len)
        target_texts : list[str] ground-truth answer texts
        vocab_a      : answer Vocabulary
        max_len      : max decode length
        device       : torch device
        temperature  : sampling temperature (1.0 = unbiased)

    Returns:
        loss : scalar — REINFORCE loss (to be .backward()-ed)
    """
    # 1. Greedy baseline (no gradient)
    greedy_texts = _greedy_decode(model, model_type, imgs, questions,
                                  vocab_a, max_len, device)

    # 2. Sampling decode (with gradient)
    sample_texts, log_prob_sums = _sampling_decode(
        model, model_type, imgs, questions, vocab_a, max_len, device, temperature)

    # 3. Rewards
    r_greedy = compute_bleu4_rewards(greedy_texts, target_texts).to(device)   # (B,)
    r_sample = compute_bleu4_rewards(sample_texts, target_texts).to(device)   # (B,)

    # 4. REINFORCE: L = -mean(advantage * log_prob_sum)
    advantage = (r_sample - r_greedy).detach()            # (B,) — no grad through reward
    loss = -(advantage * log_prob_sums).mean()

    return loss


if __name__ == "__main__":
    # Standalone test of reward function
    hyps = ["yes it is a cat", "the dog is running", "a red apple"]
    refs = ["yes it is a cat", "the dog is running fast", "a green apple"]
    rewards = compute_bleu4_rewards(hyps, refs)
    print(f"BLEU-4 rewards: {rewards.tolist()}")
    # Expect: [1.0, high, low]
    print("SCST reward test PASSED")
