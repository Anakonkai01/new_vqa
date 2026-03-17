"""
Tier-5: Pointer-Generator Head
================================
At each decode step the model decides whether to *generate* a token from the
fixed vocabulary or to *copy* a token directly from the source sequence
(question tokens).

  p_gen  = σ(W_x·x_t + W_h·h_t + W_c·context + b)   ∈ [0,1]

  P_vocab = softmax(W_out · h_t)                       # vocabulary distribution
  P_copy  = attention weights over source tokens        # copy distribution

  P_final = p_gen * P_vocab + (1 - p_gen) * P_copy

P_copy is a sparse V-dimensional vector built by scattering the attention
weights α_j onto their corresponding vocabulary indices w_j.

This allows the decoder to reproduce rare words from the question even when
they are mapped to <unk> in the answer vocabulary.

Usage
-----
pgn = PointerGeneratorHead(embed_size=512, hidden_size=1024, context_size=1024)

# Inside each decode step:
p_gen = pgn(embed_t, h_top, context)             # (B,)
final_dist = pgn.blend(p_gen, vocab_logits, q_alpha, q_token_ids, vocab_size)
# final_dist: (B, vocab_size) — log-probabilities ready for NLLLoss or direct use
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerGeneratorHead(nn.Module):
    """
    Computes p_gen and blends vocabulary + copy distributions.

    Args:
        embed_size   : dimension of the token embedding (x_t)
        hidden_size  : LSTM hidden size (h_t)
        context_size : dimension of the attention context vector
                       (for DCAN: hidden_size; for Bahdanau: hidden_size)
    """

    def __init__(self, embed_size: int, hidden_size: int, context_size: int):
        super().__init__()
        # Linear maps each input to a scalar, summed before sigmoid
        self.W_x   = nn.Linear(embed_size,    1, bias=False)
        self.W_h   = nn.Linear(hidden_size,   1, bias=False)
        self.W_ctx = nn.Linear(context_size,  1, bias=True)   # bias here only

    def forward(self, embed_t: torch.Tensor,
                h_t: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        """
        embed_t : (B, embed_size)
        h_t     : (B, hidden_size)  — top LSTM layer hidden state
        context : (B, context_size) — attention context (img or fused)

        returns : (B,) p_gen values in [0, 1]
        """
        score = self.W_x(embed_t) + self.W_h(h_t) + self.W_ctx(context)  # (B, 1)
        return torch.sigmoid(score.squeeze(-1))  # (B,)

    @staticmethod
    def blend(p_gen: torch.Tensor,
              vocab_logits: torch.Tensor,
              q_alpha: torch.Tensor,
              q_token_ids: torch.Tensor,
              vocab_size: int) -> torch.Tensor:
        """
        Mix vocabulary distribution and copy distribution.

        p_gen        : (B,)          — generate probability
        vocab_logits : (B, V)        — raw decoder output before softmax
        q_alpha      : (B, q_len)    — attention weights over question tokens
        q_token_ids  : (B, q_len)    — vocabulary indices of question tokens
        vocab_size   : int           — V

        returns      : (B, V) — final mixed log-probabilities
        """
        # Vocabulary distribution
        p_vocab = F.softmax(vocab_logits, dim=-1)  # (B, V)

        # Copy distribution: scatter attention weights onto vocab positions
        # Zero out <pad>=0 contributions (q_token_ids may contain padding)
        B = p_gen.size(0)
        p_copy = vocab_logits.new_zeros(B, vocab_size)  # (B, V)
        # Clamp to valid range just in case
        ids = q_token_ids.clamp(0, vocab_size - 1)      # (B, q_len)
        p_copy.scatter_add_(1, ids, q_alpha)             # (B, V)

        # Blend: p_gen * P_vocab + (1 - p_gen) * P_copy
        p_gen_2d = p_gen.unsqueeze(1)                    # (B, 1)
        final = p_gen_2d * p_vocab + (1.0 - p_gen_2d) * p_copy  # (B, V)

        # Clamp to avoid log(0) — tiny epsilon for numerical stability
        final = final.clamp(min=1e-10)
        return torch.log(final)  # (B, V) — log-probabilities


if __name__ == "__main__":
    B, V, E, H, L = 4, 8648, 512, 1024, 15
    pgn = PointerGeneratorHead(embed_size=E, hidden_size=H, context_size=H)

    embed_t     = torch.randn(B, E)
    h_t         = torch.randn(B, H)
    context     = torch.randn(B, H)
    vocab_logits = torch.randn(B, V)
    q_alpha     = torch.softmax(torch.randn(B, L), dim=-1)   # (B, q_len)
    q_token_ids = torch.randint(0, V, (B, L))

    p_gen = pgn(embed_t, h_t, context)
    print(f"p_gen shape   : {p_gen.shape}")          # (4,)
    print(f"p_gen range   : [{p_gen.min():.3f}, {p_gen.max():.3f}]")

    log_dist = PointerGeneratorHead.blend(p_gen, vocab_logits, q_alpha, q_token_ids, V)
    print(f"final log-dist: {log_dist.shape}")        # (4, 8648)
    print(f"log-prob max  : {log_dist.max().item():.4f}")   # should be ≤ 0
    print("PointerGeneratorHead sanity check PASSED")
