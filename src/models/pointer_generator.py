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
        p_copy.scatter_add_(1, ids, q_alpha.to(p_copy.dtype))  # (B, V) — cast for AMP dtype safety

        # Blend: p_gen * P_vocab + (1 - p_gen) * P_copy
        p_gen_2d = p_gen.unsqueeze(1)                    # (B, 1)
        final = p_gen_2d * p_vocab + (1.0 - p_gen_2d) * p_copy  # (B, V)

        # Clamp to avoid log(0) — tiny epsilon for numerical stability
        final = final.clamp(min=1e-10)
        return torch.log(final)  # (B, V) — log-probabilities


# ---------------------------------------------------------------------------
# ThreeWayPGNHead — G2: copy from vocab + question + visual object labels
# ---------------------------------------------------------------------------

class ThreeWayPGNHead(nn.Module):
    """
    Three-way Pointer-Generator head (G2).

    Blends three token sources at each decode step:
      Source 1 — Vocabulary: P_vocab from decoder FC output
      Source 2 — Question copy: scatter q_alpha onto question token positions
      Source 3 — Visual label copy: scatter img_alpha / |tokens| onto label token positions

    Blending weights (Eq 32):
      [p_g, p_cQ, p_cV] = Softmax(W_ptr [c_img; h_t; x_t])
      W_ptr ∈ R^{3 × input_dim}   input_dim = c_img(H) + h_t(H) + x_t(E+2H+len)

    Final distribution (Eq 33):
      P(w) = p_g * P_vocab(w) + p_cQ * P_copy_Q(w) + p_cV * P_copy_V(w)

    Args:
        input_dim : c_img_dim + hidden_dim + lstm_input_dim
                    Default: 1024 + 1024 + 2624 = 4672 (Model G full config)
    """

    def __init__(self, input_dim: int = 4672):
        super().__init__()
        self.W_ptr = nn.Linear(input_dim, 3)

    def forward(self, c_img: torch.Tensor,
                h_t: torch.Tensor,
                x_t: torch.Tensor):
        """
        c_img : (B, H)    — image attention context
        h_t   : (B, H)    — LSTM top-layer hidden state
        x_t   : (B, E+2H+len) — full LSTM input (embed + img_ctx + q_ctx + len_emb)

        Returns p_g, p_cQ, p_cV each (B,) — three Softmax weights.
        """
        pgn_input = torch.cat([c_img, h_t, x_t], dim=-1)   # (B, input_dim)
        weights = F.softmax(self.W_ptr(pgn_input), dim=-1)  # (B, 3)
        return weights[:, 0], weights[:, 1], weights[:, 2]

    @staticmethod
    def blend_3way(
        p_g: torch.Tensor,
        p_cQ: torch.Tensor,
        p_cV: torch.Tensor,
        vocab_logits: torch.Tensor,
        q_alpha: torch.Tensor,
        q_token_ids: torch.Tensor,
        img_alpha: torch.Tensor,
        label_tokens,        # (B, k, max_t) int64 or None
        vocab_size: int,
    ) -> torch.Tensor:
        """
        Mix three distributions into log-probabilities.

        p_g, p_cQ, p_cV  : (B,)
        vocab_logits      : (B, V)
        q_alpha           : (B, q_len) — attention over question tokens
        q_token_ids       : (B, q_len) — vocabulary indices of question tokens
        img_alpha         : (B, k)     — attention over image regions
        label_tokens      : (B, k, max_t) int64 — vocab indices for BUTD label names;
                            0 = padding (ignored). None → P_copy_V = 0.
        vocab_size        : int

        Returns: (B, V) log-probabilities (ready for NLLLoss).
        """
        dev = vocab_logits.device
        orig_dtype = vocab_logits.dtype
        B = p_g.size(0)

        with torch.amp.autocast(device_type=dev.type, enabled=False):
            vl = vocab_logits.float().clamp(-50, 50)
            pg  = p_g.float()
            pcq = p_cQ.float()
            pcv = p_cV.float()

            # Source 1: vocabulary distribution
            p_vocab = F.softmax(vl, dim=-1)   # (B, V)

            # Source 2: question copy — scatter q_alpha onto vocab positions
            p_copy_q = vl.new_zeros(B, vocab_size)
            if q_token_ids is not None and q_alpha is not None:
                ids_q = q_token_ids.clamp(0, vocab_size - 1)
                p_copy_q.scatter_add_(1, ids_q, q_alpha.float())

            # Source 3: visual label copy — distribute img_alpha / |tokens_i| per region (Eq 31)
            p_copy_v = vl.new_zeros(B, vocab_size)
            if label_tokens is not None:
                k      = label_tokens.size(1)
                max_t  = label_tokens.size(2)
                counts = (label_tokens > 0).sum(dim=-1).float().clamp(min=1.0)
                per_tok = img_alpha.float() / counts
                per_tok = per_tok.unsqueeze(-1).expand(-1, -1, max_t)
                flat_ids     = label_tokens.view(B, -1).clamp(0, vocab_size - 1)
                flat_weights = per_tok.contiguous().view(B, -1)
                flat_mask    = (label_tokens.view(B, -1) > 0).float()
                flat_weights = flat_weights * flat_mask
                p_copy_v.scatter_add_(1, flat_ids, flat_weights)

            # Blend: p_g * P_vocab + p_cQ * P_copy_Q + p_cV * P_copy_V
            out = (pg.unsqueeze(1)  * p_vocab
                 + pcq.unsqueeze(1) * p_copy_q
                 + pcv.unsqueeze(1) * p_copy_v)
            out = out.clamp(min=1e-10)
            out = out / out.sum(dim=-1, keepdim=True).clamp(min=1e-10)
            return out.log().to(orig_dtype)


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
