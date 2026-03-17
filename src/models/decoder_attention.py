"""
LSTM Decoder with Multi-Head Cross-Attention (MHCA)
====================================================

ARCHITECTURAL EVOLUTION
-----------------------
v1: BahdanauAttention (additive, single-head)
    → problem: limited expressiveness, only image-side attention
v2: DenseCoAttention  (intra-modal self-attention + cross-attention)
    → problem: self-attention inside decode_step runs T times
      → O(T·(L_v² + L_q²)) complexity; violates no-self-attention constraint
v3: MultiHeadCrossAttention (this file)
    → pure cross-attention: Q = h_t (LSTM hidden state), K/V = memory
    → O(T·S) where S = num_regions or q_len — LINEAR in sequence length
    → NO intra-modal self-attention → constraint compliant

DECODE STEP (v3)
----------------
At step t:
  1. img_context, img_alpha = img_mhca(h_t, img_features)        # (B, H), (B, 49)
  2. q_context,   q_alpha   = q_mhca(h_t, q_hidden_states)       # (B, H), (B, q_len)
  3. lstm_input = cat([embed_t, img_context, q_context], dim=-1)  # (B, E+2H)
  4. h_{t+1}, c_{t+1} = LSTM(lstm_input, h_t, c_t)
  5. logit = fc(out_proj(h_{t+1}))                                 # (B, V)
  6. [optional] logit = PGN.blend(p_gen, logit, q_alpha, q_token_ids)

The q_alpha produced by q_mhca IS the correct distribution for PGN's P_copy:
it tells the decoder exactly where it is "looking" in the question at step t.

COVERAGE (image side only)
--------------------------
When use_coverage=True, img_mhca adds a learned coverage bias to attention
scores before softmax:
    bias = cov_scale * coverage   (learned scalar cov_scale, init 0)
This discourages the model from repeatedly attending to the same image region.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from decoder_lstm import LayerNormLSTMStack, WeightDrop
from pointer_generator import PointerGeneratorHead


# ── Legacy class kept for checkpoint compatibility (no longer used in decoder) ─
class BahdanauAttention(nn.Module):
    """Legacy Bahdanau additive attention. Superseded by MultiHeadCrossAttention."""
    def __init__(self, hidden_size, attn_dim=512, use_coverage=False):
        super().__init__()
        self.use_coverage = use_coverage
        self.W_h   = nn.Linear(hidden_size, attn_dim)
        self.W_img = nn.Linear(hidden_size, attn_dim)
        if use_coverage:
            self.W_cov = nn.Linear(1, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, hidden, img_features, coverage=None):
        h_proj   = self.W_h(hidden).unsqueeze(1)
        img_proj = self.W_img(img_features)
        energy   = h_proj + img_proj
        if self.use_coverage and coverage is not None:
            energy = energy + self.W_cov(coverage.unsqueeze(-1))
        scores  = self.v(torch.tanh(energy)).squeeze(-1)
        alpha   = F.softmax(scores, dim=1)
        context = (alpha.unsqueeze(2) * img_features).sum(dim=1)
        return context, alpha


# ── v3 Attention: Multi-Head Cross-Attention ────────────────────────────────────

class MultiHeadCrossAttention(nn.Module):
    """
    Pure Multi-Head Cross-Attention.

    Query  : h_t   (B, H)    — LSTM hidden state (single timestep)
    Key/Val: memory (B, S, H) — image regions or question token states

    No intra-modal self-attention.  Q and K/V always come from different
    modalities (LSTM vs. CNN features; LSTM vs. question encoder states).

    Optional coverage bias (image side only):
      scores += cov_scale * coverage   where cov_scale is a learned scalar.
      coverage (B, S): cumulative attention from previous decode steps.
      This discourages re-attending to the same region repeatedly.

    Args:
        hidden_size  : H — must be divisible by num_heads
        num_heads    : number of attention heads (default 4)
        use_coverage : whether to add coverage bias to scores (image side only)
    """

    def __init__(self, hidden_size: int, num_heads: int = 4,
                 use_coverage: bool = False):
        super().__init__()
        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        self.hidden_size  = hidden_size
        self.num_heads    = num_heads
        self.d_head       = hidden_size // num_heads
        self.scale        = self.d_head ** -0.5
        self.use_coverage = use_coverage

        # Q from LSTM hidden state; K/V from memory
        self.Q_proj   = nn.Linear(hidden_size, hidden_size, bias=False)
        self.K_proj   = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V_proj   = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Coverage: a single learnable scalar that scales the coverage signal.
        # Init at 0 so training starts from unconstrained attention.
        if use_coverage:
            self.cov_scale = nn.Parameter(torch.zeros(1))

    def forward(self, query: torch.Tensor, memory: torch.Tensor,
                coverage: torch.Tensor = None):
        """
        Args:
            query    : (B, H)    — LSTM h_t
            memory   : (B, S, H) — image regions (49) or question tokens
            coverage : (B, S) or None — cumulative attention (image side only)

        Returns:
            context  : (B, H)    — attended summary of memory
            alpha    : (B, S)    — attention weights averaged over heads
        """
        B, S, H = memory.shape

        # Project Q from (B, H) to (B, 1, H) then to heads
        q = self.Q_proj(query).unsqueeze(1)             # (B, 1, H)
        k = self.K_proj(memory)                          # (B, S, H)
        v = self.V_proj(memory)                          # (B, S, H)

        # Reshape to multi-head format: (B, nh, seq, d_head)
        q = q.view(B, 1, self.num_heads, self.d_head).transpose(1, 2)   # (B, nh, 1, d)
        k = k.view(B, S, self.num_heads, self.d_head).transpose(1, 2)   # (B, nh, S, d)
        v = v.view(B, S, self.num_heads, self.d_head).transpose(1, 2)   # (B, nh, S, d)

        # Scaled dot-product attention: Q·Kᵀ / sqrt(d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale       # (B, nh, 1, S)

        # Optional coverage bias: penalize already-attended positions
        if self.use_coverage and coverage is not None:
            # coverage: (B, S) → (B, 1, 1, S) to broadcast over heads and query dim
            scores = scores + self.cov_scale * coverage.unsqueeze(1).unsqueeze(2)

        alpha = F.softmax(scores, dim=-1)                                 # (B, nh, 1, S)

        # Weighted sum of values
        context = torch.matmul(alpha, v)                                  # (B, nh, 1, d)
        context = context.squeeze(2)                                      # (B, nh, d)
        context = context.transpose(1, 2).contiguous().view(B, H)        # (B, H)
        context = self.out_proj(context)                                  # (B, H)

        # Average attention weights over heads for visualization / coverage / PGN
        alpha_mean = alpha.mean(dim=1).squeeze(1)                        # (B, S)

        return context, alpha_mean


# ── Decoder ────────────────────────────────────────────────────────────────────

class LSTMDecoderWithAttention(nn.Module):
    """
    LSTM Decoder with dual Multi-Head Cross-Attention (image + question).

    Two MHCA modules attend SEPARATELY to image regions and question tokens
    at each decode step — pure cross-attention, no self-attention stacks.

    Architecture per step t:
      img_context, img_alpha = img_mhca(h_t, img_features)   O(S_img)
      q_context,   q_alpha   = q_mhca  (h_t, q_hidden)       O(S_q)
      lstm_input = cat([embed_t, img_context, q_context])     (B, E+2H)
      h_{t+1} = LSTM(lstm_input, h_t)
      logit   = fc(out_proj(h_{t+1}))

    Coverage (image side):
      Tracks cumulative img_alpha across steps.
      Adds learned scalar bias to img_mhca scores to discourage re-attendance.

    Pointer-Generator (Tier 5):
      q_alpha is the natural copy distribution for PGN: it tells the decoder
      exactly where it is attending in the question at step t.

    Backward-compatible flags:
      use_dcan — accepted but ignored (MHCA is always used; DCAN removed)
    """

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers,
                 attn_dim=512, dropout=0.5, pretrained_embeddings=None,
                 use_coverage=False, use_layer_norm=False, use_dropconnect=False,
                 use_dcan=False, num_heads=4, use_pgn=False):
        """
        use_layer_norm  : Tier 1A+1C — LayerNormLSTMStack with highway connections
        use_dropconnect : Tier 1B    — DropConnect on hidden-to-hidden weights
        use_dcan        : IGNORED    — kept for backward compatibility only.
                          MHCA is always used (supersedes both Bahdanau and DCAN).
        num_heads       : MHCA heads (default 4, hidden_size must be divisible)
        use_pgn         : Tier 5 — Pointer-Generator Network (copy from question)
        """
        super().__init__()

        self.hidden_size    = hidden_size
        self.num_layers     = num_layers
        self.use_coverage   = use_coverage
        self.use_layer_norm = use_layer_norm
        self.use_pgn        = use_pgn
        self.vocab_size     = vocab_size

        # ── Embedding ──────────────────────────────────────────────────────
        if pretrained_embeddings is not None:
            glove_dim = pretrained_embeddings.shape[1]
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=False, padding_idx=0)
            self.embed_proj = nn.Linear(glove_dim, embed_size) \
                              if glove_dim != embed_size else None
        else:
            self.embedding  = nn.Embedding(vocab_size, embed_size, padding_idx=0)
            self.embed_proj = None

        # ── Dual MHCA (image + question) ──────────────────────────────────
        # img_mhca: optionally uses coverage bias
        # q_mhca:   no coverage (question is fully accessible at all steps)
        self.img_mhca = MultiHeadCrossAttention(
            hidden_size, num_heads=num_heads, use_coverage=use_coverage)
        self.q_mhca   = MultiHeadCrossAttention(
            hidden_size, num_heads=num_heads, use_coverage=False)

        # ── LSTM ────────────────────────────────────────────────────────────
        # Input = embed_t + img_context + q_context
        lstm_input_size = embed_size + hidden_size * 2
        if use_layer_norm:
            self.lstm = LayerNormLSTMStack(
                input_size=lstm_input_size, hidden_size=hidden_size,
                num_layers=num_layers, dropout=dropout, use_highway=True)
        else:
            _lstm = nn.LSTM(
                input_size=lstm_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            if use_dropconnect and num_layers >= 1:
                self.lstm = WeightDrop(_lstm, ['weight_hh_l0'], dropout=0.3)
            else:
                self.lstm = _lstm

        # ── Output projection + weight tying ───────────────────────────────
        if pretrained_embeddings is not None:
            self.out_proj = nn.Linear(hidden_size, embed_size)
            self.fc       = nn.Linear(embed_size, vocab_size, bias=False)
        else:
            actual_embed  = self.embedding.embedding_dim
            self.out_proj = nn.Linear(hidden_size, actual_embed)
            self.fc       = nn.Linear(actual_embed, vocab_size, bias=False)
            self.fc.weight = self.embedding.weight   # weight tying

        self.dropout = nn.Dropout(dropout)

        # ── Tier 5: Pointer-Generator Head ─────────────────────────────────
        if use_pgn:
            actual_embed = self.embedding.embedding_dim \
                           if pretrained_embeddings is None else embed_size
            self.pgn = PointerGeneratorHead(
                embed_size=actual_embed,
                hidden_size=hidden_size,
                context_size=hidden_size,   # img_context dim
            )

    # ── Teacher-forcing forward (training) ─────────────────────────────────────

    def forward(self, encoder_hidden, img_features, q_hidden_states, target_seq,
                q_token_ids=None):
        """
        Training mode — Teacher Forcing with dual MHCA.

        Args:
            encoder_hidden   : (num_layers, B, H) — initial (h, c) from fusion
            img_features     : (B, 49, H)         — spatial CNN output
            q_hidden_states  : (B, q_len, H)       — question encoder states
            target_seq       : (B, max_len)        — [<start>, w1, w2, ...]
            q_token_ids      : (B, q_len) or None  — required for PGN

        Returns:
            logits        : (B, max_len, vocab_size)
            coverage_loss : scalar tensor (0.0 if coverage disabled)
        """
        batch   = target_seq.size(0)
        max_len = target_seq.size(1)

        embeds = self.dropout(self.embedding(target_seq))    # (B, max_len, E)
        if self.embed_proj is not None:
            embeds = self.embed_proj(embeds)

        hidden      = encoder_hidden                          # tuple (h, c)
        num_regions = img_features.size(1)                   # 49

        # Coverage vector: (B, S) — cumulative img attention across steps
        coverage = img_features.new_zeros(batch, num_regions) \
                   if self.use_coverage else None
        cov_loss = img_features.new_zeros(1)

        logits_list = []

        for t in range(max_len):
            embed_t = embeds[:, t, :]                         # (B, E)
            h_top   = hidden[0][-1]                           # (B, H) — last layer

            # Image cross-attention (with optional coverage bias)
            img_context, img_alpha = self.img_mhca(h_top, img_features, coverage)

            # Question cross-attention (q_alpha → PGN copy distribution)
            q_context, q_alpha = self.q_mhca(h_top, q_hidden_states)

            # Coverage: accumulate + compute penalty
            if self.use_coverage:
                # Penalize re-attending to already-attended regions
                # (same formulation as v1: Σ_t min(α_t, coverage_t))
                cov_loss = cov_loss + torch.min(img_alpha, coverage).sum(dim=1).mean()
                coverage = coverage + img_alpha

            # LSTM step: input = concat(embed_t, img_context, q_context)
            lstm_input = torch.cat([embed_t, img_context, q_context], dim=1)
            output, hidden = self.lstm(lstm_input.unsqueeze(1), hidden)

            vocab_logit = self.fc(self.out_proj(output.squeeze(1)))  # (B, V)

            # Pointer-Generator: blend vocab and copy distributions
            if self.use_pgn and q_token_ids is not None:
                p_gen = self.pgn(embed_t, h_top, img_context)        # (B,)
                logit = PointerGeneratorHead.blend(
                    p_gen, vocab_logit, q_alpha, q_token_ids, self.vocab_size)
            else:
                logit = vocab_logit

            logits_list.append(logit)

        logits       = torch.stack(logits_list, dim=1)               # (B, max_len, V)
        coverage_loss = cov_loss / max_len if self.use_coverage else cov_loss.squeeze()
        return logits, coverage_loss

    # ── Autoregressive decode step (inference) ─────────────────────────────────

    def decode_step(self, token, hidden, img_features, q_hidden_states,
                    coverage=None, q_token_ids=None):
        """
        Inference mode — one autoregressive step.

        Args:
            token           : (B, 1)           — current token id
            hidden          : (h, c) each (num_layers, B, H)
            img_features    : (B, 49, H)
            q_hidden_states : (B, q_len, H)
            coverage        : (B, 49) or None  — cumulative image attention
            q_token_ids     : (B, q_len) or None — required for PGN

        Returns:
            logit    : (B, V)
            hidden   : updated (h, c)
            img_alpha: (B, 49)  — for visualization
            coverage : (B, 49) or None — updated cumulative attention
        """
        embed = self.dropout(self.embedding(token))   # (B, 1, E)
        if self.embed_proj is not None:
            embed = self.embed_proj(embed)
        embed_1d = embed.squeeze(1)                   # (B, E)
        h_top    = hidden[0][-1]                      # (B, H)

        img_context, img_alpha = self.img_mhca(h_top, img_features, coverage)
        q_context,   q_alpha   = self.q_mhca(h_top, q_hidden_states)

        # Update coverage
        if self.use_coverage:
            if coverage is None:
                coverage = torch.zeros_like(img_alpha)
            coverage = coverage + img_alpha

        lstm_input = torch.cat([embed_1d, img_context, q_context], dim=1).unsqueeze(1)
        output, hidden = self.lstm(lstm_input, hidden)
        vocab_logit = self.fc(self.out_proj(output.squeeze(1)))

        if self.use_pgn and q_token_ids is not None:
            p_gen = self.pgn(embed_1d, h_top, img_context)
            logit = PointerGeneratorHead.blend(
                p_gen, vocab_logit, q_alpha, q_token_ids, self.vocab_size)
        else:
            logit = vocab_logit

        return logit, hidden, img_alpha, coverage


# ── Quick sanity test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch

    B, V, E, H, L, T, S, Q = 4, 3000, 512, 1024, 2, 10, 49, 15

    dec = LSTMDecoderWithAttention(vocab_size=V, embed_size=E, hidden_size=H,
                                   num_layers=L, use_coverage=True, num_heads=4)

    h0 = torch.zeros(L, B, H)
    c0 = torch.zeros(L, B, H)
    img = torch.randn(B, S, H)
    qh  = torch.randn(B, Q, H)
    tgt = torch.randint(0, V, (B, T))

    logits, cov = dec((h0, c0), img, qh, tgt)
    print(f"logits : {logits.shape}")          # (4, 10, 3000)
    print(f"cov    : {cov.item():.4f}")

    tok = torch.randint(0, V, (B, 1))
    logit, hid, alpha, cov2 = dec.decode_step(tok, (h0, c0), img, qh)
    print(f"step logit : {logit.shape}")        # (4, 3000)
    print(f"img_alpha  : {alpha.shape}")        # (4, 49)
    print(f"coverage   : {cov2.shape}")         # (4, 49)
    print("MHCA decoder test PASSED")
