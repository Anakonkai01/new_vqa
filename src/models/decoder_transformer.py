"""Pre-norm Transformer Decoder — used by Model H.

Upgrade over Model G (LSTM + MHA)
-----------------------------------
The LSTM decoder compresses all past token context into a single hidden vector
``h_{t-1}`` of size 1024. For VQA-E explanations of 15–25 tokens, this vector
must simultaneously encode token history, grammatical state, and semantic intent —
a hard compression with no direct gradient path between distant tokens.

The Transformer decoder eliminates this by keeping all previous token
representations explicit::

    # Step t attends directly over [y_1, ..., y_{t-1}] — O(1) gradient path.
    self_attn_t  = MaskedMHA(y_t, [y_1,…,y_{t-1}])
    cross_img_t  = MHA(y_t, img_patches, img_patches)   # 49 visual regions
    cross_q_t    = MHA(y_t, q_tokens, q_tokens)         # 77 language tokens
    y_t_out      = FFN(LayerNorm(self_attn_t + cross_img_t + cross_q_t))

Training (teacher forcing): all T positions are processed in parallel via causal
mask — O(T) in sequence length, O(1) in gradient path depth.

Inference: autoregressive KV-cache.  The accumulated token embedding buffer
``(B, t, d_model)`` serves as the KV cache — no separate key/value projection
caching needed here since we accumulate full embeddings.

Pre-norm style
--------------
LayerNorm is applied *before* each attention / FFN sub-layer (not after).
This stabilises gradients at depth — the residual stream maintains unit variance
regardless of layer count, avoiding the warm-up sensitivity of Post-norm.

SCST compatibility
------------------
``sample()`` signature matches ``LSTMDecoderWithAttention`` and
``LSTMDecoderWithMHA``.  The ``encoder_hidden`` argument is ignored (no LSTM
state needed); state is maintained via the accumulated token embedding buffer.

Shape contracts::

    # forward (teacher forcing)
    img_features    : (B, 49, H)
    q_hidden_states : (B, 77, H)
    target_seq      : (B, T)     — [<start>, w_1, ..., w_{T-1}]
    logits          : (B, T, vocab_size)

    # decode_step (autoregressive)
    token      : (B, 1)
    kv_cache   : (B, t, H) or None
    returns    : logit (B, vocab_size), new_kv_cache (B, t+1, H), attn_weights, None
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── Positional Encoding ────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Args:
        d_model: Embedding dimension.
        max_len: Maximum sequence length supported.
        dropout: Dropout applied after adding positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)            # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )                                                          # (d_model/2,)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)                            # (max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to ``x``.

        Args:
            x: ``FloatTensor (B, T, d_model)``

        Returns:
            ``FloatTensor (B, T, d_model)``
        """
        seq_len = x.size(1)
        return self.dropout(x + self.pe[:seq_len])   # type: ignore[index]


# ── Single Transformer Decoder Layer ──────────────────────────────────────────

class _TransformerDecoderLayer(nn.Module):
    """Pre-norm Transformer decoder layer with dual cross-attention.

    Sub-layer order (pre-norm)::

        x = x + MaskedMHA(LN(x), LN(x), causal_mask)    # 1. masked self-attention
        x = x + MHA(LN(x), img_features, img_features)  # 2. cross-attention: image
        x = x + MHA(LN(x), q_tokens, q_tokens)          # 3. cross-attention: question
        x = x + FFN(LN(x))                              # 4. position-wise FFN

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension (default 4 × d_model).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # 1. Masked self-attention
        self.self_attn    = nn.MultiheadAttention(d_model, num_heads, dropout=0.0, batch_first=True)
        self.norm1        = nn.LayerNorm(d_model)

        # 2. Cross-attention over image patches
        self.img_cross    = nn.MultiheadAttention(d_model, num_heads, dropout=0.0, batch_first=True)
        self.norm2        = nn.LayerNorm(d_model)

        # 3. Cross-attention over question tokens
        self.q_cross      = nn.MultiheadAttention(d_model, num_heads, dropout=0.0, batch_first=True)
        self.norm3        = nn.LayerNorm(d_model)

        # 4. Position-wise FFN: Linear(d_model, d_ff) → GELU → Linear(d_ff, d_model)
        self.ffn          = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm4        = nn.LayerNorm(d_model)
        self.dropout      = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        img_features: Tensor,
        q_hidden_states: Tensor,
        causal_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass for one layer.

        Args:
            x: ``FloatTensor (B, T, d_model)`` — current token representations.
            img_features: ``FloatTensor (B, 49, d_model)``.
            q_hidden_states: ``FloatTensor (B, 77, d_model)``.
            causal_mask: ``FloatTensor (T, T)`` — additive causal mask
                (0 at valid positions, -inf at future positions).

        Returns:
            Tuple of:
                x         – ``FloatTensor (B, T, d_model)``
                img_alpha – ``FloatTensor (B, T, 49)`` — averaged cross-attn weights.
        """
        # 1. Masked self-attention (pre-norm)
        x_norm = self.norm1(x)
        sa_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=causal_mask)
        x = x + self.dropout(sa_out)

        # 2. Cross-attention over image patches (pre-norm)
        x_norm = self.norm2(x)
        img_out, img_alpha = self.img_cross(x_norm, img_features, img_features)
        # img_alpha: (B, T, 49) averaged over heads
        x = x + self.dropout(img_out)

        # 3. Cross-attention over question tokens (pre-norm)
        x_norm = self.norm3(x)
        q_out, _ = self.q_cross(x_norm, q_hidden_states, q_hidden_states)
        x = x + self.dropout(q_out)

        # 4. FFN (pre-norm)
        x_norm = self.norm4(x)
        x = x + self.dropout(self.ffn(x_norm))

        return x, img_alpha


# ── TransformerDecoder ────────────────────────────────────────────────────────

class TransformerDecoder(nn.Module):
    """Pre-norm Transformer decoder with dual cross-attention.

    Used by **Model H**.

    Architecture::

        embed → pos_enc → N × _TransformerDecoderLayer → final_norm → fc

    Training: parallel processing of all T positions via causal mask.
    Inference: token-by-token with KV cache (accumulated embedding buffer).

    Args:
        vocab_size: Answer vocabulary size.
        embed_size: Token embedding dimension (= d_model).
        hidden_size: Model dimension d_model (same as embed_size in this design).
        num_layers: Number of decoder layers ``N`` (default 4).
        num_heads: Number of attention heads (default 8).
        d_ff: FFN hidden size (default 4 × hidden_size).
        dropout: Dropout probability.
        max_len: Maximum sequence length for positional encoding.
        pretrained_embeddings: Optional GloVe matrix ``(vocab_size, glove_dim)``.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        max_len: int = 512,
        pretrained_embeddings: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        d_ff = d_ff or hidden_size * 4

        # ── Embedding ──────────────────────────────────────────────────────────
        if pretrained_embeddings is not None:
            glove_dim = pretrained_embeddings.shape[1]
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=False, padding_idx=0
            )
            self.embed_proj: Optional[nn.Linear] = (
                nn.Linear(glove_dim, hidden_size) if glove_dim != hidden_size else None
            )
        else:
            self.embedding  = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
            self.embed_proj = None

        self.pos_enc = SinusoidalPositionalEncoding(hidden_size, max_len=max_len, dropout=dropout)

        # ── Decoder layers ─────────────────────────────────────────────────────
        self.layers = nn.ModuleList([
            _TransformerDecoderLayer(hidden_size, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_size)

        # ── Output projection ──────────────────────────────────────────────────
        self.fc = nn.Linear(hidden_size, vocab_size, bias=False)
        # Weight tie fc ↔ embedding (when not using GloVe path)
        if pretrained_embeddings is None:
            self.fc.weight = self.embedding.weight

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _make_causal_mask(seq_len: int, device: torch.device) -> Tensor:
        """Build a ``(T, T)`` additive causal mask.

        ``mask[i, j] = 0`` if ``j <= i`` (position i can attend to j),
        ``mask[i, j] = -inf`` if ``j > i`` (future position, masked out).
        """
        mask = torch.zeros(seq_len, seq_len, device=device)
        mask.fill_(float("-inf"))
        mask.triu_(diagonal=1)   # upper-triangular (excluding diagonal) = -inf
        return mask

    # ── Forward (teacher forcing) ──────────────────────────────────────────────

    def forward(
        self,
        encoder_hidden: Tuple[Tensor, Tensor],   # ignored — API compat with LSTM decoders
        img_features: Tensor,
        q_hidden_states: Tensor,
        target_seq: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Teacher-forcing forward pass (all positions in parallel).

        Args:
            encoder_hidden: Ignored.  Accepted for API compatibility with
                ``LSTMDecoderWithAttention`` / ``LSTMDecoderWithMHA``.
            img_features: ``FloatTensor (B, 49, H)``.
            q_hidden_states: ``FloatTensor (B, 77, H)``.
            target_seq: ``LongTensor (B, T)`` — ``[<start>, w_1, …, w_{T-1}]``.

        Returns:
            Tuple of:
                logits        – ``FloatTensor (B, T, vocab_size)``
                coverage_loss – ``torch.zeros(1)`` (API compatibility)
        """
        T      = target_seq.size(1)
        device = target_seq.device

        embeds = self.embedding(target_seq)         # (B, T, H or glove_dim)
        if self.embed_proj is not None:
            embeds = self.embed_proj(embeds)        # (B, T, H)
        embeds = self.pos_enc(embeds)               # (B, T, H) + positional

        causal_mask = self._make_causal_mask(T, device)   # (T, T)

        x = embeds
        for layer in self.layers:
            x, _ = layer(x, img_features, q_hidden_states, causal_mask)

        x = self.final_norm(x)                      # (B, T, H)
        logits = self.fc(x)                         # (B, T, vocab_size)

        coverage_loss = img_features.new_zeros(1).squeeze()
        return logits, coverage_loss

    # ── Decode step (autoregressive inference) ─────────────────────────────────

    def decode_step(
        self,
        token: Tensor,
        hidden: Optional[Tensor],           # kv_cache: (B, t, H) or None
        img_features: Tensor,
        q_hidden_states: Tensor,
        coverage: Optional[Tensor] = None,  # unused, API compat
    ) -> Tuple[Tensor, Tensor, Tensor, None]:
        """Single autoregressive decode step using token accumulation as KV cache.

        At step t, ``hidden`` is the accumulated embedding buffer ``(B, t, H)``
        representing tokens ``[y_0, y_1, ..., y_{t-1}]``.

        Args:
            token: ``LongTensor (B, 1)`` — current token.
            hidden: ``FloatTensor (B, t, H)`` accumulated embeddings, or ``None``
                at the first step.
            img_features: ``FloatTensor (B, 49, H)``.
            q_hidden_states: ``FloatTensor (B, 77, H)``.
            coverage: Ignored (API compatibility).

        Returns:
            Tuple of:
                logit     – ``FloatTensor (B, vocab_size)``
                kv_cache  – ``FloatTensor (B, t+1, H)`` — updated accumulation.
                img_alpha – ``FloatTensor (B, 49)`` — image cross-attn weights
                            at the last position (for visualisation).
                None      – coverage placeholder.
        """
        device = token.device

        # Embed current token.
        embed = self.embedding(token)             # (B, 1, H or glove_dim)
        if self.embed_proj is not None:
            embed = self.embed_proj(embed)        # (B, 1, H)

        # Append to KV cache.
        if hidden is None:
            buf = embed                           # (B, 1, H)
        else:
            buf = torch.cat([hidden, embed], dim=1)  # (B, t+1, H)

        # Add positional encoding to the full buffer.
        x = self.pos_enc(buf)                    # (B, t+1, H)

        t_plus1 = x.size(1)
        causal_mask = self._make_causal_mask(t_plus1, device)   # (t+1, t+1)

        last_img_alpha: Tensor = img_features.new_zeros(img_features.size(0), img_features.size(1))

        for layer in self.layers:
            x, img_alpha = layer(x, img_features, q_hidden_states, causal_mask)
            last_img_alpha = img_alpha[:, -1, :]   # (B, 49) — last position's attention

        x = self.final_norm(x)                   # (B, t+1, H)
        logit = self.fc(x[:, -1, :])             # (B, vocab_size) — last position

        return logit, buf, last_img_alpha, None

    # ── Sample (SCST) ──────────────────────────────────────────────────────────

    def sample(
        self,
        encoder_hidden: Tuple[Tensor, Tensor],  # ignored
        img_features: Tensor,
        q_hidden_states: Tensor,
        max_len: int,
        start_idx: int,
        end_idx: int,
        method: str = "greedy",
    ) -> Tuple[Tensor, Tensor]:
        """Generate sequences and log-probabilities for SCST training.

        Args:
            encoder_hidden: Ignored (no LSTM state in Transformer decoder).
            img_features: ``FloatTensor (B, 49, H)``.
            q_hidden_states: ``FloatTensor (B, 77, H)``.
            max_len: Maximum number of tokens to generate.
            start_idx: Index of ``<start>`` token.
            end_idx: Index of ``<end>`` token.
            method: ``'greedy'`` or ``'sample'``.

        Returns:
            Tuple of:
                seqs      – ``LongTensor  (B, max_len)``
                log_probs – ``FloatTensor (B, max_len)``
        """
        batch  = img_features.size(0)
        device = img_features.device

        token      = torch.full((batch, 1), start_idx, dtype=torch.long, device=device)
        unfinished = torch.ones(batch, dtype=torch.bool, device=device)
        kv_cache: Optional[Tensor] = None

        seqs:      list[Tensor] = []
        log_probs: list[Tensor] = []

        for _ in range(max_len):
            logit, kv_cache, _, _ = self.decode_step(
                token, kv_cache, img_features, q_hidden_states
            )   # logit: (B, vocab_size)

            if method == "sample":
                predicted_token = torch.multinomial(
                    F.softmax(logit, dim=-1), num_samples=1
                )                                                           # (B, 1)
            else:
                predicted_token = torch.argmax(logit, dim=-1, keepdim=True)   # (B, 1)

            step_log_probs     = F.log_softmax(logit, dim=-1)
            selected_log_probs = step_log_probs.gather(1, predicted_token)   # (B, 1)

            seqs.append(predicted_token)
            log_probs.append(selected_log_probs)

            token      = predicted_token * unfinished.unsqueeze(1).long()
            unfinished = unfinished & (predicted_token.squeeze(1) != end_idx)

        seqs      = torch.cat(seqs,      dim=1)   # (B, max_len)
        log_probs = torch.cat(log_probs, dim=1)   # (B, max_len)
        return seqs, log_probs


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, V, H, T, N, Q = 4, 3000, 1024, 12, 49, 77

    decoder = TransformerDecoder(
        vocab_size=V, embed_size=H, hidden_size=H, num_layers=4, num_heads=8
    )

    # Teacher-forcing forward
    img     = torch.randn(B, N, H)
    q_hid   = torch.randn(B, Q, H)
    target  = torch.randint(0, V, (B, T))
    h_dummy = (torch.zeros(1), torch.zeros(1))

    logits, cov = decoder(h_dummy, img, q_hid, target)
    print(f"logits      : {logits.shape}")       # (4, 12, 3000)
    print(f"cov_loss    : {cov.item():.4f}")     # 0.0

    # Autoregressive decode_step
    token    = torch.randint(0, V, (B, 1))
    logit, kv, alpha, _ = decoder.decode_step(token, None, img, q_hid)
    print(f"step logit  : {logit.shape}")        # (4, 3000)
    print(f"kv_cache    : {kv.shape}")           # (4, 1, 1024)
    print(f"img_alpha   : {alpha.shape}")        # (4, 49)

    token2 = torch.randint(0, V, (B, 1))
    logit2, kv2, alpha2, _ = decoder.decode_step(token2, kv, img, q_hid)
    print(f"step2 logit : {logit2.shape}")       # (4, 3000)
    print(f"kv_cache2   : {kv2.shape}")          # (4, 2, 1024)
