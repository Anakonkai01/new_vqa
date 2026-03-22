"""
LSTMDecoderWithAttention — dual MHCA decoder for Models C-F (re-export).
LSTMDecoderG             — Model G decoder: G2 (3-way PGN) + G5 (length embedding).

Step D implementation of G2 + G5 in this file.

G5 changes LSTM input size: embed(512) + img_ctx(1024) + q_ctx(1024) + len_emb(64) = 2624.
G2 uses ThreeWayPGNHead instead of PointerGeneratorHead (softmax 3-way vs. sigmoid scalar).
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from decoder_attention import LSTMDecoderWithAttention, MultiHeadCrossAttention
from decoder_lstm import LayerNormLSTMStack, WeightDrop
from models.pointer_generator import ThreeWayPGNHead


# ---------------------------------------------------------------------------
# LSTMDecoderG — G2 (3-way PGN) + G5 (length embedding)
# ---------------------------------------------------------------------------

class LSTMDecoderG(nn.Module):
    """
    Model G LSTM decoder with dual MHCA + three-way PGN + length conditioning.

    Extends the Model F decoder (LSTMDecoderWithAttention) with:
      G5: LengthEmbedding(3, 64) — appended to LSTM input at each step.
          LSTM input: embed(512) + img_ctx(1024) + q_ctx(1024) + len_emb(64) = 2624.
          At inference: always feed LENGTH_BIN_LONG (bin=2) via length_bin arg.
      G2: ThreeWayPGNHead — replaces sigmoid scalar p_gen with softmax [p_g, p_cQ, p_cV].
          Distributes image attention weights over visual object label tokens (Eq 31).

    Args:
        vocab_size        : answer vocabulary size |V_A|
        embed_size        : token embedding dim (512)
        hidden_size       : LSTM + MHCA hidden dim (1024)
        num_layers        : LSTM layers (2)
        dropout           : dropout probability
        pretrained_embeddings : GloVe matrix (V_A, glove_dim) or None
        use_coverage      : enable image-side coverage bias
        use_layer_norm    : use LayerNormLSTMStack + highway connections
        use_dropconnect   : DropConnect on hidden-to-hidden weights
        num_heads         : MHCA heads (4)
        len_embed_dim     : G5 length embedding dimension (64)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        attn_dim: int = 512,
        dropout: float = 0.5,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        use_coverage: bool = False,
        use_layer_norm: bool = False,
        use_dropconnect: bool = False,
        num_heads: int = 4,
        len_embed_dim: int = 64,
        use_mac_in_decoder: bool = False,
    ):
        super().__init__()

        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.use_coverage  = use_coverage
        self.vocab_size    = vocab_size
        self.len_embed_dim = len_embed_dim
        self.use_mac_in_decoder = use_mac_in_decoder

        # ── G5: Length embedding ───────────────────────────────────────────
        self.len_embedding = nn.Embedding(3, len_embed_dim)   # 3 bins × 64-dim

        # ── Token embedding ────────────────────────────────────────────────
        if pretrained_embeddings is not None:
            glove_dim = pretrained_embeddings.shape[1]
            self.embedding  = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=False, padding_idx=0)
            self.embed_proj = nn.Linear(glove_dim, embed_size) \
                              if glove_dim != embed_size else None
        else:
            self.embedding  = nn.Embedding(vocab_size, embed_size, padding_idx=0)
            self.embed_proj = None

        # ── Dual MHCA ─────────────────────────────────────────────────────
        self.img_mhca = MultiHeadCrossAttention(
            hidden_size, num_heads=num_heads, use_coverage=use_coverage)
        self.q_mhca   = MultiHeadCrossAttention(
            hidden_size, num_heads=num_heads, use_coverage=False)

        # ── LSTM (embed + 2*H + len_emb [+ H MAC ctx for Model H]) ─────────
        base_lstm_in = embed_size + hidden_size * 2 + len_embed_dim  # 2624
        if use_mac_in_decoder:
            self.mac_ctx_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            lstm_input_size = base_lstm_in + hidden_size
        else:
            self.mac_ctx_proj = None
            lstm_input_size = base_lstm_in
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

        # ── Output projection + weight tying ──────────────────────────────
        if pretrained_embeddings is not None:
            glove_dim = pretrained_embeddings.shape[1]
            self.out_proj = nn.Linear(hidden_size, glove_dim)
            self.fc       = nn.Linear(glove_dim, vocab_size, bias=False)
            self.fc.weight = self.embedding.weight   # Force Weight Tying!
        else:
            actual_embed  = self.embedding.embedding_dim
            self.out_proj = nn.Linear(hidden_size, actual_embed)
            self.fc       = nn.Linear(actual_embed, vocab_size, bias=False)
            self.fc.weight = self.embedding.weight   # weight tying

        self.dropout = nn.Dropout(dropout)

        # ── G2: Three-way PGN ──────────────────────────────────────────────
        # Input: [c_img(H); h_t(H); x_t(lstm_input)] — lstm_input includes optional MAC ctx
        pgn_input_dim = hidden_size + hidden_size + lstm_input_size
        self.pgn = ThreeWayPGNHead(pgn_input_dim)

    # ── Teacher-forcing forward (training) ─────────────────────────────────

    def forward(
        self,
        encoder_hidden: Tuple[torch.Tensor, torch.Tensor],
        img_features: torch.Tensor,
        q_hidden_states: torch.Tensor,
        target_seq: torch.Tensor,
        q_token_ids: Optional[torch.Tensor] = None,
        img_mask: Optional[torch.Tensor] = None,
        length_bin: Optional[torch.Tensor] = None,
        label_tokens: Optional[torch.Tensor] = None,
        mac_memory: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training mode — Teacher Forcing with dual MHCA + G2 + G5.

        Args:
            encoder_hidden  : (h, c) each (num_layers, B, H) — from MUTAN fusion
            img_features    : (B, k, H)  — BUTD projected + L2-normed features
            q_hidden_states : (B, L, H)  — BiLSTM output for MHCA
            target_seq      : (B, T)     — [<start>, w1, w2, ...] teacher forcing
            q_token_ids     : (B, L) or None — question token indices for PGN
            img_mask        : (B, k) bool  — True=valid region (BUTD padding mask)
            length_bin      : (B,) int64 in {0,1,2} — G5; defaults to LONG if None
            label_tokens    : (B, k, max_t) int64 — G2 visual label indices; None→skip
            mac_memory      : (B, H) final MAC memory — Model H only; concatenated each step

        Returns:
            logits        : (B, T, V) log-probabilities (from ThreeWayPGNHead)
            coverage_loss : scalar tensor
        """
        batch   = target_seq.size(0)
        max_len = target_seq.size(1)

        embeds = self.dropout(self.embedding(target_seq))   # (B, T, E)
        if self.embed_proj is not None:
            embeds = self.embed_proj(embeds)

        # G5: resolve length bin → embedding; default = LONG at inference
        if length_bin is None:
            length_bin = target_seq.new_full((batch,), 2)   # LONG
        len_emb = self.len_embedding(length_bin)             # (B, 64)

        if mac_memory is not None:
            if not self.use_mac_in_decoder:
                raise ValueError("mac_memory set but decoder was built with use_mac_in_decoder=False")
            mac_ctx = self.mac_ctx_proj(mac_memory)
        else:
            mac_ctx = None

        hidden      = encoder_hidden
        num_regions = img_features.size(1)
        coverage    = img_features.new_zeros(batch, num_regions) \
                      if self.use_coverage else None
        cov_loss    = img_features.new_zeros(1)

        logits_list = []

        for t in range(max_len):
            embed_t = embeds[:, t, :]        # (B, E)
            h_top   = hidden[0][-1]          # (B, H) — top LSTM layer

            # Image cross-attention
            img_context, img_alpha = self.img_mhca(
                h_top, img_features, coverage, mask=img_mask)

            # Question cross-attention
            q_context, q_alpha = self.q_mhca(h_top, q_hidden_states)

            # Coverage update
            if self.use_coverage:
                cov_loss = cov_loss + torch.min(img_alpha, coverage).sum(dim=1).mean()
                coverage = coverage + img_alpha

            # G5 (+ optional MAC context for Model H)
            if mac_ctx is not None:
                lstm_input = torch.cat([embed_t, img_context, q_context, len_emb, mac_ctx], dim=1)
            else:
                lstm_input = torch.cat([embed_t, img_context, q_context, len_emb], dim=1)

            output, hidden = self.lstm(lstm_input.unsqueeze(1), hidden)
            vocab_logit = self.fc(self.out_proj(output.squeeze(1)))  # (B, V)

            # G2: three-way PGN
            p_g, p_cQ, p_cV = self.pgn(img_context, h_top, lstm_input)
            logit = ThreeWayPGNHead.blend_3way(
                p_g, p_cQ, p_cV,
                vocab_logit, q_alpha, q_token_ids,
                img_alpha, label_tokens, self.vocab_size,
            )
            logits_list.append(logit)

        logits       = torch.stack(logits_list, dim=1)            # (B, T, V)
        coverage_loss = cov_loss / max_len if self.use_coverage else cov_loss.squeeze()
        return logits, coverage_loss

    # ── Single autoregressive step (inference) ─────────────────────────────

    def decode_step(
        self,
        token: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        img_features: torch.Tensor,
        q_hidden_states: torch.Tensor,
        coverage: Optional[torch.Tensor] = None,
        q_token_ids: Optional[torch.Tensor] = None,
        img_mask: Optional[torch.Tensor] = None,
        length_bin: Optional[torch.Tensor] = None,
        label_tokens: Optional[torch.Tensor] = None,
        mac_memory: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple, torch.Tensor, Optional[torch.Tensor]]:
        """
        Single autoregressive decode step for beam search.

        Args:
            token       : (B, 1) current token id
            hidden      : (h, c) each (num_layers, B, H)
            img_features: (B, k, H)
            q_hidden_states: (B, L, H)
            coverage    : (B, k) accumulated image attention or None
            q_token_ids : (B, L) or None — for G2
            img_mask    : (B, k) bool or None
            length_bin  : (B,) int64 — G5; defaults to LONG (2) if None
            label_tokens: (B, k, max_t) int64 or None — G2
            mac_memory  : (B, H) final MAC memory — Model H only

        Returns:
            logit      : (B, V) log-probabilities
            hidden     : updated (h, c)
            img_alpha  : (B, k) for visualization / coverage
            coverage   : (B, k) updated or None
        """
        embed = self.dropout(self.embedding(token))   # (B, 1, E)
        if self.embed_proj is not None:
            embed = self.embed_proj(embed)
        embed_1d = embed.squeeze(1)                   # (B, E)
        h_top    = hidden[0][-1]                      # (B, H)

        # G5 length embedding — always LONG at inference
        if length_bin is None:
            length_bin = token.new_full((token.size(0),), 2)
        len_emb = self.len_embedding(length_bin)       # (B, 64)

        if mac_memory is not None:
            if not self.use_mac_in_decoder:
                raise ValueError("mac_memory set but decoder was built with use_mac_in_decoder=False")
            mac_ctx = self.mac_ctx_proj(mac_memory)
        else:
            mac_ctx = None

        img_context, img_alpha = self.img_mhca(
            h_top, img_features, coverage, mask=img_mask)
        q_context, q_alpha = self.q_mhca(h_top, q_hidden_states)

        if self.use_coverage:
            if coverage is None:
                coverage = torch.zeros_like(img_alpha)
            coverage = coverage + img_alpha

        if mac_ctx is not None:
            lstm_input = torch.cat([embed_1d, img_context, q_context, len_emb, mac_ctx], dim=1)
        else:
            lstm_input = torch.cat([embed_1d, img_context, q_context, len_emb], dim=1)
        output, hidden = self.lstm(lstm_input.unsqueeze(1), hidden)
        vocab_logit = self.fc(self.out_proj(output.squeeze(1)))  # (B, V)

        # G2: three-way blend
        p_g, p_cQ, p_cV = self.pgn(img_context, h_top, lstm_input)
        logit = ThreeWayPGNHead.blend_3way(
            p_g, p_cQ, p_cV,
            vocab_logit, q_alpha, q_token_ids,
            img_alpha, label_tokens, self.vocab_size,
        )

        return logit, hidden, img_alpha, coverage


__all__ = ["LSTMDecoderWithAttention", "MultiHeadCrossAttention", "LSTMDecoderG"]
