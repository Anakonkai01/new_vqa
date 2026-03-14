"""LSTM decoder with Dual Multi-Head Cross-Attention — used by Model G.

Upgrade over Model E/F
----------------------
Model E/F uses single-head Bahdanau (additive) attention::

    energy_i = v · tanh(W_h h + W_img img_i)   # scalar per region
    alpha    = softmax(energy)                   # (49,) — one distribution
    context  = Σ alpha_i · img_i                # (H,)  — one context vector

A single head can express **one query intent** per step.  Multi-Head Attention
(Vaswani et al., 2017) runs ``h`` independent sub-spaces in parallel::

    head_i  = softmax( (h W_i^Q)(img W_i^K)^T / √d_k ) · (img W_i^V)
    output  = concat(head_1, …, head_h) · W_O

where ``d_k = H / num_heads = 1024 / 8 = 128``.

The ``1/√d_k`` scaling keeps dot-product variance at 1.0 regardless of dimension,
preventing the softmax saturation that occurs without it.

Eight heads can simultaneously specialise in patterns such as:
- "where is the main object?" (localisation)
- "what colour is it?" (attribute)
- "is this a binary yes/no question?" (task type)
- ...

API compatibility
-----------------
The ``forward``, ``decode_step``, and ``sample`` signatures are **identical** to
``LSTMDecoderWithAttention`` in ``decoder_attention.py`` — drop-in for Model G.

Shape contract (teacher forcing)::

    encoder_hidden  : (num_layers, B, hidden_size)
    img_features    : (B, 49, hidden_size)
    q_hidden_states : (B, 77, hidden_size)   ← CLIP tokens (Model F+)
    target_seq      : (B, T)
    logits (output) : (B, T, vocab_size)
    coverage_loss   : scalar (always 0.0 — coverage not used with MHA)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LSTMDecoderWithMHA(nn.Module):
    """LSTM decoder with dual multi-head cross-attention over image and question.

    Used by **Model G**.

    Architecture per decode step::

        h_top     = hidden[0][-1]                              # (B, H)
        query     = h_top.unsqueeze(1)                         # (B, 1, H)
        img_ctx,_ = img_mha(query, img_features, img_features) # (B, 1, H)
        q_ctx, _  = q_mha(query, q_hidden_states, q_hidden_states)  # (B, 1, H)
        img_ctx   = img_ctx.squeeze(1)                         # (B, H)
        q_ctx     = q_ctx.squeeze(1)                           # (B, H)
        lstm_in   = [embed ; img_ctx ; q_ctx]                  # (B, embed+2H)
        out, h    = lstm(lstm_in.unsqueeze(1), hidden)
        logit     = fc(out_proj(out.squeeze(1)))               # (B, vocab_size)

    Args:
        vocab_size: Number of tokens in the answer vocabulary.
        embed_size: Token embedding dimension.
        hidden_size: LSTM hidden state dimension (and MHA embed_dim).
        num_layers: Number of stacked LSTM layers.
        num_heads: Number of attention heads (default 8).
            Must divide ``hidden_size`` evenly.
        dropout: Dropout probability.
        pretrained_embeddings: Optional GloVe matrix ``(vocab_size, glove_dim)``.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int = 8,
        dropout: float = 0.5,
        pretrained_embeddings: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # ── Embedding ──────────────────────────────────────────────────────────
        if pretrained_embeddings is not None:
            glove_dim = pretrained_embeddings.shape[1]
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=False, padding_idx=0
            )
            self.embed_proj: Optional[nn.Linear] = (
                nn.Linear(glove_dim, embed_size) if glove_dim != embed_size else None
            )
        else:
            self.embedding  = nn.Embedding(vocab_size, embed_size, padding_idx=0)
            self.embed_proj = None

        # ── Dual Multi-Head Cross-Attention ────────────────────────────────────
        # batch_first=True so tensors are (B, seq, dim) throughout.
        self.img_mha = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            dropout=0.0, batch_first=True,
        )
        self.q_mha = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            dropout=0.0, batch_first=True,
        )

        # ── LSTM ───────────────────────────────────────────────────────────────
        # input_size = embed_size + hidden_size * 2  (token + img_ctx + q_ctx)
        self.lstm = nn.LSTM(
            input_size=embed_size + hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ── Output projection (with optional weight tying) ─────────────────────
        if pretrained_embeddings is not None:
            self.out_proj = nn.Linear(hidden_size, embed_size)
            self.fc       = nn.Linear(embed_size, vocab_size, bias=False)
        else:
            actual_embed_dim = self.embedding.embedding_dim
            self.out_proj    = nn.Linear(hidden_size, actual_embed_dim)
            self.fc          = nn.Linear(actual_embed_dim, vocab_size, bias=False)
            self.fc.weight   = self.embedding.weight   # weight tying

        self.dropout = nn.Dropout(dropout)

    # ── Internal helper ────────────────────────────────────────────────────────

    def _attend(
        self,
        h_top: Tensor,
        img_features: Tensor,
        q_hidden_states: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Run dual MHA for one decode step.

        Args:
            h_top: ``FloatTensor (B, H)`` — top LSTM layer hidden state.
            img_features: ``FloatTensor (B, 49, H)``.
            q_hidden_states: ``FloatTensor (B, 77, H)``.

        Returns:
            Tuple of:
                img_ctx  – ``FloatTensor (B, H)``
                q_ctx    – ``FloatTensor (B, H)``
                img_alpha – ``FloatTensor (B, 49)`` — averaged attention weights
                             across heads (for visualisation / coverage compat).
        """
        query = h_top.unsqueeze(1)   # (B, 1, H)

        img_ctx, img_weights = self.img_mha(query, img_features, img_features)
        # img_ctx:     (B, 1, H)
        # img_weights: (B, 1, 49)  — averaged over heads

        q_ctx, _ = self.q_mha(query, q_hidden_states, q_hidden_states)
        # q_ctx:   (B, 1, H)

        img_ctx   = img_ctx.squeeze(1)               # (B, H)
        q_ctx     = q_ctx.squeeze(1)                 # (B, H)
        img_alpha = img_weights.squeeze(1)           # (B, 49)

        return img_ctx, q_ctx, img_alpha

    # ── Forward (teacher-forcing) ──────────────────────────────────────────────

    def forward(
        self,
        encoder_hidden: Tuple[Tensor, Tensor],
        img_features: Tensor,
        q_hidden_states: Tensor,
        target_seq: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Teacher-forcing forward pass with dual MHA.

        Args:
            encoder_hidden: Tuple ``(h_0, c_0)``, each ``(num_layers, B, H)``.
            img_features: ``FloatTensor (B, 49, H)`` — spatial image features.
            q_hidden_states: ``FloatTensor (B, 77, H)`` — CLIP question tokens.
            target_seq: ``LongTensor (B, T)`` — decoder input (teacher-forced).

        Returns:
            Tuple of:
                logits        – ``FloatTensor (B, T, vocab_size)``
                coverage_loss – scalar 0.0 tensor (API compatibility with C/D/E)
        """
        batch, max_len = target_seq.shape

        embeds = self.dropout(self.embedding(target_seq))   # (B, T, E or glove_dim)
        if self.embed_proj is not None:
            embeds = self.embed_proj(embeds)                # (B, T, E)

        hidden: Tuple[Tensor, Tensor] = encoder_hidden
        logits_list: list[Tensor] = []

        for t in range(max_len):
            embed_t = embeds[:, t, :]              # (B, E)
            h_top   = hidden[0][-1]                # (B, H)

            img_ctx, q_ctx, _ = self._attend(h_top, img_features, q_hidden_states)

            lstm_input = torch.cat(
                [embed_t, img_ctx, q_ctx], dim=1
            ).unsqueeze(1)                         # (B, 1, E+2H)

            output, hidden = self.lstm(lstm_input, hidden)   # (B, 1, H)
            logit = self.fc(self.out_proj(output.squeeze(1)))  # (B, vocab_size)
            logits_list.append(logit)

        logits = torch.stack(logits_list, dim=1)   # (B, T, vocab_size)
        coverage_loss = img_features.new_zeros(1).squeeze()
        return logits, coverage_loss

    # ── Decode step (autoregressive inference) ─────────────────────────────────

    def decode_step(
        self,
        token: Tensor,
        hidden: Tuple[Tensor, Tensor],
        img_features: Tensor,
        q_hidden_states: Tensor,
        coverage: Optional[Tensor] = None,   # accepted but unused (API compat)
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor, None]:
        """Single autoregressive decode step.

        Args:
            token: ``LongTensor (B, 1)`` — current input token.
            hidden: Tuple ``(h, c)`` each ``(num_layers, B, H)``.
            img_features: ``FloatTensor (B, 49, H)``.
            q_hidden_states: ``FloatTensor (B, 77, H)``.
            coverage: Ignored (kept for API compatibility with Bahdanau decoder).

        Returns:
            Tuple of:
                logit    – ``FloatTensor (B, vocab_size)``
                hidden   – Updated ``(h, c)`` tuple.
                img_alpha – ``FloatTensor (B, 49)`` — averaged attention weights.
                None     – coverage placeholder (not used).
        """
        embed  = self.dropout(self.embedding(token))    # (B, 1, E or glove_dim)
        if self.embed_proj is not None:
            embed = self.embed_proj(embed)              # (B, 1, E)

        h_top = hidden[0][-1]                           # (B, H)
        img_ctx, q_ctx, img_alpha = self._attend(h_top, img_features, q_hidden_states)

        lstm_input = torch.cat(
            [embed.squeeze(1), img_ctx, q_ctx], dim=1
        ).unsqueeze(1)                                  # (B, 1, E+2H)

        output, hidden = self.lstm(lstm_input, hidden)  # output: (B, 1, H)
        logit = self.fc(self.out_proj(output.squeeze(1)))  # (B, vocab_size)

        return logit, hidden, img_alpha, None

    # ── Sample (SCST) ──────────────────────────────────────────────────────────

    def sample(
        self,
        encoder_hidden: Tuple[Tensor, Tensor],
        img_features: Tensor,
        q_hidden_states: Tensor,
        max_len: int,
        start_idx: int,
        end_idx: int,
        method: str = "greedy",
    ) -> Tuple[Tensor, Tensor]:
        """Generate sequences and log-probabilities for SCST training.

        Args:
            encoder_hidden: Tuple ``(h_0, c_0)`` from fusion.
            img_features: ``FloatTensor (B, 49, H)``.
            q_hidden_states: ``FloatTensor (B, 77, H)``.
            max_len: Maximum sequence length to generate.
            start_idx: Index of the ``<start>`` token.
            end_idx: Index of the ``<end>`` token.
            method: ``'greedy'`` (argmax baseline) or ``'sample'`` (multinomial).

        Returns:
            Tuple of:
                seqs      – ``LongTensor  (B, max_len)``
                log_probs – ``FloatTensor (B, max_len)``
        """
        batch  = img_features.size(0)
        device = img_features.device
        hidden = encoder_hidden

        token      = torch.full((batch, 1), start_idx, dtype=torch.long, device=device)
        unfinished = torch.ones(batch, dtype=torch.bool, device=device)

        seqs:      list[Tensor] = []
        log_probs: list[Tensor] = []

        for _ in range(max_len):
            logit, hidden, _, _ = self.decode_step(
                token, hidden, img_features, q_hidden_states
            )   # logit: (B, vocab_size)

            if method == "sample":
                predicted_token = torch.multinomial(
                    F.softmax(logit, dim=-1), num_samples=1
                )                                                          # (B, 1)
            else:
                predicted_token = torch.argmax(logit, dim=-1, keepdim=True)  # (B, 1)

            step_log_probs     = F.log_softmax(logit, dim=-1)
            selected_log_probs = step_log_probs.gather(1, predicted_token)  # (B, 1)

            seqs.append(predicted_token)
            log_probs.append(selected_log_probs)

            token      = predicted_token * unfinished.unsqueeze(1).long()
            unfinished = unfinished & (predicted_token.squeeze(1) != end_idx)

        seqs      = torch.cat(seqs,      dim=1)   # (B, max_len)
        log_probs = torch.cat(log_probs, dim=1)   # (B, max_len)
        return seqs, log_probs


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, V, E, H, T, N, Q = 4, 3000, 512, 1024, 10, 49, 77

    decoder = LSTMDecoderWithMHA(vocab_size=V, embed_size=E, hidden_size=H, num_layers=2)

    h      = torch.zeros(2, B, H)
    c      = torch.zeros(2, B, H)
    img    = torch.randn(B, N, H)
    q_hid  = torch.randn(B, Q, H)
    target = torch.randint(0, V, (B, T))

    logits, cov = decoder((h, c), img, q_hid, target)
    print(f"logits      : {logits.shape}")       # (4, 10, 3000)
    print(f"cov_loss    : {cov.item():.4f}")     # 0.0

    token = torch.randint(0, V, (B, 1))
    logit, hid2, alpha, _ = decoder.decode_step(token, (h, c), img, q_hid)
    print(f"decode_step : {logit.shape}")        # (4, 3000)
    print(f"img_alpha   : {alpha.shape}")        # (4, 49)
