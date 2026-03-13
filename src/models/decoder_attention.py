"""LSTM decoder with Dual Bahdanau Attention — used by Models C, D, and E.

This decoder attends over **two** sources at every decode step:
1. **Image regions** — 49 spatial regions from the CNN/ViT encoder.
2. **Question tokens** — all BiLSTM hidden states from the question encoder.

This dual-attention design allows the decoder to simultaneously focus on
relevant image regions *and* relevant question words at each generation step.

Bahdanau (Additive) Attention — one decode step
------------------------------------------------
Given:
    query  = h_t               (B, hidden_size)     — decoder hidden state
    keys   = img_features      (B, N, hidden_size)  — N spatial regions
    values = img_features      (same as keys in Bahdanau)

Compute::

    energy  = tanh(W_h(h_t).unsqueeze(1) + W_img(img_features))
                                           # (B, N, attn_dim)
    scores  = v(energy).squeeze(-1)        # (B, N)
    alpha   = softmax(scores, dim=1)       # (B, N)  — sums to 1 across regions
    context = (alpha.unsqueeze(2) * img_features).sum(dim=1)  # (B, hidden_size)

Coverage Mechanism (optional, See et al. 2017)
----------------------------------------------
Tracks cumulative attention to penalize the decoder for repeatedly attending
to the same image regions. We use the formula::

    cov_loss_t = (alpha_t * log(coverage_{t-1} + 1.0)).sum(dim=1).mean()

**Note on formula choice**: The original See et al. paper uses
``sum(min(alpha_t, coverage_{t-1}))`` (the overlap between current attention
and cumulative past attention). We use ``alpha * log(coverage + 1)`` instead,
which is a logarithmic penalty that grows smoothly with accumulated attention.
The functional behavior is similar — both penalize re-attention on high-coverage
regions — but the log form avoids the non-differentiability of ``min()``
and produces smoother gradients.

Shape contract (teacher forcing)::

    encoder_hidden  : (num_layers, B, hidden_size)
    img_features    : (B, 49, hidden_size)
    q_hidden_states : (B, Q, hidden_size)
    target_seq      : (B, T)
    logits (output) : (B, T, vocab_size)
    coverage_loss   : scalar
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention module.

    Computes a context vector as a weighted sum of encoder outputs, where the
    weights (alpha) are derived from the compatibility between the current
    decoder hidden state and each encoder output.

    Optionally supports a coverage mechanism that penalizes re-attending to
    already-covered positions.

    Args:
        hidden_size: Dimensionality of both the decoder hidden state and the
            encoder region features.
        attn_dim: Internal attention projection dimension.
        use_coverage: If True, incorporate cumulative attention history
            into the energy computation.
    """

    def __init__(
        self,
        hidden_size: int,
        attn_dim: int = 512,
        use_coverage: bool = False,
    ) -> None:
        super().__init__()
        self.use_coverage = use_coverage

        self.W_h   = nn.Linear(hidden_size, attn_dim)        # project decoder hidden
        self.W_img = nn.Linear(hidden_size, attn_dim)        # project each region
        self.v     = nn.Linear(attn_dim, 1, bias=False)      # score scalar per region

        if use_coverage:
            # Projects a scalar coverage value per region → attn_dim.
            self.W_cov = nn.Linear(1, attn_dim, bias=False)

    def forward(
        self,
        hidden: Tensor,
        img_features: Tensor,
        coverage: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute context vector and attention weights for one decode step.

        Args:
            hidden: ``FloatTensor (B, hidden_size)`` — decoder hidden state.
            img_features: ``FloatTensor (B, N, hidden_size)`` — encoder regions.
            coverage: ``FloatTensor (B, N)`` or None — cumulative attention
                from all previous decode steps.

        Returns:
            Tuple of:
                context – ``FloatTensor (B, hidden_size)``
                alpha   – ``FloatTensor (B, N)`` — attention weights (sum to 1).
        """
        # h_proj  : (B, hidden_size) → (B, 1, attn_dim)  (unsqueeze for broadcast)
        # img_proj: (B, N, hidden_size) → (B, N, attn_dim)
        h_proj   = self.W_h(hidden).unsqueeze(1)   # (B, 1,    attn_dim)
        img_proj = self.W_img(img_features)        # (B, N,    attn_dim)

        energy = h_proj + img_proj                 # (B, N,    attn_dim)  broadcast add

        if self.use_coverage and coverage is not None:
            # coverage: (B, N) → (B, N, 1) → (B, N, attn_dim)
            energy = energy + self.W_cov(coverage.unsqueeze(-1))

        energy = torch.tanh(energy)                # (B, N, attn_dim)
        scores = self.v(energy).squeeze(-1)        # (B, N, 1) → (B, N)
        alpha  = F.softmax(scores, dim=1)          # (B, N)   — sums to 1 over N

        # Weighted sum of image regions.
        context = (alpha.unsqueeze(2) * img_features).sum(dim=1)  # (B, hidden_size)

        return context, alpha


class LSTMDecoderWithAttention(nn.Module):
    """LSTM decoder with dual Bahdanau attention over image and question.

    Used by **Models C, D, and E**.

    At each decode step, the LSTM input is the concatenation of:
    - The token embedding
    - An image context vector (weighted sum of 49 spatial regions)
    - A question context vector (weighted sum of Q question tokens)

    This gives ``input_size = embed_size + hidden_size * 2``.

    The same weight-tying and GloVe-compatibility logic as ``LSTMDecoder``
    is applied to the output projection.

    Args:
        vocab_size: Number of tokens in the answer vocabulary.
        embed_size: Token embedding dimension.
        hidden_size: LSTM hidden state dimension.
        num_layers: Number of stacked LSTM layers.
        attn_dim: Internal attention projection dimension.
        dropout: Dropout probability on embeddings and between LSTM layers.
        pretrained_embeddings: Optional ``FloatTensor (vocab_size, glove_dim)``.
        use_coverage: If True, enable the coverage mechanism on image attention.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        attn_dim: int = 512,
        dropout: float = 0.5,
        pretrained_embeddings: Optional[Tensor] = None,
        use_coverage: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.use_coverage = use_coverage

        # ── Embedding ────────────────────────────────────────────────────────
        if pretrained_embeddings is not None:
            glove_dim = pretrained_embeddings.shape[1]
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=False, padding_idx=0
            )
            self.embed_proj: Optional[nn.Linear] = (
                nn.Linear(glove_dim, embed_size) if glove_dim != embed_size else None
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
            self.embed_proj = None

        # ── Dual attention modules ────────────────────────────────────────────
        # Coverage only on image attention — spatial repetition is the problem
        # we want to penalise; question attention has no analogous issue.
        self.img_attention = BahdanauAttention(hidden_size, attn_dim, use_coverage=use_coverage)
        self.q_attention   = BahdanauAttention(hidden_size, attn_dim, use_coverage=False)

        # ── LSTM ─────────────────────────────────────────────────────────────
        # input_size = embed_size + hidden_size * 2  (token + img_ctx + q_ctx)
        self.lstm = nn.LSTM(
            input_size=embed_size + hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ── Output projection (with optional weight tying) ───────────────────
        if pretrained_embeddings is not None:
            # GloVe path: no weight tying (bottleneck avoidance).
            self.out_proj = nn.Linear(hidden_size, embed_size)
            self.fc       = nn.Linear(embed_size, vocab_size, bias=False)
        else:
            # Standard path: tie fc.weight ≡ embedding.weight.
            actual_embed_dim = self.embedding.embedding_dim
            self.out_proj    = nn.Linear(hidden_size, actual_embed_dim)
            self.fc          = nn.Linear(actual_embed_dim, vocab_size, bias=False)
            self.fc.weight   = self.embedding.weight   # weight tying

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        encoder_hidden: Tuple[Tensor, Tensor],
        img_features: Tensor,
        q_hidden_states: Tensor,
        target_seq: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Teacher-forcing forward pass with dual attention.

        Args:
            encoder_hidden: Tuple ``(h_0, c_0)``, each
                ``(num_layers, B, hidden_size)`` — from fusion.
            img_features: ``FloatTensor (B, 49, hidden_size)`` — spatial CNN/ViT features.
            q_hidden_states: ``FloatTensor (B, Q, hidden_size)`` — BiLSTM question states.
            target_seq: ``LongTensor (B, T)`` — decoder input tokens.

        Returns:
            Tuple of:
                logits        – ``FloatTensor (B, T, vocab_size)``
                coverage_loss – Scalar tensor (0.0 if coverage disabled).
        """
        batch, max_len = target_seq.shape
        num_regions    = img_features.size(1)   # 49

        # Embed the full target sequence upfront (efficient for teacher forcing).
        embeds = self.dropout(self.embedding(target_seq))  # (B, T, embed_size or glove_dim)
        if self.embed_proj is not None:
            embeds = self.embed_proj(embeds)               # (B, T, embed_size)

        hidden: Tuple[Tensor, Tensor] = encoder_hidden

        # Coverage accumulator: (B, 49), zeros at step 0.
        coverage = img_features.new_zeros(batch, num_regions) if self.use_coverage else None
        cov_loss = img_features.new_zeros(1)

        logits_list: list[Tensor] = []

        # Step-by-step loop — required because attention at step t depends on
        # the hidden state from step t-1.
        for t in range(max_len):
            embed_t = embeds[:, t, :]              # (B, embed_size)
            h_top   = hidden[0][-1]                # (B, hidden_size) — top LSTM layer

            # Dual attention: image (with optional coverage) + question.
            img_context, img_alpha = self.img_attention(h_top, img_features, coverage)
            # img_context : (B, hidden_size)
            # img_alpha   : (B, 49)

            q_context, _ = self.q_attention(h_top, q_hidden_states)
            # q_context   : (B, hidden_size)

            # Coverage loss: penalise re-attending to already-covered regions.
            # Formula: cov_loss_t = (alpha_t * log(coverage_{t-1} + 1)).sum(dim=1).mean()
            # At t=0: coverage=0 → log(1)=0 → no penalty (correct, no history yet).
            # At t>0: high coverage → higher penalty for re-attending.
            if self.use_coverage:
                cov_loss = cov_loss + (img_alpha * torch.log(coverage + 1.0)).sum(dim=1).mean()
                coverage = coverage + img_alpha    # accumulate for next step

            # Concatenate: token_embed + image_context + question_context.
            lstm_input = torch.cat(
                [embed_t, img_context, q_context], dim=1
            ).unsqueeze(1)
            # (B, embed_size + hidden_size + hidden_size) → (B, 1, embed_size + hidden_size*2)

            output, hidden = self.lstm(lstm_input, hidden)  # output: (B, 1, hidden_size)
            logit = self.fc(self.out_proj(output.squeeze(1)))  # (B, vocab_size)
            logits_list.append(logit)

        logits = torch.stack(logits_list, dim=1)  # (B, T, vocab_size)

        # Normalise coverage loss by sequence length to keep its scale invariant.
        coverage_loss = (cov_loss / max_len) if self.use_coverage else cov_loss.squeeze()
        return logits, coverage_loss

    def decode_step(
        self,
        token: Tensor,
        hidden: Tuple[Tensor, Tensor],
        img_features: Tensor,
        q_hidden_states: Tensor,
        coverage: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor, Optional[Tensor]]:
        """Single autoregressive decode step for inference.

        Args:
            token: ``LongTensor (B, 1)`` — current input token.
            hidden: Tuple ``(h, c)`` each ``(num_layers, B, hidden_size)``.
            img_features: ``FloatTensor (B, 49, hidden_size)``.
            q_hidden_states: ``FloatTensor (B, Q, hidden_size)``.
            coverage: ``FloatTensor (B, 49)`` or None — cumulative attention.

        Returns:
            Tuple of:
                logit    – ``FloatTensor (B, vocab_size)``
                hidden   – Updated ``(h, c)`` tuple.
                img_alpha – ``FloatTensor (B, 49)`` — image attention weights
                            (used for heatmap visualization).
                coverage – Updated ``FloatTensor (B, 49)`` or None.
        """
        embed  = self.dropout(self.embedding(token))   # (B, 1, embed_size or glove_dim)
        if self.embed_proj is not None:
            embed = self.embed_proj(embed)             # (B, 1, embed_size)
        h_top  = hidden[0][-1]                        # (B, hidden_size)

        img_context, img_alpha = self.img_attention(h_top, img_features, coverage)
        q_context, _           = self.q_attention(h_top, q_hidden_states)

        if self.use_coverage:
            if coverage is None:
                coverage = torch.zeros_like(img_alpha)
            coverage = coverage + img_alpha

        lstm_input = torch.cat(
            [embed.squeeze(1), img_context, q_context], dim=1
        ).unsqueeze(1)
        # (B, embed_size + hidden_size * 2) → (B, 1, embed_size + hidden_size * 2)

        output, hidden = self.lstm(lstm_input, hidden)
        logit = self.fc(self.out_proj(output.squeeze(1)))  # (B, vocab_size)

        return logit, hidden, img_alpha, coverage

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
            img_features: ``FloatTensor (B, 49, hidden_size)``.
            q_hidden_states: ``FloatTensor (B, Q, hidden_size)``.
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

        token = torch.full((batch, 1), start_idx, dtype=torch.long, device=device)

        coverage = (
            img_features.new_zeros(batch, img_features.size(1))
            if self.use_coverage else None
        )
        unfinished = torch.ones(batch, dtype=torch.bool, device=device)

        seqs:      list[Tensor] = []
        log_probs: list[Tensor] = []

        for _ in range(max_len):
            logit, hidden, _, coverage = self.decode_step(
                token, hidden, img_features, q_hidden_states, coverage
            )  # logit: (B, vocab_size)

            if method == "sample":
                predicted_token = torch.multinomial(
                    F.softmax(logit, dim=-1), num_samples=1
                )                                                     # (B, 1)
            else:
                predicted_token = torch.argmax(logit, dim=-1, keepdim=True)  # (B, 1)

            step_log_probs     = F.log_softmax(logit, dim=-1)                    # (B, vocab_size)
            selected_log_probs = step_log_probs.gather(1, predicted_token)       # (B, 1)

            seqs.append(predicted_token)
            log_probs.append(selected_log_probs)

            token      = predicted_token * unfinished.unsqueeze(1).long()
            unfinished = unfinished & (predicted_token.squeeze(1) != end_idx)

        seqs      = torch.cat(seqs,      dim=1)   # (B, max_len)
        log_probs = torch.cat(log_probs, dim=1)   # (B, max_len)
        return seqs, log_probs


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, V, E, H, T, N, Q = 4, 3000, 512, 1024, 10, 49, 15

    decoder = LSTMDecoderWithAttention(vocab_size=V, embed_size=E, hidden_size=H, num_layers=2)

    h           = torch.zeros(2, B, H)
    c           = torch.zeros(2, B, H)
    img_feats   = torch.randn(B, N, H)
    q_hidden    = torch.randn(B, Q, H)
    target      = torch.randint(0, V, (B, T))

    logits, cov_loss = decoder((h, c), img_feats, q_hidden, target)
    print(f"logits      : {logits.shape}")         # expect (4, 10, 3000)
    print(f"cov_loss    : {cov_loss.item():.4f}")  # expect 0.0 (coverage disabled)

    token     = torch.randint(0, V, (B, 1))
    logit, hidden_new, alpha, cov = decoder.decode_step(token, (h, c), img_feats, q_hidden)
    print(f"decode_step logit : {logit.shape}")    # expect (4, 3000)
    print(f"attention alpha   : {alpha.shape}")    # expect (4, 49)
