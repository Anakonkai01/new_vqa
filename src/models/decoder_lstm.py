"""LSTM decoder without attention — used by Models A and B.

Decodes a sequence token-by-token given an initial hidden state produced by
the encoder fusion module.

Training (Teacher Forcing)
--------------------------
The full target sequence is passed at once. At each step *t*, the GT token
at position *t* is used as input regardless of the model's previous output.
This is efficient (parallelisable via batched LSTM) but introduces exposure
bias at inference time (addressed by Scheduled Sampling in Phase 3).

Inference (Autoregressive)
--------------------------
One token is generated at a time via ``decode_step()``.
The predicted token at step *t* is fed back as input at step *t+1*.

RL Sampling (SCST)
------------------
``sample()`` generates sequences with log-probabilities for the REINFORCE
gradient estimator used in Phase 4 SCST training.

Shape contract (teacher forcing)::

    encoder_hidden : (num_layers, B, hidden_size)  — from fusion
    target_seq     : (B, T)                        — [<start>, w1, ..., w_{T-1}]
    logits (output): (B, T, vocab_size)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LSTMDecoder(nn.Module):
    """LSTM decoder without spatial attention.

    Used by **Models A and B** where the image is represented as a single
    global vector rather than a spatial grid.

    The decoder is initialized with the multimodal fusion vector as the LSTM
    hidden state, and subsequently generates tokens autoregressively.

    **Weight Tying** (Press & Wolf, 2017): The output projection matrix is
    tied to the input embedding matrix when no pre-trained GloVe embeddings
    are used. This reduces parameters and regularizes the model by forcing
    the output space to align with the input embedding space.

    When GloVe embeddings are used, weight tying is intentionally disabled.
    GloVe vectors have dimension 300, creating a bottleneck
    ``(hidden_size=1024 → 300 → vocab_size)`` that would severely constrain
    the model. Instead, we project through ``embed_size`` (512) without tying.

    Args:
        vocab_size: Number of tokens in the answer vocabulary.
        embed_size: Token embedding dimension.
        hidden_size: LSTM hidden state dimension.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability on embeddings and between LSTM layers.
        pretrained_embeddings: Optional ``FloatTensor (vocab_size, glove_dim)``.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.5,
        pretrained_embeddings: Optional[Tensor] = None,
    ) -> None:
        super().__init__()

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
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=0
            )
            self.embed_proj = None

        self.dropout = nn.Dropout(dropout)

        # ── LSTM ─────────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=embed_size,
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

    def forward(
        self,
        encoder_hidden: Tuple[Tensor, Tensor],
        target_seq: Tensor,
    ) -> Tensor:
        """Teacher-forcing forward pass over the full target sequence.

        Args:
            encoder_hidden: Tuple ``(h_0, c_0)``, each
                ``FloatTensor (num_layers, B, hidden_size)`` — from fusion.
            target_seq: ``LongTensor (B, T)`` — decoder input tokens
                ``[<start>, w_1, ..., w_{T-1}]``.

        Returns:
            ``FloatTensor (B, T, vocab_size)`` — unnormalized logits.
        """
        # (B, T) → (B, T, embed_size)
        embeds = self.dropout(self.embedding(target_seq))
        if self.embed_proj is not None:
            embeds = self.embed_proj(embeds)  # (B, T, glove_dim) → (B, T, embed_size)

        # (B, T, embed_size) → (B, T, hidden_size)
        outputs, _hidden = self.lstm(embeds, encoder_hidden)

        # (B, T, hidden_size) → (B, T, embed_size) → (B, T, vocab_size)
        logits = self.fc(self.out_proj(outputs))
        return logits

    def decode_step(
        self,
        token: Tensor,
        hidden: Tuple[Tensor, Tensor],
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Single autoregressive decode step for inference.

        Args:
            token: ``LongTensor (B, 1)`` — current input token.
            hidden: Tuple ``(h, c)``, each ``(num_layers, B, hidden_size)``.

        Returns:
            Tuple of:
                logit  – ``FloatTensor (B, vocab_size)`` — unnormalized scores.
                hidden – Updated ``(h, c)`` tuple.
        """
        embed = self.dropout(self.embedding(token))    # (B, 1) → (B, 1, embed_size)
        if self.embed_proj is not None:
            embed = self.embed_proj(embed)             # (B, 1, glove_dim) → (B, 1, embed_size)

        output, hidden = self.lstm(embed, hidden)      # output: (B, 1, hidden_size)
        logit = self.fc(self.out_proj(output.squeeze(1)))  # (B, hidden_size) → (B, vocab_size)
        return logit, hidden

    def sample(
        self,
        encoder_hidden: Tuple[Tensor, Tensor],
        max_len: int,
        start_idx: int,
        end_idx: int,
        method: str = "greedy",
    ) -> Tuple[Tensor, Tensor]:
        """Generate sequences and their log-probabilities for SCST training.

        Args:
            encoder_hidden: Tuple ``(h_0, c_0)`` from fusion.
            max_len: Maximum number of tokens to generate.
            start_idx: Index of the ``<start>`` token.
            end_idx: Index of the ``<end>`` token.
            method: ``'greedy'`` (argmax, used as baseline) or ``'sample'``
                (multinomial sampling, used for exploration).

        Returns:
            Tuple of:
                seqs      – ``LongTensor  (B, max_len)`` — generated token IDs.
                log_probs – ``FloatTensor (B, max_len)`` — log-prob of each
                            selected token (used in the REINFORCE loss).
        """
        hidden  = encoder_hidden
        batch   = hidden[0].size(1)
        device  = hidden[0].device

        token = torch.full((batch, 1), start_idx, dtype=torch.long, device=device)

        seqs:      list[Tensor] = []
        log_probs: list[Tensor] = []
        unfinished = torch.ones(batch, dtype=torch.bool, device=device)

        for _ in range(max_len):
            logit, hidden = self.decode_step(token, hidden)   # (B, vocab_size)

            if method == "sample":
                predicted_token = torch.multinomial(
                    F.softmax(logit, dim=-1), num_samples=1
                )                                             # (B, 1)
            else:
                predicted_token = torch.argmax(logit, dim=-1, keepdim=True)  # (B, 1)

            # Gather log-prob of the selected token for the REINFORCE loss.
            step_log_probs     = F.log_softmax(logit, dim=-1)                   # (B, vocab_size)
            selected_log_probs = step_log_probs.gather(1, predicted_token)      # (B, 1)

            seqs.append(predicted_token)
            log_probs.append(selected_log_probs)

            # Force finished sequences to output padding (0) to avoid
            # accumulating log-probs after <end> is generated.
            token      = predicted_token * unfinished.unsqueeze(1).long()
            unfinished = unfinished & (predicted_token.squeeze(1) != end_idx)

        seqs      = torch.cat(seqs,      dim=1)  # (B, max_len)
        log_probs = torch.cat(log_probs, dim=1)  # (B, max_len)
        return seqs, log_probs


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    decoder = LSTMDecoder(vocab_size=3000, embed_size=512,
                          hidden_size=1024, num_layers=2)
    h = torch.zeros(2, 4, 1024)
    c = torch.zeros(2, 4, 1024)
    target = torch.randint(0, 3000, (4, 10))
    logits = decoder((h, c), target)
    print(f"logits: {logits.shape}")   # expect (4, 10, 3000)
