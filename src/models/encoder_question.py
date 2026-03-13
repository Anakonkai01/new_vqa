"""BiLSTM question encoder for the VQA project.

Encodes a tokenized question into a fixed-size feature vector and a sequence
of hidden states (used by the dual-attention decoder in Models C/D/E).

Shape contract::

    input : (B, Q)           — question token indices, padded with 0
    output: (B, hidden_size) — last-layer BiLSTM hidden state (concat fwd+bwd)
            (B, Q, hidden_size) — all BiLSTM hidden states (for question attention)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class QuestionEncoder(nn.Module):
    """Bidirectional LSTM question encoder.

    Converts a padded sequence of question token indices into:

    1. A **global feature vector** ``q_feature`` of shape ``(B, hidden_size)``
       formed by concatenating the final forward and backward hidden states.
    2. A **sequence of hidden states** ``q_hidden_states`` of shape
       ``(B, Q, hidden_size)`` used by the dual-attention decoder.

    **BiLSTM sizing**: Each direction uses ``hidden_size // 2`` units so that
    concatenating forward + backward yields exactly ``hidden_size``.

    Optionally initializes the embedding layer with pre-trained GloVe vectors.
    If the GloVe dimension differs from ``embed_size``, a learned linear
    projection bridges the gap before the LSTM.

    Args:
        vocab_size: Size of the question vocabulary.
        embed_size: Embedding dimension (input to the LSTM).
        hidden_size: Total hidden size after BiLSTM (each direction = hidden_size // 2).
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability applied to embeddings and between LSTM layers.
        pretrained_embeddings: Optional ``FloatTensor (vocab_size, glove_dim)``
            of pre-trained GloVe vectors.
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
            # If GloVe dim ≠ embed_size, add a learned bridge projection.
            self.embed_proj: Optional[nn.Linear] = (
                nn.Linear(glove_dim, embed_size) if glove_dim != embed_size else None
            )
        else:
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=0
            )
            self.embed_proj = None

        # ── BiLSTM ───────────────────────────────────────────────────────────
        # Each direction has hidden_size // 2 units.
        # Concatenating forward[-1] + backward[-1] → (B, hidden_size).
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, questions: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode a batch of padded question token sequences.

        Args:
            questions: ``LongTensor (B, Q)`` — padded question token indices.

        Returns:
            Tuple of:
                q_feature       – ``FloatTensor (B, hidden_size)``
                                  Global question representation (fwd + bwd concat).
                q_hidden_states – ``FloatTensor (B, Q, hidden_size)``
                                  All BiLSTM timestep outputs (for question attention).
        """
        # (B, Q) → (B, Q, glove_dim or embed_size)
        embeds = self.dropout(self.embedding(questions))

        if self.embed_proj is not None:
            embeds = self.embed_proj(embeds)  # (B, Q, glove_dim) → (B, Q, embed_size)

        # output : (B, Q, hidden_size)          — concat of fwd+bwd at each step
        # hidden : (num_layers*2, B, hidden_size//2) — per-direction last hidden states
        output, (hidden, _cell) = self.lstm(embeds)

        # Concatenate the last forward (hidden[-2]) and last backward (hidden[-1])
        # hidden states from the top layer to form the global question feature.
        #   hidden[-2]: (B, hidden_size//2)  — forward  last layer
        #   hidden[-1]: (B, hidden_size//2)  — backward last layer
        q_feature = torch.cat([hidden[-2], hidden[-1]], dim=1)
        # q_feature: (B, hidden_size//2 + hidden_size//2) = (B, hidden_size)

        return q_feature, output  # output = q_hidden_states: (B, Q, hidden_size)


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = QuestionEncoder(vocab_size=7000, embed_size=512,
                            hidden_size=1024, num_layers=2, dropout=0.5)
    q = torch.randint(0, 7000, (4, 20))
    q_feat, q_hidden = model(q)
    print(f"q_feature      : {q_feat.shape}")   # expect (4, 1024)
    print(f"q_hidden_states: {q_hidden.shape}")  # expect (4, 20, 1024)
