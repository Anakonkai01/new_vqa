"""CLIP Text Encoder — drop-in replacement for BiLSTM QuestionEncoder (Model F+).

Why CLIP text > BiLSTM
----------------------
CLIP was trained with an InfoNCE contrastive objective on 400M image-text pairs::

    L = -log [ exp(sim(I_i, T_i) / τ) / Σ_j exp(sim(I_i, T_j) / τ) ]

where ``sim(I, T) = cosine(I, T)``.  This forces the image and text encoders to
produce embeddings in the **same pre-aligned space** — without any VQA fine-tuning,
``img_feat · text_feat`` is already a meaningful cross-modal similarity.

The FiLM generator ``MLP(q_feat)`` in Model E/F therefore starts from a
representation that already "understands" visual concepts, making the modulation
signal richer from step 0.

Additionally, the CLIP text transformer's self-attention has O(1) gradient path
between any two tokens (no vanishing-gradient product of Jacobians as in BiLSTM).

API
---
Matches ``QuestionEncoder.forward()`` exactly — drop-in replacement::

    q_feat, q_hidden_states = encoder(questions)
    # q_feat          : (B, hidden_size)    — pooled [EOS] representation
    # q_hidden_states : (B, 77, hidden_size) — all token representations
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from transformers import CLIPTextModel


class CLIPTextEncoder(nn.Module):
    """CLIP text transformer encoder for VQA question encoding.

    Wraps ``openai/clip-vit-base-patch32``'s text transformer, which outputs
    512-dimensional features.  A linear projection maps these to ``hidden_size``
    (1024 by default) to match the rest of the VQA pipeline.

    The full-sequence hidden states ``(B, 77, hidden_size)`` are returned
    alongside the pooled EOS representation, enabling Models G and H to apply
    cross-attention over all question tokens.

    Args:
        hidden_size: Target feature dimension for the VQA pipeline (default 1024).
        freeze: If True, freeze the CLIP text backbone weights.  Only the
            linear projection ``text_proj`` is trained during Phase 1.
    """

    def __init__(self, hidden_size: int = 1024, freeze: bool = True) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        # Load pre-trained CLIP text transformer.
        self.clip_text = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

        if freeze:
            for p in self.clip_text.parameters():
                p.requires_grad_(False)

        # CLIP text output dim is 512 for ViT-B/32.
        clip_text_dim = self.clip_text.config.hidden_size  # 512

        # Single linear projection — keeps the pipeline to ``hidden_size``.
        self.text_proj = nn.Linear(clip_text_dim, hidden_size)

    def forward(self, input_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode a batch of tokenised questions.

        Args:
            input_ids: ``LongTensor (B, 77)`` — CLIP BPE token ids, fixed length 77.
                Produced by ``CLIPProcessor`` / ``VQAEDatasetCLIP``.

        Returns:
            Tuple of:
                q_feat          – ``FloatTensor (B, hidden_size)`` — pooled EOS
                                  representation, used as the FiLM modulation signal
                                  and the LSTM decoder's initial state.
                q_hidden_states – ``FloatTensor (B, 77, hidden_size)`` — per-token
                                  representations, used by cross-attention in G and H.
        """
        # attention_mask is all-ones: CLIP always pads to exactly 77 tokens and
        # the EOS token at position 76 is the designated pooling position.
        attention_mask = (input_ids != 0).long()  # 0 is the CLIP pad id

        outputs = self.clip_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # last_hidden_state : (B, 77, 512)
        # pooler_output      : (B, 512) — features at the EOS position

        last_hidden = outputs.last_hidden_state   # (B, 77, 512)
        pooled      = outputs.pooler_output       # (B, 512)

        # Project both to pipeline hidden_size.
        q_feat          = self.text_proj(pooled)      # (B, hidden_size)
        q_hidden_states = self.text_proj(last_hidden) # (B, 77, hidden_size)

        return q_feat, q_hidden_states

    def unfreeze_top_layers(self, n: int = 2) -> None:
        """Unfreeze the top ``n`` transformer encoder layers for Phase 2 fine-tuning.

        Args:
            n: Number of top layers to unfreeze (counted from the last layer).
        """
        layers = self.clip_text.text_model.encoder.layers
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad_(True)


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, H = 4, 1024
    encoder = CLIPTextEncoder(hidden_size=H, freeze=True)

    # Simulate CLIP-tokenised input (77 tokens, BOS=49406 EOS=49407 PAD=0)
    input_ids = torch.zeros(B, 77, dtype=torch.long)
    input_ids[:, 0]  = 49406  # BOS
    input_ids[:, 5]  = 49407  # EOS after 4 content tokens
    input_ids[:, 1:5] = torch.randint(1000, 5000, (B, 4))

    q_feat, q_hidden = encoder(input_ids)
    print(f"q_feat          : {q_feat.shape}")    # expect (4, 1024)
    print(f"q_hidden_states : {q_hidden.shape}")  # expect (4, 77, 1024)
