"""
Base protocols and output types for Model G.

VQAOutput       — standardized return type from all VQAModel.forward() calls.
DecoderProtocol — ABC contract for LSTM decoders (A/B: no attn, C-G: MHCA).
VQAModelProtocol — ABC contract for top-level VQAModel wrappers.

Design notes:
- VQAOutput uses Optional fields so downstream code checks `if output.attention_img is not None`
  rather than branching on model type. Zero overhead when field is None.
- infonce_z carries (z_img, z_text) needed by G3 InfoNCE loss in the trainer;
  these projections are discarded at inference (model.eval() path skips them).
- DecoderProtocol.decode_step() is the single-step interface used by inference.py
  for greedy/beam search — both with and without attention.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# VQAOutput — standardized forward-pass return
# ---------------------------------------------------------------------------

@dataclass
class VQAOutput:
    """
    Unified output container from VQAModel.forward().

    Fields present for ALL models:
      logits        (B, T, |V_A|) or (B, T, |V_ext|) when PGN active

    Fields present for attention models (C-G):
      attention_img (B, T, k)   — head-averaged image spatial attention
      attention_q   (B, T, L)   — head-averaged question attention
      coverage      (B, k)      — accumulated coverage after full sequence

    Fields present when PGN active (F, G):
      pgn_weights   (B, T, 2)   — [p_gen, p_copy_Q]           (2-way, Model F)
                    (B, T, 3)   — [p_gen, p_copy_Q, p_copy_V]  (3-way, G2)

    Fields present when InfoNCE active (G3, training only):
      infonce_z     Tuple[(B, d_z), (B, d_z)] = (z_img, z_text)
      Trainer checks `if output.infonce_z is not None` to compute L_InfoNCE.
      Always None during model.eval().
    """

    # Primary output — always present
    logits: Tensor                                              # (B, T, V)

    # Attention (C-G)
    attention_img: Optional[Tensor] = None                     # (B, T, k)
    attention_q: Optional[Tensor] = None                       # (B, T, L)
    coverage: Optional[Tensor] = None                          # (B, k)

    # Pointer-Generator (F, G)
    pgn_weights: Optional[Tensor] = None                       # (B, T, 2 or 3)

    # InfoNCE projections (G3, training only)
    infonce_z: Optional[Tuple[Tensor, Tensor]] = None          # (z_img, z_text)
    # Model H: symmetric InfoNCE computed inside forward_with_cov (single encode)
    infonce_loss: Optional[Tensor] = None

    # -----------------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------------

    @property
    def batch_size(self) -> int:
        return self.logits.size(0)

    @property
    def seq_len(self) -> int:
        return self.logits.size(1)

    @property
    def vocab_size(self) -> int:
        return self.logits.size(2)

    def has_attention(self) -> bool:
        return self.attention_img is not None

    def has_pgn(self) -> bool:
        return self.pgn_weights is not None

    def has_infonce(self) -> bool:
        return self.infonce_z is not None

    def detach_aux(self) -> "VQAOutput":
        """Detach non-logit fields for logging/visualization (no grad needed)."""
        return VQAOutput(
            logits=self.logits,
            attention_img=self.attention_img.detach() if self.attention_img is not None else None,
            attention_q=self.attention_q.detach() if self.attention_q is not None else None,
            coverage=self.coverage.detach() if self.coverage is not None else None,
            pgn_weights=self.pgn_weights.detach() if self.pgn_weights is not None else None,
            infonce_z=None,  # never needed after loss computation
        )


# ---------------------------------------------------------------------------
# DecoderProtocol
# ---------------------------------------------------------------------------

class DecoderProtocol(ABC):
    """
    Contract for LSTM decoder modules (LSTMDecoder and LSTMDecoderWithAttention).

    forward()     — teacher-forced training pass over full target sequence.
    decode_step() — single autoregressive step for greedy/beam inference.

    Both methods must be implemented. Attention-free decoders (A/B) return
    None for attention and coverage outputs.
    """

    @abstractmethod
    def forward(
        self,
        targets: Tensor,                        # (B, T) token indices (teacher forcing)
        encoder_out: Tensor,                    # (B, k, H) visual features OR (B, H) global
        q_hidden: Tensor,                       # (B, H) pooled question feature
        q_seq: Optional[Tensor] = None,         # (B, L, H) question hidden sequence (MHCA)
        img_mask: Optional[Tensor] = None,      # (B, k) bool, True = valid region
        h0: Optional[Tensor] = None,            # (num_layers, B, H) initial hidden
        c0: Optional[Tensor] = None,            # (num_layers, B, H) initial cell
        length_bin: Optional[Tensor] = None,    # (B,) int64 in {0,1,2} for G5
        label_tokens: Optional[Tensor] = None,  # (B, k, max_toks) for G2 visual copy
    ) -> VQAOutput:
        """
        Returns VQAOutput with:
          logits        (B, T, V)
          attention_img (B, T, k)  or None
          attention_q   (B, T, L)  or None
          coverage      (B, k)     or None
          pgn_weights   (B, T, 2/3) or None
        """
        ...

    @abstractmethod
    def decode_step(
        self,
        token: Tensor,                          # (B,) current input token
        h: Tensor,                              # (num_layers, B, H)
        c: Tensor,                              # (num_layers, B, H)
        encoder_out: Tensor,                    # (B, k, H) or (B, H)
        q_seq: Optional[Tensor] = None,         # (B, L, H)
        coverage: Optional[Tensor] = None,      # (B, k) accumulated coverage
        img_mask: Optional[Tensor] = None,      # (B, k) bool
        length_bin: Optional[Tensor] = None,    # (B,) int64
        label_tokens: Optional[Tensor] = None,  # (B, k, max_toks)
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Single decode step.

        Returns:
          logits_t      (B, V)       — unnormalized scores for this step
          h_new         (num_layers, B, H)
          c_new         (num_layers, B, H)
          attn_img_t    (B, k)       or None
          attn_q_t      (B, L)       or None
        """
        ...


# ---------------------------------------------------------------------------
# VQAModelProtocol
# ---------------------------------------------------------------------------

class VQAModelProtocol(ABC):
    """
    Contract for top-level VQAModel wrappers (VQAModelA through VQAModelG).

    encode()  — runs image + question encoders, returns raw features.
    forward() — full forward pass (encode + fuse + decode), returns VQAOutput.

    The encode() / decode() split is critical for:
    - SCST: encode once, sample + greedy decode twice without re-encoding
    - inference.py: encode_and_init() reuse across beam hypotheses
    """

    @abstractmethod
    def encode(
        self,
        images_or_feats: Tensor,                # (B, 3, H, W) images OR (B, k, roi_dim+geo) feats
        questions: Tensor,                      # (B, L) token indices
        img_mask: Optional[Tensor] = None,      # (B, k) bool, True = valid region
        char_seqs: Optional[Tensor] = None,     # (B, L, max_char) for CharCNN
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
          V         (B, k, H)  — projected + L2-normed visual features (spatial)
                    (B, H)     — global visual feature (non-spatial models A/B)
          q_feat    (B, H)     — attention-pooled question summary
          Q_H       (B, L, H)  — full BiLSTM hidden sequence (for MHCA)
        """
        ...

    @abstractmethod
    def forward(
        self,
        images_or_feats: Tensor,
        questions: Tensor,
        targets: Tensor,                        # (B, T) for teacher forcing
        img_mask: Optional[Tensor] = None,
        char_seqs: Optional[Tensor] = None,
        length_bin: Optional[Tensor] = None,    # (B,) G5 length bin
        label_tokens: Optional[Tensor] = None,  # (B, k, max_toks) G2 visual labels
    ) -> VQAOutput:
        """Full forward pass. Returns VQAOutput."""
        ...
