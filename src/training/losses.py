"""
Autoregressive Masked Focal Loss — Tier D3 (corrected)
=======================================================

Replaces the flawed 1D class-weight approach (CrossEntropyLoss with
per-class static weights) for autoregressive seq2seq training.

THE FLAW OF STATIC CLASS WEIGHTS IN AUTOREGRESSIVE MODELS
----------------------------------------------------------
CrossEntropyLoss(weight=w) applies the same scalar w[c] to EVERY position
in the sequence that emits class c.  In VQA-E, a target sequence looks like:

  <start> yellow  because  the  banana  is  ripe  <end>
      1    4293      87      2   1041    6   904     2

Weight w[4293] (for "yellow") is 5.0 (rare answer word).
Weight w[87]   (for "because") is 0.1 (very common grammar word).

But "because" is the structural hinge of the VQA-E format.  Down-weighting
it at every position it appears — regardless of whether the model predicts it
correctly or not — causes the decoder to systematically drop it, producing
answers that lack the explanation clause entirely.

THE SOLUTION: FOCAL LOSS (position-aware dynamic weighting)
----------------------------------------------------------
Focal Loss is implicitly frequency-balanced: the model already predicts
common tokens ("because", "the") with high confidence, so p_t → 1 and
the focal factor (1-p_t)^γ → 0.  The loss is naturally suppressed for
easy/common tokens without an explicit weight tensor.  Rare/hard tokens
(minority answer words, unusual explanation constructions) produce low p_t,
so the focal factor is close to 1 and they receive full gradient signal.

Formula (applied per token, per position, then masked over padding):
    ce_t         = -log P(y_t | context)          ∈ ℝ+
    p_t          = exp(-ce_t)                      ∈ [0, 1]
    focal_loss_t = (1 - p_t)^γ * ce_t
    loss         = Σ_{t: target_t ≠ pad} focal_loss_t
                   ──────────────────────────────────────
                   |{t: target_t ≠ pad}|

γ = 2 (default) is standard from Lin et al. 2017 RetinaNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceFocalLoss(nn.Module):
    """
    Focal Loss for autoregressive sequence generation.

    Unlike CrossEntropyLoss(weight=...) which applies static class weights
    globally (wrong for temporal sequences), this loss dynamically focuses
    on hard tokens at each position.

    Args:
        gamma        : focusing parameter (default 2.0, standard from RetinaNet)
        pad_idx      : token index to ignore in loss and normalization (default 0)
        label_smoothing: optional label smoothing applied before focal weighting
    """

    def __init__(self, gamma: float = 2.0, pad_idx: int = 0,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma           = gamma
        self.pad_idx         = pad_idx
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (N, V)  — already flattened from (B, T, V) → (B*T, V)
            targets : (N,)    — already flattened from (B, T) → (B*T,)

        Returns:
            scalar focal loss
        """
        # Per-token CE without reduction — shape: (N,)
        # ignore_index=pad_idx sets ce_loss[pad] = 0.0 automatically
        ce_loss = F.cross_entropy(
            logits, targets,
            ignore_index=self.pad_idx,
            label_smoothing=self.label_smoothing,
            reduction='none',
        )

        # p_t = model's confidence in the correct token
        # High p_t (easy / common token) → (1-p_t)^γ → 0 → suppressed
        # Low p_t  (hard / rare  token)  → (1-p_t)^γ → 1 → full gradient
        p_t          = torch.exp(-ce_loss)                     # (N,)
        focal_weight = (1.0 - p_t) ** self.gamma               # (N,)
        focal_loss   = focal_weight * ce_loss                  # (N,)

        # Mask padding tokens for the normalization denominator
        mask = (targets != self.pad_idx).float()               # (N,)

        return (focal_loss * mask).sum() / mask.sum().clamp(min=1.0)
