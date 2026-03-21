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


# ---------------------------------------------------------------------------
# FocalSequenceLoss — per-sample T-normalization (Model G, Eq 43-44)
# ---------------------------------------------------------------------------

class FocalSequenceLoss(nn.Module):
    """
    Per-token normalized Focal loss with label smoothing. (Model G training)

    Key difference from SequenceFocalLoss:
      - Accepts (B, T, V) directly (no pre-flattening)
      - Per-SAMPLE T normalization (not global batch normalization):
            L_sample = sum_t(focal_t) / T_valid
            L_batch  = mean(L_sample)
        This ensures a VQA v2.0 sample (T=3) and a VQA-E sample (T=25)
        contribute equal gradient magnitude.
      - Auto-detects log-probs input (from ThreeWayPGNHead.blend_3way)

    Args:
        gamma          : Focal exponent (default 2.0 per spec)
        label_smoothing: eps (default 0.1 per spec)
        ignore_index   : Pad token (default 0 = <pad>)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        ignore_index: int = 0,
    ):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.ignore_index    = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits  : (B, T, V) — raw logits OR log-probs (auto-detected)
            targets : (B, T) — token indices; ignore_index = pad

        Returns:
            scalar loss
        """
        B, T, V = logits.shape

        # Auto-detect: log-probs have max ≤ 0
        if logits.max().item() <= 0.01:
            log_p = logits
        else:
            log_p = F.log_softmax(logits, dim=-1)   # (B, T, V)

        # Gather log p_t for ground-truth token
        tgt_clamp = targets.clamp(min=0)   # (B, T)
        log_p_t   = log_p.gather(2, tgt_clamp.unsqueeze(2)).squeeze(2)  # (B, T)
        p_t       = log_p_t.exp()                                        # (B, T)

        # Label smoothing: smoothed loss = -(1-eps)*log_p_t - (eps/V)*sum_w log_p_w
        if self.label_smoothing > 0.0:
            smooth_loss = -(
                (1.0 - self.label_smoothing) * log_p_t
                + (self.label_smoothing / V) * log_p.sum(dim=-1)
            )   # (B, T)
            p_t_focal = (
                (1.0 - self.label_smoothing) * p_t
                + self.label_smoothing / V
            ).detach()
        else:
            smooth_loss = -log_p_t                 # (B, T)
            p_t_focal   = p_t.detach()

        # Focal weight
        focal_weight = (1.0 - p_t_focal) ** self.gamma   # (B, T)
        token_loss   = focal_weight * smooth_loss         # (B, T)

        # Padding mask
        mask  = (targets != self.ignore_index).float()    # (B, T)
        token_loss = token_loss * mask

        # Per-sample T normalization → batch mean
        counts      = mask.sum(dim=1).clamp(min=1.0)     # (B,)
        sample_loss = token_loss.sum(dim=1) / counts     # (B,)
        return sample_loss.mean()


def build_criterion(
    gamma: float = 2.0,
    label_smoothing: float = 0.1,
    use_focal: bool = True,
    ignore_index: int = 0,
) -> nn.Module:
    """
    Factory for Model G sequence loss criterion.

    Phase 1-3: use_focal=True  (FocalSequenceLoss, gamma=2.0, eps=0.1)
    Phase 4:   use_focal=False (plain NLL with label smoothing, per spec)

    Returns callable (B,T,V), (B,T) → scalar.
    """
    if use_focal:
        return FocalSequenceLoss(
            gamma=gamma,
            label_smoothing=label_smoothing,
            ignore_index=ignore_index,
        )
    # Phase 4: plain CE with label smoothing
    # nn.CrossEntropyLoss expects (B,V,T) target convention — wrap in adapter
    return _PlainCEWrapper(label_smoothing=label_smoothing, ignore_index=ignore_index)


class _PlainCEWrapper(nn.Module):
    """Adapter: accept (B,T,V),(B,T) → (B,V,T) for nn.CrossEntropyLoss."""

    def __init__(self, label_smoothing: float = 0.0, ignore_index: int = 0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=ignore_index,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B,T,V) → permute to (B,V,T) for CrossEntropyLoss
        return self.ce(logits.permute(0, 2, 1), targets)
