"""
Standardized collate functions for VQAGenerativeDataset.

VQABatch — dataclass replacing the (img, q, a) tuple and (feats, q, a, mask) 4-tuple.
           All downstream code (trainer, scst, inference) unpacks from VQABatch fields.

Two collate functions:
  image_collate_fn  — for image-based datasets (Models A-E, dev/test without BUTD)
  butd_collate_fn   — for BUTD feature datasets (Models F, G)

Both return VQABatch. The trainer checks `batch.img_mask is not None` to determine
whether BUTD or image mode — no model-type string dispatch needed.

label_tokens:
  Used by G2 three-way PGN. Each region i has a label name (e.g. "fire hydrant").
  Multi-word labels are tokenized → shared integer vocab index.
  Shape: (B, max_k, max_label_toks) — padded with 0 (<pad>).
  None if label_names were not extracted (most F checkpoints, pre-G2 features).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


# ---------------------------------------------------------------------------
# VQABatch
# ---------------------------------------------------------------------------

@dataclass
class VQABatch:
    """
    Standardized batch container for all VQA data sources.

    Fields:
      feats       (B, 3, H, W)      — images, OR
                  (B, max_k, D)     — BUTD RoI features (padded)
      questions   (B, max_q)        — padded question token indices
      targets     (B, max_t)        — padded target sequence token indices
      img_mask    (B, max_k) bool   — True = valid region; None for image batches
      length_bins (B,) int64        — 0/1/2 for G5 length conditioning; None if not used
      label_tokens (B, max_k, max_toks) int64 — G2 visual label indices; None if absent
      sources     List[str]         — per-sample source tag (for curriculum / logging)
      label_names per-sample BUTD label strings (SCST OHP); None for image batches
    """
    feats: Tensor
    questions: Tensor
    targets: Tensor
    img_mask: Optional[Tensor] = None
    length_bins: Optional[Tensor] = None
    label_tokens: Optional[Tensor] = None
    grid_feats: Optional[Tensor] = None
    grid_valid: Optional[Tensor] = None  # (B,) bool — False → encode ignores grid (uses v_proj only)
    sources: Optional[List[str]] = None
    label_names: Optional[List[Optional[List[str]]]] = None

    # -----------------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------------

    @property
    def batch_size(self) -> int:
        return self.feats.size(0)

    def is_butd(self) -> bool:
        """True if feats are BUTD RoI features (3D), False if raw images (4D)."""
        return self.feats.dim() == 3

    def to(self, device) -> "VQABatch":
        """Move all tensors to device. sources list stays on CPU."""
        def _maybe(t):
            return t.to(device) if t is not None else None

        return VQABatch(
            feats=self.feats.to(device),
            questions=self.questions.to(device),
            targets=self.targets.to(device),
            img_mask=_maybe(self.img_mask),
            length_bins=_maybe(self.length_bins),
            label_tokens=_maybe(self.label_tokens),
            grid_feats=_maybe(self.grid_feats),
            grid_valid=_maybe(self.grid_valid),
            sources=self.sources,
            label_names=self.label_names,
        )


# ---------------------------------------------------------------------------
# Label token building (for G2)
# ---------------------------------------------------------------------------

def _build_label_tokens(
    label_names_batch: List[Optional[List[str]]],
    a_vocab,
) -> Optional[Tensor]:
    """
    Convert per-sample label_names lists to a padded integer tensor.

    Args:
        label_names_batch : list of length B, each element is List[str] of length k_i
                            or None (feature file has no label metadata).
        a_vocab           : answer Vocabulary for tokenizing label names.

    Returns:
        Tensor (B, max_k, max_toks) int64, padded with 0 (<pad>),
        or None if ALL samples have label_names=None.

    Multi-word label "fire hydrant" → [fire_idx, hydrant_idx] in answer vocab.
    Unknown words map to <unk> (idx 3).
    """
    if all(ln is None for ln in label_names_batch):
        return None

    # Tokenize each label name into a list of vocab indices
    def _tok(name: str):
        return [
            a_vocab.word2idx.get(w, a_vocab.word2idx["<unk>"])
            for w in name.lower().split()
        ] or [a_vocab.word2idx["<unk>"]]

    # Build per-sample (k_i, max_toks_i) tensors
    sample_tensors = []
    for label_names in label_names_batch:
        if label_names is None:
            # No labels for this sample — single zero region as placeholder
            sample_tensors.append(torch.zeros(1, 1, dtype=torch.long))
        else:
            toks = [_tok(n) for n in label_names]
            max_t = max(len(t) for t in toks)
            mat = torch.zeros(len(toks), max_t, dtype=torch.long)
            for i, t in enumerate(toks):
                mat[i, :len(t)] = torch.tensor(t, dtype=torch.long)
            sample_tensors.append(mat)   # (k_i, max_t_i)

    # Pad along k and tok dimensions to (B, max_k, max_toks)
    max_k = max(t.size(0) for t in sample_tensors)
    max_t = max(t.size(1) for t in sample_tensors)

    B = len(sample_tensors)
    out = torch.zeros(B, max_k, max_t, dtype=torch.long)
    for b, t in enumerate(sample_tensors):
        k, nt = t.shape
        out[b, :k, :nt] = t

    return out


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def image_collate_fn(batch, a_vocab=None):
    """
    Collate for image-based VQAGenerativeDataset.
    Returns VQABatch with feats shape (B, 3, H, W).

    Args:
        batch   : list of (img_tensor, q_tensor, a_tensor, length_bin, label_names, source)
        a_vocab : answer Vocabulary — only needed for G2 label_tokens; pass None to skip.
    """
    imgs, grids, questions, answers, bins, label_names_list, sources = zip(*batch)

    feats     = torch.stack(imgs, dim=0)                         # (B, 3, H, W)
    questions = pad_sequence(questions, batch_first=True)        # (B, max_q)
    targets   = pad_sequence(answers,   batch_first=True)        # (B, max_t)
    length_bins = torch.tensor(bins, dtype=torch.long)           # (B,)

    label_tokens = None
    if a_vocab is not None:
        label_tokens = _build_label_tokens(list(label_names_list), a_vocab)

    return VQABatch(
        feats=feats,
        questions=questions,
        targets=targets,
        img_mask=None,
        length_bins=length_bins,
        label_tokens=label_tokens,
        grid_feats=None,
        sources=list(sources),
        label_names=None,
    )


def butd_collate_fn(batch, a_vocab=None):
    """
    Collate for BUTD-feature VQAGenerativeDataset.
    Returns VQABatch with feats shape (B, max_k, feat_dim).

    The img_mask field (B, max_k) bool is critical for:
      1. Masked attention: -inf padding in MHCA for invalid regions.
      2. Masked mean: global feature = sum(valid_regions) / count(valid_regions).
    Both bugs existed in pre-Step-B code when img_mask was None.

    Args:
        batch   : list of (feat_tensor, q_tensor, a_tensor, length_bin, label_names, source)
        a_vocab : answer Vocabulary — only needed for G2 label_tokens; pass None to skip.
    """
    feats_list, grids_list, questions, answers, bins, label_names_list, sources = zip(*batch)

    feats_padded = pad_sequence(feats_list, batch_first=True)    # (B, max_k, feat_dim)
    questions    = pad_sequence(questions,  batch_first=True)    # (B, max_q)
    targets      = pad_sequence(answers,    batch_first=True)    # (B, max_t)
    length_bins  = torch.tensor(bins, dtype=torch.long)          # (B,)

    # Grid: keep batch tensor even when some samples lack grid (zero-pad + grid_valid mask)
    grid_feats = None
    grid_valid = None
    if any(g is not None for g in grids_list):
        ref = next(g for g in grids_list if g is not None)
        D = ref.numel()
        rows = []
        valid_flags = []
        for g in grids_list:
            if g is None:
                rows.append(torch.zeros(D, dtype=ref.dtype))
                valid_flags.append(False)
            else:
                rows.append(g)
                valid_flags.append(True)
        grid_feats = torch.stack(rows, dim=0)
        grid_valid = torch.tensor(valid_flags, dtype=torch.bool)

    # Build img_mask: True = valid region (non-zero row), False = padding
    # abs().sum(-1) > 0 is robust to L2-normalised features
    img_mask = feats_padded.abs().sum(dim=-1) > 0                # (B, max_k) bool

    label_tokens = None
    if a_vocab is not None:
        label_tokens = _build_label_tokens(list(label_names_list), a_vocab)

    return VQABatch(
        feats=feats_padded,
        questions=questions,
        targets=targets,
        img_mask=img_mask,
        length_bins=length_bins,
        label_tokens=label_tokens,
        grid_feats=grid_feats,
        grid_valid=grid_valid,
        sources=list(sources),
        label_names=list(label_names_list),
    )


# ---------------------------------------------------------------------------
# Convenience: make_collate_fn (used by DataLoader)
# ---------------------------------------------------------------------------

def make_collate_fn(use_butd: bool, a_vocab=None):
    """
    Return the appropriate collate_fn as a partial with a_vocab bound.
    Usage: DataLoader(..., collate_fn=make_collate_fn(use_butd=True, a_vocab=vocab))
    """
    import functools
    fn = butd_collate_fn if use_butd else image_collate_fn
    return functools.partial(fn, a_vocab=a_vocab)
