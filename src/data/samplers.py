"""
Samplers for multi-source VQA training.

build_mixed_sampler  — N-source WeightedRandomSampler (replaces 2-source version
                       in old src/dataset.py, now supports arbitrary source counts)
build_replay_sampler — Phase 2/3 experience replay: explanation data + VQA v2.0 buffer

Data mix per phase (from Architecture_Specification_v2.md):
  Phase 1: 40% VQA v2.0 + 30% VQA-E + 30% A-OKVQA  → 3-source mixed sampler
  Phase 2: 100% explanation + 20% VQA v2.0 replay    → replay sampler
  Phase 3: same as Phase 2 + scheduled sampling      → replay sampler
  Phase 4: VQA-E + VQA-X only (CIDEr requires refs)  → standard DataLoader, no sampler

CurriculumSampler integration: the existing src/training/curriculum.py
CurriculumSampler is a separate concern (reorders by complexity within a source).
build_mixed_sampler operates across sources; CurriculumSampler operates within one.
They compose: use CurriculumSampler per-source then build_mixed_sampler across sources.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import torch
from torch.utils.data import ConcatDataset, Dataset, Subset, WeightedRandomSampler


# ---------------------------------------------------------------------------
# build_mixed_sampler
# ---------------------------------------------------------------------------

def build_mixed_sampler(
    datasets: List[Dataset],
    fractions: List[float],
    num_samples: Optional[int] = None,
) -> Tuple[ConcatDataset, WeightedRandomSampler]:
    """
    Build a ConcatDataset + WeightedRandomSampler mixing N datasets at controlled ratios.

    Each sample's weight = desired_fraction / dataset_size.
    All samples within the same source share identical weight.

    Args:
        datasets  : list of Dataset objects, one per source (e.g. [vqa_v2, vqa_e, aokvqa])
        fractions : desired fraction of each batch from each source.
                    Does NOT need to sum to 1.0 — will be renormalized internally.
                    Example: [0.40, 0.30, 0.30] for Phase 1 mix.
        num_samples : total samples drawn per epoch.
                      Default: sum of all dataset lengths (1 pass through largest).

    Returns:
        concat  : ConcatDataset([d0, d1, ...])
        sampler : WeightedRandomSampler with per-sample weights

    Example:
        concat, sampler = build_mixed_sampler(
            [vqa_v2_ds, vqae_ds, aokvqa_ds],
            fractions=[0.40, 0.30, 0.30],
        )
        loader = DataLoader(concat, batch_size=192, sampler=sampler, ...)
    """
    assert len(datasets) == len(fractions), \
        f"datasets ({len(datasets)}) and fractions ({len(fractions)}) must have same length"
    assert all(len(d) > 0 for d in datasets), "all datasets must be non-empty"
    assert all(f > 0.0 for f in fractions), "all fractions must be positive"

    # Renormalize fractions to sum to 1.0
    total_frac = sum(fractions)
    fractions = [f / total_frac for f in fractions]

    weights: List[float] = []
    for ds, frac in zip(datasets, fractions):
        w = frac / len(ds)
        weights.extend([w] * len(ds))

    total = num_samples or sum(len(d) for d in datasets)
    concat = ConcatDataset(datasets)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=total,
        replacement=True,
    )
    return concat, sampler


# ---------------------------------------------------------------------------
# build_replay_sampler
# ---------------------------------------------------------------------------

def build_replay_sampler(
    explanation_dataset: Dataset,
    replay_dataset: Dataset,
    replay_fraction: float = 0.2,
    num_samples: Optional[int] = None,
) -> Tuple[ConcatDataset, WeightedRandomSampler]:
    """
    Build a sampler for Phases 2 and 3: explanation data + experience replay.

    Experience replay prevents catastrophic forgetting of basic QA ability
    when dropping VQA v2.0 from 40% (Phase 1) to 0% (Phase 2).

    The replay buffer is a FIXED random subset of the replay_dataset (VQA v2.0),
    mixed into every epoch at `replay_fraction` rate.

    Args:
        explanation_dataset : dataset of explanation samples (VQA-E + VQA-X + A-OKVQA)
        replay_dataset      : full VQA v2.0 dataset (or any replay source)
        replay_fraction     : fraction of each batch from replay (default 0.2 = 20%)
        num_samples         : epoch length; defaults to len(explanation_dataset)

    Returns:
        concat  : ConcatDataset([explanation_dataset, replay_subset])
        sampler : WeightedRandomSampler

    Note: replay_dataset samples receive length_bin=SHORT automatically because
    they are VQA v2.0 samples with 1-3 token answers (short answers → SHORT bin
    computed in VQAGenerativeDataset.__getitem__).
    """
    assert 0.0 < replay_fraction < 1.0, "replay_fraction must be in (0, 1)"
    expl_fraction = 1.0 - replay_fraction

    return build_mixed_sampler(
        datasets=[explanation_dataset, replay_dataset],
        fractions=[expl_fraction, replay_fraction],
        num_samples=num_samples or len(explanation_dataset),
    )


# ---------------------------------------------------------------------------
# build_phase4_loader_kwargs
# ---------------------------------------------------------------------------

def build_phase4_sources(
    merged_dataset: Dataset,
    allowed_sources: Optional[List[str]] = None,
) -> Dataset:
    """
    Phase 4 (SCST RL) uses only VQA-E + VQA-X because CIDEr-D requires
    reference sentences. A-OKVQA rationales are excluded (see Data Strategy §6.4).

    If merged_dataset is a VQAGenerativeDataset, use its filter_by_source().
    Otherwise returns merged_dataset unchanged.

    Default allowed_sources = ['vqa_e', 'vqa_x']
    """
    if allowed_sources is None:
        allowed_sources = ["vqa_e", "vqa_x"]

    if hasattr(merged_dataset, "filter_by_source"):
        return merged_dataset.filter_by_source(allowed_sources)

    return merged_dataset
