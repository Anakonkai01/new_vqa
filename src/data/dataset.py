"""
VQAGenerativeDataset — single Dataset class replacing the 3-class inheritance hierarchy.

Design: composition over inheritance.
  - One class handles all data sources (merged JSON, VQA v2.0, legacy VQA-E)
  - Feature loading is pluggable via a `feature_loader` callable
  - No inheritance hack (old BUTDDataset(VQAEDataset) with image_dir='' is gone)

Feature loaders (returned by factory functions below):
  make_image_loader(image_dir, split, augment)  →  f(img_id) → Tensor (3, 224, 224)
  make_butd_loader(feat_dir)                    →  f(img_id) → Tensor (k, feat_dim)

Fixes applied vs old dataset.py:
  - Bug fix: length_bin recomputed from FULL sequence (answer + because + explanation)
    not explanation-only. Fixes G5 length conditioning (20.0% LONG, not 9.1%).
  - label_tokens: extracted from BUTD .pt file when 'labels'/'label_names' key present (G2).
  - A-OKVQA multi-rationale: explanation list → pick 1 randomly per epoch (already in old code,
    but now consistent across all sources via unified annotation format).
  - VQA v2.0 format: factory classmethod from_vqa_v2() normalizes to unified format inline.

Backward compat: old src/dataset.py is untouched. train.py still imports from there.
New trainer (Step D) imports from src/data/dataset.py.
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Words indicating spatial orientation — flipping would change the answer
SPATIAL_KEYWORDS = frozenset({
    'left', 'right', 'east', 'west', 'leftmost', 'rightmost',
    'lefthand', 'righthand', 'clockwise', 'counterclockwise',
})

# Length bin indices for G5 length-conditioned decoding
LENGTH_BIN_SHORT  = 0   # full sequence ≤ 5 tokens
LENGTH_BIN_MEDIUM = 1   # full sequence 6–14 tokens
LENGTH_BIN_LONG   = 2   # full sequence ≥ 15 tokens  ← always used at inference


def _full_seq_length_bin(answer: str, explanation: str) -> int:
    """
    Compute length bin from FULL target sequence: '{answer} because {explanation}'.

    Fixes the known discrepancy in merged_train_filtered.json where length_bin
    was computed from explanation-only word count (avg 10.5w, 9.1% LONG) instead
    of the full sequence (avg 12.6w, 20.0% LONG).

    The LSTM decoder generates the full sequence, so the bin must match it.
    We add +2 for 'because' (1 word) and +1 conservative for tokenization overhead.
    """
    if explanation:
        full_len = len(answer.split()) + 1 + len(explanation.split())  # 'because' counts as 1
    else:
        full_len = len(answer.split())

    if full_len <= 5:
        return LENGTH_BIN_SHORT
    elif full_len <= 14:
        return LENGTH_BIN_MEDIUM
    else:
        return LENGTH_BIN_LONG


# ---------------------------------------------------------------------------
# Feature loader factories
# ---------------------------------------------------------------------------

def make_image_loader(
    image_dir: str,
    split: str = "train2014",
    augment: bool = False,
) -> Callable[[int], torch.Tensor]:
    """
    Returns a callable f(img_id) → Tensor(3, 224, 224).
    Augmentation includes RandomResizedCrop + RandAugment + spatial-aware flip.

    Automatically tries both train2014/ and val2014/ sub-directories because
    A-OKVQA uses COCO 2017 images which are distributed across both COCO 2014 splits.
    Search order: requested split first, then the other split as fallback.
    """
    if augment:
        _transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        ])
    else:
        _transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    # Determine parent dir and both split candidates
    _parent = image_dir if not image_dir.endswith(("train2014", "val2014")) else os.path.dirname(image_dir)
    _primary_split = split
    _fallback_split = "val2014" if split == "train2014" else "train2014"

    def _load(img_id: int, q_text: str = "") -> torch.Tensor:
        # Try primary split first, then fallback (handles A-OKVQA COCO-2017 images)
        for sp in (_primary_split, _fallback_split):
            img_name = f"COCO_{sp}_{img_id:012d}.jpg"
            img_path = os.path.join(_parent, sp, img_name)
            if os.path.exists(img_path):
                break
        else:
            raise FileNotFoundError(
                f"Image {img_id} not found in {_parent}/train2014/ or {_parent}/val2014/"
            )
        image = Image.open(img_path).convert("RGB")
        # Spatial-aware flip guard
        if augment:
            words = set(q_text.lower().split())
            if not (words & SPATIAL_KEYWORDS) and random.random() < 0.5:
                image = TF.hflip(image)
        return _transform(image)

    return _load


def unpack_stored_visual_features(data: dict) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[str]], str]:
    """
    Normalize .pt payloads from:
      - extract_features_model_h.py → keys region_feat, optional grid_feat, label_names
      - extract_features_model_f.py → key feat only (Model F legacy), no grid / labels

    Returns:
        feat   : (k, D) region+spatial concatenated
        grid   : 1D global vector or None
        labels : per-RoI class names or None
        fmt    : 'model_h_vg' | 'legacy_butd_feat'
    """
    labels = data.get("label_names", None)
    if isinstance(data.get("region_feat"), torch.Tensor):
        feat = data["region_feat"]
        grid = data.get("grid_feat", None)
        if feat.dim() != 2:
            raise ValueError(f"region_feat must be 2D (k, D), got {tuple(feat.shape)}")
        fmt = "model_h_vg"
    elif isinstance(data.get("feat"), torch.Tensor):
        feat = data["feat"]
        grid = None
        if feat.dim() != 2:
            raise ValueError(f"feat must be 2D (k, D), got {tuple(feat.shape)}")
        fmt = "legacy_butd_feat"
    else:
        raise KeyError(
            "Feature .pt must contain tensor 'region_feat' (Model H) or 'feat' (legacy BUTD extract)"
        )
    return feat, grid, labels, fmt


def audit_butd_feature_dir(
    feat_dir: str,
    *,
    max_files: int = 512,
    seed: int = 42,
    raise_on_mismatch: bool = True,
) -> Dict[str, Any]:
    """
    Sample .pt files under feat_dir and verify a single region feature width D and consistent grid layout.

    Model H expects one D everywhere (e.g. 1029 = 1024 box + 5 spatial from extract_features_model_h).
    Mixing with extract_features_model_f (1031 = 1024 + 7 spatial) in the same folder breaks Linear layers.

    Uses weights_only=False for robust reading of dicts that may contain label_names (str lists).
    """
    import glob as _glob

    paths = sorted(_glob.glob(os.path.join(feat_dir, "*.pt")))
    n_total = len(paths)
    if n_total == 0:
        return {
            "n_total": 0,
            "n_scanned": 0,
            "region_dim": None,
            "grid_dim": None,
            "format_counts": {},
            "issues": [f"No .pt files in {feat_dir!r}"],
        }

    rng = random.Random(seed)
    sample_paths = list(paths)
    if len(sample_paths) > max_files:
        rng.shuffle(sample_paths)
        sample_paths = sample_paths[:max_files]

    dim_counts: Counter = Counter()
    grid_counts: Counter = Counter()
    fmt_counts: Counter = Counter()
    issues: List[str] = []
    example_by_dim: Dict[int, str] = {}

    for p in sample_paths:
        try:
            data = torch.load(p, map_location="cpu", weights_only=False)
            if not isinstance(data, dict):
                issues.append(f"Non-dict in {os.path.basename(p)}")
                continue
            feat, grid, _labels, fmt = unpack_stored_visual_features(data)
            d = int(feat.shape[1])
            dim_counts[d] += 1
            if d not in example_by_dim:
                example_by_dim[d] = os.path.basename(p)
            fmt_counts[fmt] += 1
            if grid is not None and grid.dim() >= 1:
                grid_counts[int(grid.shape[0])] += 1
            else:
                grid_counts[None] += 1  # type: ignore[arg-type]
        except Exception as e:
            issues.append(f"{os.path.basename(p)}: {e}")

    if not dim_counts:
        msg = "No valid feature tensors parsed in sample (check .pt format and keys)."
        issues.append(msg)
        out = {
            "n_total": n_total,
            "n_scanned": len(sample_paths),
            "region_dim": None,
            "grid_dim": None,
            "format_counts": dict(fmt_counts),
            "dim_histogram": {},
            "grid_histogram": {},
            "issues": issues,
        }
        if raise_on_mismatch:
            raise ValueError(msg)
        return out

    if len(dim_counts) > 1:
        msg = (
            f"Inconsistent region feature width D across sample: {dict(dim_counts)}. "
            f"Examples: { {d: example_by_dim.get(d) for d in dim_counts} }. "
            "Do not mix extract_features_model_h (typically D=1029) with extract_features_model_f (D=1031) "
            "in the same directory."
        )
        issues.append(msg)
        if raise_on_mismatch:
            raise ValueError(msg)
        return {
            "n_total": n_total,
            "n_scanned": len(sample_paths),
            "region_dim": None,
            "grid_dim": None,
            "format_counts": dict(fmt_counts),
            "dim_histogram": dict(dim_counts),
            "grid_histogram": {str(k): v for k, v in grid_counts.items()},
            "issues": issues,
        }

    nonnull_grid_dims = [g for g in grid_counts if g is not None]
    if len(nonnull_grid_dims) > 1:
        msg = f"Inconsistent grid_feat channel counts: {dict(grid_counts)}"
        issues.append(msg)
        if raise_on_mismatch:
            raise ValueError(msg)
        return {
            "n_total": n_total,
            "n_scanned": len(sample_paths),
            "region_dim": int(dim_counts.most_common(1)[0][0]),
            "grid_dim": None,
            "format_counts": dict(fmt_counts),
            "dim_histogram": dict(dim_counts),
            "grid_histogram": {str(k): v for k, v in grid_counts.items()},
            "issues": issues,
        }

    majority_dim = int(dim_counts.most_common(1)[0][0])
    # grid_dim: prefer most common non-None; else None
    gc_items = [(g, c) for g, c in grid_counts.items() if g is not None]
    grid_dim: Optional[int] = None
    if gc_items:
        grid_dim = max(gc_items, key=lambda x: x[1])[0]

    n_no_grid = int(grid_counts.get(None, 0))
    n_with_grid = sum(c for g, c in grid_counts.items() if g is not None)
    if n_no_grid > 0 and n_with_grid > 0:
        issues.append(
            f"Mixed files with and without grid_feat ({n_with_grid} with, {n_no_grid} without in sample). "
            "Batches may be inconsistent; prefer a single extraction pipeline for Model H."
        )

    return {
        "n_total": n_total,
        "n_scanned": len(sample_paths),
        "region_dim": majority_dim,
        "grid_dim": grid_dim,
        "format_counts": dict(fmt_counts),
        "dim_histogram": dict(dim_counts),
        "grid_histogram": {str(k): v for k, v in grid_counts.items()},
        "issues": issues,
    }


def make_butd_loader(feat_dir: str) -> Callable[[int], Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[str]]]]:
    """
    Returns a callable f(img_id) → (feat_tensor, grid_feat, label_names_or_None).

    feat_tensor : Tensor (k, feat_dim) — pre-extracted BUTD RoI features
    label_names : List[str] of length k, e.g. ['dog', 'fire hydrant', ...]
                  None if the .pt file was extracted without label metadata.

    The label_names are used by G2 (three-way PGN) to build visual copy distribution.
    """
    def _load(img_id: int, q_text: str = "") -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[str]]]:
        feat_path = os.path.join(feat_dir, f"{img_id}.pt")
        data = torch.load(feat_path, map_location="cpu", weights_only=True)
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict in {feat_path}, got {type(data)}")
        feat, grid, labels, _fmt = unpack_stored_visual_features(data)
        return feat, grid, labels

    return _load


# ---------------------------------------------------------------------------
# VQAGenerativeDataset
# ---------------------------------------------------------------------------

class VQAGenerativeDataset(Dataset):
    """
    Unified dataset for all VQA data sources used in Model G training.

    Supports:
    - merged_train_filtered.json  (VQA-E + VQA-X + A-OKVQA, unified format)
    - VQA v2.0 annotation JSONs   (via from_vqa_v2() classmethod)
    - BUTD feature loading        (via make_butd_loader)
    - Image loading               (via make_image_loader)

    Returns per sample:
    - feat_or_img : Tensor (3,H,W) or (k, feat_dim)
    - q_tensor    : Tensor (q_len,) — numericalized question
    - a_tensor    : Tensor (t_len,) — numericalized full target sequence
    - length_bin  : int — 0/1/2 for G5 length conditioning
    - label_names : List[str] or None — visual object labels for G2
    - source      : str — 'vqa_e'|'vqa_x'|'aokvqa'|'vqa_v2'|'synthetic'

    Args:
        annotations   : list of dicts in unified format (see from_merged_json)
        q_vocab       : Vocabulary for questions
        a_vocab       : Vocabulary for answers
        feature_loader: f(img_id, q_text) callable — returns (tensor, label_names?)
                        Use make_image_loader() or make_butd_loader()
        use_butd      : True if feature_loader returns (feat, labels) tuple
                        False if it returns plain Tensor (image loader)
        max_samples   : limit dataset size (debugging)
        always_long   : force all length_bins to LONG (inference mode)
    """

    def __init__(
        self,
        annotations: List[dict],
        q_vocab,
        a_vocab,
        feature_loader: Callable,
        use_butd: bool = False,
        max_samples: Optional[int] = None,
        always_long: bool = False,
        explanation_mode: str = "random",
    ):
        self.annotations = annotations
        if max_samples is not None:
            self.annotations = self.annotations[:max_samples]

        self.q_vocab = q_vocab
        self.a_vocab = a_vocab
        self.feature_loader = feature_loader
        self.use_butd = use_butd
        self.always_long = always_long
        self.explanation_mode = explanation_mode

    # -----------------------------------------------------------------------
    # Standard Dataset interface
    # -----------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int):
        ann = self.annotations[index]
        img_id = ann["img_id"]
        q_text = ann["question"]
        answer = ann.get("multiple_choice_answer", "")
        source = ann.get("source", "unknown")

        # VQA v2.0: target is a short answer only (pool = list of annotator strings in explanation)
        if source == "vqa_v2":
            pool = ann.get("explanation", [])
            if isinstance(pool, list) and pool:
                picked = random.choice(
                    [str(e).strip().lower() for e in pool if isinstance(e, str) and str(e).strip()]
                )
            else:
                picked = (answer or "").strip().lower()
            a_text = picked
            explanation = ""
            q_tensor = torch.tensor(self.q_vocab.numericalize(q_text), dtype=torch.long)
            a_tensor = torch.tensor(self.a_vocab.numericalize(a_text), dtype=torch.long)
            if self.always_long:
                length_bin = LENGTH_BIN_LONG
            else:
                length_bin = _full_seq_length_bin(picked, "")
        else:
            # Pick explanation — randomly sample from list (A-OKVQA has 3 rationales)
            exp_list = ann.get("explanation", [])
            valid_exps = [e for e in exp_list if isinstance(e, str) and e.strip()]
            if not valid_exps:
                explanation = ""
            elif self.explanation_mode == "first":
                explanation = valid_exps[0]
            else:
                explanation = random.choice(valid_exps)

            # Build target text
            a_text = f"{answer} because {explanation}" if explanation else answer

            # Numericalize
            q_tensor = torch.tensor(self.q_vocab.numericalize(q_text), dtype=torch.long)
            a_tensor = torch.tensor(self.a_vocab.numericalize(a_text), dtype=torch.long)

            # Length bin (G5) — computed from FULL sequence, not explanation-only
            if self.always_long:
                length_bin = LENGTH_BIN_LONG
            else:
                length_bin = _full_seq_length_bin(answer, explanation)

        # Load visual features
        label_names = None
        grid_feat = None
        if self.use_butd:
            feat_or_img, grid_feat, label_names = self.feature_loader(img_id, q_text)
        else:
            feat_or_img = self.feature_loader(img_id, q_text)

        return feat_or_img, grid_feat, q_tensor, a_tensor, length_bin, label_names, source

    # -----------------------------------------------------------------------
    # Factory classmethods
    # -----------------------------------------------------------------------

    @classmethod
    def from_merged_json(
        cls,
        json_path: str,
        q_vocab,
        a_vocab,
        feature_loader: Callable,
        use_butd: bool = False,
        max_samples: Optional[int] = None,
        sources_filter: Optional[List[str]] = None,
        min_quality_score: Optional[int] = None,
        always_long: bool = False,
    ) -> "VQAGenerativeDataset":
        """
        Load from data/processed/merged_train_filtered.json.

        Args:
            sources_filter    : if set, keep only samples from these sources
                                e.g. ['vqa_x', 'aokvqa'] for Phase 4
            min_quality_score : filter by quality score (None = keep all)
        """
        with open(json_path, "r") as f:
            annotations = json.load(f)

        if sources_filter is not None:
            annotations = [a for a in annotations if a.get("source") in sources_filter]

        if min_quality_score is not None:
            annotations = [a for a in annotations
                           if a.get("quality_score", 5) >= min_quality_score]

        return cls(
            annotations=annotations,
            q_vocab=q_vocab,
            a_vocab=a_vocab,
            feature_loader=feature_loader,
            use_butd=use_butd,
            max_samples=max_samples,
            always_long=always_long,
            explanation_mode="random",
        )

    @classmethod
    def from_vqa_v2(
        cls,
        question_json: str,
        annotation_json: str,
        q_vocab,
        a_vocab,
        feature_loader: Callable,
        use_butd: bool = False,
        split: str = "train2014",
        max_samples: Optional[int] = None,
    ) -> "VQAGenerativeDataset":
        """
        Load from VQA v2.0 question + annotation JSONs.
        Normalizes to unified format inline (no separate preprocess script needed).
        All resulting samples have source='vqa_v2' and no explanation → SHORT bin.
        """
        with open(question_json, "r") as f:
            questions = json.load(f)["questions"]

        with open(annotation_json, "r") as f:
            raw_anns = json.load(f)["annotations"]

        # Build qid → answers lookup
        qid2answers: Dict[int, List[str]] = {
            ann["question_id"]: [a["answer"] for a in ann.get("answers", [])]
                                 or [ann["multiple_choice_answer"]]
            for ann in raw_anns
        }

        # Normalize to unified format
        annotations = []
        for q in questions:
            qid = q["question_id"]
            answers = qid2answers.get(qid, [""])
            # Randomly pick at construction time for fixed splits,
            # or store all — we store all and pick in __getitem__ via list
            mc_answer = answers[0] if answers else ""
            annotations.append({
                "img_id": q["image_id"],
                "question": q["question"],
                "multiple_choice_answer": mc_answer,
                "explanation": answers,   # store all 10 for random pick in __getitem__
                "source": "vqa_v2",
                "split": split,
                "length_bin": "short",    # VQA v2.0 answers are short (1-3 tokens)
                "quality_score": 5,
            })

        return cls(
            annotations=annotations,
            q_vocab=q_vocab,
            a_vocab=a_vocab,
            feature_loader=feature_loader,
            use_butd=use_butd,
            max_samples=max_samples,
            always_long=False,
            explanation_mode="random",
        )

    @classmethod
    def from_legacy_vqae_json(
        cls,
        vqa_e_json: str,
        q_vocab,
        a_vocab,
        feature_loader: Callable,
        use_butd: bool = False,
        max_samples: Optional[int] = None,
    ) -> "VQAGenerativeDataset":
        """
        Load from raw VQA-E JSON (VQA-E_train_set.json format: explanation=[str, float]).
        Bridge for backward compat — new code should use from_merged_json().
        """
        with open(vqa_e_json, "r") as f:
            raw = json.load(f)

        annotations = []
        for ann in raw:
            exp_raw = ann.get("explanation", [])
            exp_text = exp_raw[0] if isinstance(exp_raw, list) and exp_raw else str(exp_raw)
            annotations.append({
                "img_id": ann["img_id"],
                "question": ann["question"],
                "multiple_choice_answer": ann.get("multiple_choice_answer", ""),
                "explanation": [exp_text] if isinstance(exp_text, str) else [],
                "source": "vqa_e",
                "split": "train",
                "quality_score": 5,
            })

        return cls(
            annotations=annotations,
            q_vocab=q_vocab,
            a_vocab=a_vocab,
            feature_loader=feature_loader,
            use_butd=use_butd,
            max_samples=max_samples,
            explanation_mode="random",
        )

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def subset(self, indices: List[int]) -> "VQAGenerativeDataset":
        """Return a new dataset containing only the specified indices."""
        return VQAGenerativeDataset(
            annotations=[self.annotations[i] for i in indices],
            q_vocab=self.q_vocab,
            a_vocab=self.a_vocab,
            feature_loader=self.feature_loader,
            use_butd=self.use_butd,
            always_long=self.always_long,
            explanation_mode=self.explanation_mode,
        )

    def filter_by_source(self, sources: List[str]) -> "VQAGenerativeDataset":
        kept = [a for a in self.annotations if a.get("source") in sources]
        return VQAGenerativeDataset(
            annotations=kept,
            q_vocab=self.q_vocab,
            a_vocab=self.a_vocab,
            feature_loader=self.feature_loader,
            use_butd=self.use_butd,
            always_long=self.always_long,
            explanation_mode=self.explanation_mode,
        )

    def source_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for ann in self.annotations:
            s = ann.get("source", "unknown")
            counts[s] = counts.get(s, 0) + 1
        return counts
