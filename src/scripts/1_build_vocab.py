"""Build question and answer vocabularies from VQA-E and COCO Captions data.

Run from the repository root::

    python src/scripts/1_build_vocab.py

Outputs saved to ``data/processed/``:

    vocab_questions.json  – question vocabulary (threshold=3)
    vocab_answers.json    – answer + explanation + caption vocabulary (threshold=3)
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vocab import Vocabulary

# ── Candidate data file paths (first existing path wins) ────────────────────
_VQAE_PATHS: List[str] = [
    "data/vqa_e/VQA-E_train_set.json",
    "data/raw/vqa_e_json/VQA-E_train_set.json",
]
_CAPTIONS_PATHS: List[str] = [
    "data/raw/annotations/captions_train2014.json",
    "data/vqa_data_json/captions_train2014.json",
]
_OUTPUT_DIR: str = "data/processed"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _first_existing(paths: List[str]) -> Optional[str]:
    """Return the first path in *paths* that exists on disk, or ``None``.

    Args:
        paths: Candidate file paths in priority order.

    Returns:
        The first existing path, or ``None`` if none exist on disk.
    """
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _build_answer_text(ann: dict) -> str:
    """Construct the decoder target string from a VQA-E annotation dict.

    The decoder is trained to generate ``"<answer> because <explanation>"``
    when an explanation is available, or just ``"<answer>"`` otherwise.
    This matches the target construction in ``VQAEDataset.__getitem__``.

    Args:
        ann: Single VQA-E annotation dictionary.

    Returns:
        Combined answer + explanation string.
    """
    answer      = ann.get("multiple_choice_answer", "")
    exp_list    = ann.get("explanation", [])
    explanation = exp_list[0] if exp_list and isinstance(exp_list[0], str) else ""
    return f"{answer} because {explanation}" if explanation else answer


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    # ── Load VQA-E annotations ───────────────────────────────────────────────
    vqa_path = _first_existing(_VQAE_PATHS)
    if vqa_path is None:
        print(f"ERROR: VQA-E train set not found in any of: {_VQAE_PATHS}")
        print("Please download VQA-E and place it under data/vqa_e/")
        return

    print(f"\nReading VQA-E annotations: {vqa_path}")
    with open(vqa_path, "r") as f:
        annotations: List[dict] = json.load(f)
    print(f"  Loaded {len(annotations):,} annotations.")

    # ── Load COCO Captions (optional) ────────────────────────────────────────
    cap_path = _first_existing(_CAPTIONS_PATHS)
    captions: List[dict] = []
    if cap_path:
        print(f"Reading COCO Captions: {cap_path}")
        with open(cap_path, "r") as f:
            captions = json.load(f)["annotations"]
        print(f"  Loaded {len(captions):,} captions.")
    else:
        print("COCO Captions not found — vocab will be built from VQA-E only.")

    # ── Build question vocabulary ────────────────────────────────────────────
    print("\n1. Building question vocabulary...")
    questions: List[str] = [ann["question"] for ann in annotations if "question" in ann]
    q_vocab = Vocabulary()
    q_vocab.build(questions, threshold=3)
    q_out = os.path.join(_OUTPUT_DIR, "vocab_questions.json")
    q_vocab.save(q_out)
    print(f"   {q_vocab} | Saved → {q_out}")

    # ── Build answer/caption vocabulary ─────────────────────────────────────
    print("\n2. Building answer vocabulary (answer + explanation + captions)...")
    answer_texts: List[str] = [_build_answer_text(ann) for ann in annotations]
    answer_texts.extend(cap["caption"] for cap in captions)

    a_vocab = Vocabulary()
    a_vocab.build(answer_texts, threshold=3)
    a_out = os.path.join(_OUTPUT_DIR, "vocab_answers.json")
    a_vocab.save(a_out)
    print(f"   {a_vocab} | Saved → {a_out}")

    # ── Sanity check ─────────────────────────────────────────────────────────
    print("\nSample target sequences (first 3 VQA-E entries):")
    for ann in annotations[:3]:
        print(f"  Q: {ann['question']}")
        print(f"  A: {_build_answer_text(ann)}\n")


if __name__ == "__main__":
    main()
