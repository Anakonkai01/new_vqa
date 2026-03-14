"""CLIP-tokenised VQA dataset — used by Models F, G, and H.

This module provides ``VQAEDatasetCLIP``, a thin subclass of ``VQAEDataset``
that replaces the custom ``vocab_q``-based question tokenisation with the
CLIP BPE tokeniser.  Everything else (image loading, answer/caption targets,
multi-task mixing, augmentation) is inherited unchanged.

Why a separate dataset?
-----------------------
- CLIP's BPE vocabulary is fixed (49408 tokens).  Questions must be tokenised
  by ``CLIPProcessor`` and truncated/padded to **exactly 77 tokens**, matching
  the context window the CLIP text transformer was pre-trained with.
- The standard ``vqa_collate_fn`` already pads sequences to the longest in the
  batch; but since CLIP always outputs 77-length tensors there is no variable-
  length padding issue and the existing collate function works without changes.
- Answer targets are unchanged — they continue to use ``vocab_a`` and the
  existing collate path.

Usage
-----
    from dataset_clip import VQAEDatasetCLIP, clip_collate_fn

    ds = VQAEDatasetCLIP(
        image_dir       = "data/raw/train2014",
        vqa_e_json_path = "data/vqa_e/VQA-E_train_set.json",
        vocab_a         = vocab_a,
        caption_json_path = "data/raw/annotations/captions_train2014.json",
        split           = "train2014",
        augment         = True,
    )
    loader = DataLoader(ds, batch_size=64, collate_fn=clip_collate_fn)

    # Each batch: (imgs, clip_input_ids, answers)
    # imgs           : FloatTensor (B, 3, 224, 224)
    # clip_input_ids : LongTensor  (B, 77)
    # answers        : LongTensor  (B, A)  — padded answer token ids
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import CLIPProcessor

from dataset import VQAEDataset
from vocab import Vocabulary


# ── Collate function ──────────────────────────────────────────────────────────

def clip_collate_fn(
    batch: List[Tuple[Tensor, Tensor, Tensor]],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Collate ``(image, clip_input_ids, answer)`` samples into a batch.

    Images are stacked (all ``(3, 224, 224)``).
    ``clip_input_ids`` are already fixed-length 77 — just stacked, no padding.
    Answers are zero-padded to the longest in the batch.

    Args:
        batch: List of tuples from ``VQAEDatasetCLIP.__getitem__``.

    Returns:
        Tuple of:
            imgs           – ``FloatTensor (B, 3, 224, 224)``
            clip_input_ids – ``LongTensor  (B, 77)``
            answers        – ``LongTensor  (B, A)``  zero-padded
    """
    imgs, clip_ids, answers = zip(*batch)
    imgs_stacked   = torch.stack(imgs,     dim=0)   # (B, 3, 224, 224)
    clip_stacked   = torch.stack(clip_ids, dim=0)   # (B, 77)
    answers_padded = pad_sequence(answers, batch_first=True, padding_value=0)
    return imgs_stacked, clip_stacked, answers_padded


# ── Dataset ───────────────────────────────────────────────────────────────────

class VQAEDatasetCLIP(VQAEDataset):
    """VQA-E + COCO Captions dataset with CLIP BPE question tokenisation.

    Subclasses ``VQAEDataset`` and overrides ``__getitem__`` to tokenise the
    question text with the CLIP processor instead of ``vocab_q``.

    Key differences from ``VQAEDataset``
    -------------------------------------
    - ``vocab_q`` is **not** used; the ``CLIP BPE`` vocabulary is fixed.
    - ``__getitem__`` returns ``(img_tensor, clip_input_ids, a_tensor)`` where
      ``clip_input_ids`` is a ``LongTensor (77,)`` — fixed-length, always.
    - The ``<task_vqa>`` / ``<task_cap>`` token prefix is **not** prepended to
      the question for CLIP tokenisation.  CLIP's tokeniser does not know those
      special tokens; task conditioning for F/G/H is handled by the model
      architecture (the image encoder and decoder already receive separate
      visual and language inputs, making an explicit task token redundant).
    - Answer targets are tokenised with ``vocab_a`` exactly as in ``VQAEDataset``.

    Args:
        image_dir: Path to COCO images directory.
        vqa_e_json_path: Path to the VQA-E annotation JSON file.
        vocab_a: ``Vocabulary`` object for answers/captions.
        caption_json_path: Optional COCO Captions JSON path for multi-task.
        split: COCO split name (``'train2014'`` or ``'val2014'``).
        max_samples: Optional subsample cap for quick tests.
        augment: Enable data augmentation (training split only).
        clip_model_name: HuggingFace model name for the CLIP processor.
    """

    def __init__(
        self,
        image_dir: str,
        vqa_e_json_path: str,
        vocab_a: Vocabulary,
        caption_json_path: Optional[str] = None,
        split: str = "train2014",
        max_samples: Optional[int] = None,
        augment: bool = False,
        clip_model_name: str = "openai/clip-vit-base-patch32",
    ) -> None:
        # Pass a dummy vocab_q to the parent — it will not be used for questions.
        # The parent __init__ only needs vocab_q for numericalization of q_text,
        # which we override in __getitem__.  We pass vocab_a as a stand-in to
        # avoid requiring a separate vocab_q object.
        super().__init__(
            image_dir=image_dir,
            vqa_e_json_path=vqa_e_json_path,
            vocab_q=vocab_a,          # dummy — not used for CLIP tokenisation
            vocab_a=vocab_a,
            caption_json_path=caption_json_path,
            split=split,
            max_samples=max_samples,
            augment=augment,
        )

        # CLIP processor handles BPE tokenisation + padding to 77 tokens.
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Return ``(image, clip_input_ids, answer)`` for sample at *index*.

        Returns:
            Tuple of:
                img_tensor     – ``FloatTensor (3, 224, 224)``
                clip_input_ids – ``LongTensor  (77,)`` — CLIP BPE token ids.
                a_tensor       – ``LongTensor  (A,)``  — answer token ids (vocab_a).
        """
        import json
        from PIL import Image

        ann       = self.annotations[index]
        task_type = ann.get("task_type", "vqa")

        if task_type == "vqa":
            q_text  = ann["question"]           # raw question, no task prefix
            img_id  = ann["img_id"]
            answer      = ann.get("multiple_choice_answer", "")
            exp_list    = ann.get("explanation", [])
            explanation = exp_list[0] if exp_list and isinstance(exp_list[0], str) else ""
            a_text      = f"{answer} because {explanation}" if explanation else answer
        else:
            q_text  = ""                        # no question for captions
            img_id  = ann["image_id"]
            a_text  = ann["caption"]

        # ── CLIP tokenisation ─────────────────────────────────────────────────
        # padding="max_length" → always 77 tokens; truncation=True for long Qs.
        encoded = self.clip_processor(
            text=q_text,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )
        clip_input_ids = encoded["input_ids"].squeeze(0)   # (77,)

        # ── Answer tokenisation (vocab_a, unchanged) ──────────────────────────
        a_tensor = torch.tensor(self.vocab_a.numericalize(a_text), dtype=torch.long)

        # ── Image loading ─────────────────────────────────────────────────────
        img_name   = f"COCO_{self.split}_{img_id:012d}.jpg"
        img_path   = os.path.join(self.image_dir, img_name)
        img_tensor = self.transform(Image.open(img_path).convert("RGB"))

        return img_tensor, clip_input_ids, a_tensor


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    from vocab import Vocabulary
    from torch.utils.data import DataLoader

    # Build a minimal dummy vocabulary for the smoke test.
    vocab_a = Vocabulary()
    for w in ["<pad>", "<start>", "<end>", "<unk>", "<task_vqa>", "<task_cap>",
              "yes", "no", "a", "the", "is", "on"]:
        vocab_a.add_word(w)

    # These paths must exist on disk for a live test.
    IMAGE_DIR = "data/raw/train2014"
    VQA_JSON  = "data/vqa_e/VQA-E_train_set.json"

    if not os.path.exists(VQA_JSON):
        print("Skipping smoke test — data not found.")
        sys.exit(0)

    ds = VQAEDatasetCLIP(
        image_dir=IMAGE_DIR,
        vqa_e_json_path=VQA_JSON,
        vocab_a=vocab_a,
        split="train2014",
        max_samples=8,
        augment=False,
    )

    loader = DataLoader(ds, batch_size=4, collate_fn=clip_collate_fn)
    imgs, clip_ids, answers = next(iter(loader))

    print(f"imgs           : {imgs.shape}")       # (4, 3, 224, 224)
    print(f"clip_input_ids : {clip_ids.shape}")   # (4, 77)
    print(f"answers        : {answers.shape}")    # (4, A)
