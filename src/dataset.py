"""Dataset classes for the VQA project.

Primary class : ``VQAEDataset`` — multi-task dataset combining VQA-E explanations
                and COCO Captions. Used for all active training pipelines.

Legacy class  : ``VQADataset`` — original VQA 2.0 dataset for baseline
                Models A/B/C/D. Retained for evaluation compatibility only.
"""

from __future__ import annotations

import json
import os
import random
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms

from vocab import Vocabulary

# ── ImageNet normalization constants ────────────────────────────────────────
# Required for both pretrained (ResNet101, CLIP ViT-B/32) and scratch CNN
# encoders to maintain a consistent input distribution across all models.
_IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
_IMAGENET_STD:  List[float] = [0.229, 0.224, 0.225]


def _build_transform(augment: bool) -> transforms.Compose:
    """Build the image pre-processing pipeline.

    The base pipeline resizes to 224×224 (required by ResNet101 and CLIP
    ViT-B/32 with patch_size=32), converts to a float tensor, and applies
    ImageNet normalization.

    Augmentation (train split only) adds mild perturbations that do not
    change the semantic answer:

    - ``RandomHorizontalFlip``: valid because VQA answers are not
      left/right-sensitive.
    - ``ColorJitter``: mild brightness/contrast/saturation shifts that do
      not alter scene content.

    Args:
        augment: If True, insert augmentation transforms (use for training
            split only — never for validation or test).

    Returns:
        A ``torchvision.transforms.Compose`` pipeline.
    """
    base: List = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
    if augment:
        # Insert before ToTensor (PIL-based transforms must precede tensor ops).
        base.insert(1, transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        ))
        base.insert(1, transforms.RandomHorizontalFlip(p=0.5))
    return transforms.Compose(base)


# ── Collate function ─────────────────────────────────────────────────────────

def vqa_collate_fn(
    batch: List[Tuple[Tensor, Tensor, Tensor]],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Collate a list of ``(image, question, answer)`` samples into a batch.

    Images are stacked directly (all are ``(3, 224, 224)`` after the transform).
    Questions and answers are zero-padded to the longest sequence in the batch.
    ``padding_value=0`` aligns with ``<pad>`` at index 0, which is the
    ``ignore_index`` in ``CrossEntropyLoss``.

    Args:
        batch: List of tuples returned by ``Dataset.__getitem__``.

    Returns:
        Tuple of:
            imgs      – ``FloatTensor (B, 3, 224, 224)``
            questions – ``LongTensor  (B, Q)``  zero-padded to max Q in batch
            answers   – ``LongTensor  (B, A)``  zero-padded to max A in batch
    """
    imgs, questions, answers = zip(*batch)
    imgs_stacked     = torch.stack(imgs, dim=0)
    questions_padded = pad_sequence(questions, batch_first=True, padding_value=0)
    answers_padded   = pad_sequence(answers,   batch_first=True, padding_value=0)
    return imgs_stacked, questions_padded, answers_padded


# ── Primary dataset ──────────────────────────────────────────────────────────

class VQAEDataset(Dataset):
    """Multi-task dataset combining VQA-E explanations and COCO Captions.

    Each VQA-E sample produces:

    - Question input:  ``"<task_vqa> <question text>"``
    - Target sequence: ``"<start> <answer> because <explanation> <end>"``

    Each COCO Caption sample produces:

    - Question input:  ``"<task_cap>"`` (no question; pure captioning task)
    - Target sequence: ``"<start> <caption text> <end>"``

    The ``<task_vqa>`` / ``<task_cap>`` discriminator tokens allow the shared
    decoder to distinguish the two objectives without separate heads.

    Args:
        image_dir: Path to the COCO images directory (``train2014`` or
            ``val2014``).
        vqa_e_json_path: Path to the VQA-E annotation JSON file.
        vocab_q: ``Vocabulary`` object for questions.
        vocab_a: ``Vocabulary`` object for answers/captions.
        caption_json_path: Optional path to the COCO Captions JSON for
            multi-task training. When ``None``, only VQA-E samples are used.
        split: COCO split name — ``'train2014'`` or ``'val2014'``. Used to
            construct image filenames (``COCO_{split}_{id:012d}.jpg``).
        max_samples: If set, randomly subsample to this many items using a
            fixed seed (42) for reproducibility. Useful for quick smoke tests.
        augment: If True, apply data augmentation. Use only for training split.
    """

    def __init__(
        self,
        image_dir: str,
        vqa_e_json_path: str,
        vocab_q: Vocabulary,
        vocab_a: Vocabulary,
        caption_json_path: Optional[str] = None,
        split: str = "train2014",
        max_samples: Optional[int] = None,
        augment: bool = False,
    ) -> None:
        self.image_dir = image_dir
        self.vocab_q   = vocab_q
        self.vocab_a   = vocab_a
        self.split     = split
        self.transform = _build_transform(augment)

        self.annotations: List[dict] = []

        # ── Load VQA-E annotations ───────────────────────────────────────────
        if not os.path.exists(vqa_e_json_path):
            raise FileNotFoundError(f"VQA-E JSON not found: {vqa_e_json_path}")
        with open(vqa_e_json_path, "r") as f:
            vqa_data: List[dict] = json.load(f)
        for item in vqa_data:
            item["task_type"] = "vqa"
        self.annotations.extend(vqa_data)

        # ── Load COCO Captions (optional, for multi-task) ────────────────────
        if caption_json_path is not None:
            if not os.path.exists(caption_json_path):
                raise FileNotFoundError(f"Captions JSON not found: {caption_json_path}")
            with open(caption_json_path, "r") as f:
                cap_data: List[dict] = json.load(f)["annotations"]
            for item in cap_data:
                item["task_type"] = "caption"
            self.annotations.extend(cap_data)

        # ── Optional subsetting ──────────────────────────────────────────────
        # Use an isolated Random instance (not the global one) so this never
        # affects random state in the training loop or data augmentation.
        if max_samples is not None:
            rng = random.Random(42)
            rng.shuffle(self.annotations)
            self.annotations = self.annotations[:max_samples]

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Return ``(image, question, answer)`` tensors for sample at *index*.

        Returns:
            Tuple of:
                img_tensor – ``FloatTensor (3, 224, 224)``, ImageNet-normalized.
                q_tensor   – ``LongTensor (Q,)``, question token indices.
                a_tensor   – ``LongTensor (A,)``, answer/caption token indices.
        """
        ann       = self.annotations[index]
        task_type = ann.get("task_type", "vqa")

        if task_type == "vqa":
            q_text  = f"<task_vqa> {ann['question']}"
            img_id  = ann["img_id"]   # VQA-E uses 'img_id'

            answer      = ann.get("multiple_choice_answer", "")
            exp_list    = ann.get("explanation", [])
            explanation = exp_list[0] if exp_list and isinstance(exp_list[0], str) else ""
            a_text      = f"{answer} because {explanation}" if explanation else answer

        else:  # task_type == "caption"
            q_text = "<task_cap>"
            img_id = ann["image_id"]  # COCO Captions uses 'image_id'
            a_text = ann["caption"]

        q_tensor = torch.tensor(self.vocab_q.numericalize(q_text), dtype=torch.long)
        a_tensor = torch.tensor(self.vocab_a.numericalize(a_text),  dtype=torch.long)

        img_name   = f"COCO_{self.split}_{img_id:012d}.jpg"
        img_path   = os.path.join(self.image_dir, img_name)
        img_tensor = self.transform(Image.open(img_path).convert("RGB"))

        return img_tensor, q_tensor, a_tensor


# ── Legacy dataset (VQA 2.0) ─────────────────────────────────────────────────
# Used for the baseline Models A/B/C/D trained on the original VQA 2.0 dataset.
# Not used in the active Model E pipeline (VQA-E + COCO Captions).
# Retained for backward compatibility with evaluate.py / compare.py.

class VQADataset(Dataset):
    """VQA 2.0 dataset for baseline Models A/B/C/D.

    .. deprecated::
        Use ``VQAEDataset`` for all new training. This class is retained only
        for evaluating previously trained A/B/C/D checkpoints.

    Args:
        image_dir: Path to COCO images directory.
        question_json_path: Path to VQA 2.0 questions JSON.
        annotations_json_path: Path to VQA 2.0 annotations JSON.
        vocab_q: Question vocabulary.
        vocab_a: Answer vocabulary.
        split: ``'train2014'`` or ``'val2014'``.
        max_samples: Optional cap on samples for quick testing.
        augment: Enable data augmentation (training split only).
    """

    def __init__(
        self,
        image_dir: str,
        question_json_path: str,
        annotations_json_path: str,
        vocab_q: Vocabulary,
        vocab_a: Vocabulary,
        split: str = "train2014",
        max_samples: Optional[int] = None,
        augment: bool = False,
    ) -> None:
        self.image_dir = image_dir
        self.vocab_q   = vocab_q
        self.vocab_a   = vocab_a
        self.split     = split
        self.transform = _build_transform(augment)

        with open(question_json_path, "r") as f:
            self.questions: List[dict] = json.load(f)["questions"]

        if max_samples is not None:
            self.questions = self.questions[:max_samples]

        with open(annotations_json_path, "r") as f:
            annotations = json.load(f)["annotations"]
        self.qid2ans: dict[int, str] = {
            ann["question_id"]: ann["multiple_choice_answer"] for ann in annotations
        }

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Return ``(image, question, answer)`` tensors for sample at *index*.

        Returns:
            Tuple of:
                img_tensor – ``FloatTensor (3, 224, 224)``
                q_tensor   – ``LongTensor (Q,)``
                a_tensor   – ``LongTensor (A,)``
        """
        q_info = self.questions[index]
        q_id   = q_info["question_id"]
        img_id = q_info["image_id"]

        q_tensor = torch.tensor(
            self.vocab_q.numericalize(q_info["question"]), dtype=torch.long
        )
        a_tensor = torch.tensor(
            self.vocab_a.numericalize(self.qid2ans.get(q_id, "")), dtype=torch.long
        )

        img_name   = f"COCO_{self.split}_{img_id:012d}.jpg"
        img_path   = os.path.join(self.image_dir, img_name)
        img_tensor = self.transform(Image.open(img_path).convert("RGB"))

        return img_tensor, q_tensor, a_tensor
