"""
Dataset class:
  Input : index
  Output: 1 sample (image_tensor, question_tensor, answer_tensor)

Loads raw images and encodes questions/answers to index tensors.

Changes (D1):
  - Horizontal flip guard: skips flip for spatial questions (left/right/etc.)
  - RandomResizedCrop + RandAugment + RandomErasing replace old ColorJitter pipeline
  - VQAEDataset: randomly picks one explanation per epoch (not always [0])
  - VQADataset: randomly picks one of 10 human annotations per epoch
"""

import random
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from torchvision import transforms
import json
import os


# Words that indicate spatial orientation — flipping would change the answer
SPATIAL_KEYWORDS = {
    'left', 'right', 'east', 'west', 'leftmost', 'rightmost',
    'lefthand', 'righthand', 'clockwise', 'counterclockwise',
}


def vqa_collate_fn(batch):
    """
    Handle batches with variable-length sequences.
    pad_sequence finds the longest sequence in the batch and zero-pads shorter ones.
    """
    imgs, questions, answers = zip(*batch)
    imgs_stacked     = torch.stack(imgs, dim=0)
    questions_padded = pad_sequence(questions, batch_first=True)
    answer_padded    = pad_sequence(answers,   batch_first=True)
    return imgs_stacked, questions_padded, answer_padded


def _build_transforms(augment):
    """
    Returns (transform_base, augment_flag).
    Flip is applied manually in __getitem__ with spatial keyword guard.
    """
    if augment:
        transform_base = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        ])
    else:
        transform_base = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transform_base


def _apply_flip_guard(image, q_text, augment):
    """
    Apply RandomHorizontalFlip only if the question has no spatial keywords.
    Called after PIL image is loaded, before the main transform pipeline.
    """
    if not augment:
        return image
    words = set(q_text.lower().split())
    if words & SPATIAL_KEYWORDS:
        return image   # skip flip — answer depends on left/right
    if random.random() < 0.5:
        return TF.hflip(image)
    return image


class VQAEDataset(Dataset):
    """
    Dataset for VQA-E: loads a single annotation JSON that contains
    question + answer + explanation per entry.
    Target sequence: "<start> answer because explanation <end>"
    Images: same COCO 2014 images as VQA 2.0 (no new download needed).
    """
    def __init__(self, image_dir, vqa_e_json_path, vocab_q, vocab_a,
                 split='train2014', max_samples=None, augment=False):
        self.image_dir    = image_dir
        self.vocab_q      = vocab_q
        self.vocab_a      = vocab_a
        self.split        = split
        self.augment      = augment
        self.transform    = _build_transforms(augment)

        with open(vqa_e_json_path, 'r') as f:
            self.annotations = json.load(f)

        if max_samples is not None:
            self.annotations = self.annotations[:max_samples]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        ann    = self.annotations[index]
        q_text = ann['question']
        img_id = ann['img_id']

        answer   = ann.get('multiple_choice_answer', '')
        exp_list = ann.get('explanation', [])

        # D1: randomly pick one valid explanation each epoch (not always index 0)
        valid_exps  = [e for e in exp_list if isinstance(e, str) and e.strip()]
        explanation = random.choice(valid_exps) if valid_exps else ''

        a_text   = f"{answer} because {explanation}" if explanation else answer
        q_tensor = torch.tensor(self.vocab_q.numericalize(q_text), dtype=torch.long)
        a_tensor = torch.tensor(self.vocab_a.numericalize(a_text), dtype=torch.long)

        img_name  = f"COCO_{self.split}_{img_id:012d}.jpg"
        img_path  = os.path.join(self.image_dir, img_name)
        image     = Image.open(img_path).convert("RGB")

        # D1: spatial-aware flip guard before main transform
        image     = _apply_flip_guard(image, q_text, self.augment)
        img_tensor = self.transform(image)

        return img_tensor, q_tensor, a_tensor


class VQADataset(Dataset):
    def __init__(self, image_dir, question_json_path,
                 annotations_json_path, vocab_q, vocab_a,
                 split='train2014', max_samples=None, augment=False):
        """
        split      : 'train2014' or 'val2014'
        max_samples: limit samples (useful for fast pipeline testing)
        augment    : if True, apply augmentation (train only — NOT for val/test)
        """
        self.image_dir = image_dir
        self.vocab_q   = vocab_q
        self.vocab_a   = vocab_a
        self.split     = split
        self.augment   = augment
        self.transform = _build_transforms(augment)

        with open(question_json_path, 'r') as f:
            self.questions = json.load(f)['questions']

        if max_samples is not None:
            self.questions = self.questions[:max_samples]

        with open(annotations_json_path, 'r') as f:
            annotations = json.load(f)['annotations']

        # D1: store all 10 annotations per question (not just multiple_choice_answer)
        self.qid2answers = {
            ann['question_id']: [a['answer'] for a in ann.get('answers', [])]
                                 or [ann['multiple_choice_answer']]
            for ann in annotations
        }

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        q_info = self.questions[index]
        q_text = q_info['question']
        q_id   = q_info['question_id']
        img_id = q_info['image_id']

        q_tensor = torch.tensor(self.vocab_q.numericalize(q_text), dtype=torch.long)

        # D1: randomly pick one of the 10 human annotations each epoch
        answers = self.qid2answers.get(q_id, [''])
        a_text  = random.choice(answers) if answers else ''
        a_tensor = torch.tensor(self.vocab_a.numericalize(a_text), dtype=torch.long)

        img_name   = f"COCO_{self.split}_{img_id:012d}.jpg"
        img_path   = os.path.join(self.image_dir, img_name)
        image      = Image.open(img_path).convert("RGB")

        # D1: spatial-aware flip guard
        image      = _apply_flip_guard(image, q_text, self.augment)
        img_tensor = self.transform(image)

        return img_tensor, q_tensor, a_tensor
