"""
Dataset class:
  Input : index
  Output: 1 sample (image_tensor/feat_tensor, question_tensor, answer_tensor, length_bin_tensor)

Loads raw images/features and encodes questions/answers to index tensors.

Changes (D1):
  - Horizontal flip guard: skips flip for spatial questions (left/right/etc.)
  - RandomResizedCrop + RandAugment + RandomErasing replace old ColorJitter pipeline
  - VQAEDataset: randomly picks one explanation per epoch (not always [0])
  - VQADataset: randomly picks one of 10 human annotations per epoch

Changes (D2):
  - build_mixed_sampler(): WeightedRandomSampler mixing VQA v2.0 and VQA-E
    at a configurable ratio (default 70/30) to prevent length bias.

Changes (G5 FIX):
  - Explicitly calculate `length_bin` dynamically inside `__getitem__` based on 
    the final numericalized `a_tensor` sequence length.
  - 0: SHORT (<=5), 1: MEDIUM (6-14), 2: LONG (>=15).
"""

import random
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler
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
    NOW UNPACKS 4 ELEMENTS (including length_bins).
    """
    imgs, questions, answers, length_bins = zip(*batch)
    
    imgs_stacked        = torch.stack(imgs, dim=0)
    questions_padded    = pad_sequence(questions, batch_first=True)
    answer_padded       = pad_sequence(answers,   batch_first=True)
    length_bins_stacked = torch.stack(length_bins, dim=0) # (B,)
    
    return imgs_stacked, questions_padded, answer_padded, length_bins_stacked


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
    def __init__(self, image_dir, vqa_e_json_path, vocab,
                 split='train2014', max_samples=None, augment=False):
        self.image_dir    = image_dir
        self.vocab        = vocab
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
        q_tensor = torch.tensor(self.vocab.numericalize(q_text), dtype=torch.long)
        a_tensor = torch.tensor(self.vocab.numericalize(a_text), dtype=torch.long)

        # --- G5 FIX: TÍNH TOÁN LENGTH BIN ĐỘNG ---
        seq_len = a_tensor.size(0)
        if seq_len <= 5:
            len_bin = 0    # SHORT
        elif seq_len <= 14:
            len_bin = 1    # MEDIUM
        else:
            len_bin = 2    # LONG
        len_bin_tensor = torch.tensor(len_bin, dtype=torch.long)

        img_name  = f"COCO_{self.split}_{img_id:012d}.jpg"
        img_path  = os.path.join(self.image_dir, img_name)
        image     = Image.open(img_path).convert("RGB")

        # D1: spatial-aware flip guard before main transform
        image     = _apply_flip_guard(image, q_text, self.augment)
        img_tensor = self.transform(image)

        # RETURN 4 ELEMENTS
        return img_tensor, q_tensor, a_tensor, len_bin_tensor


class VQADataset(Dataset):
    def __init__(self, image_dir, question_json_path,
                 annotations_json_path, vocab,
                 split='train2014', max_samples=None, augment=False):
        self.image_dir = image_dir
        self.vocab     = vocab
        self.split     = split
        self.augment   = augment
        self.transform = _build_transforms(augment)

        with open(question_json_path, 'r') as f:
            self.questions = json.load(f)['questions']

        if max_samples is not None:
            self.questions = self.questions[:max_samples]

        with open(annotations_json_path, 'r') as f:
            annotations = json.load(f)['annotations']

        # D1: store all 10 annotations per question
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

        q_tensor = torch.tensor(self.vocab.numericalize(q_text), dtype=torch.long)

        # D1: randomly pick one of the 10 human annotations each epoch
        answers = self.qid2answers.get(q_id, [''])
        a_text  = random.choice(answers) if answers else ''
        a_tensor = torch.tensor(self.vocab.numericalize(a_text), dtype=torch.long)

        # --- G5 FIX: TÍNH TOÁN LENGTH BIN ĐỘNG ---
        seq_len = a_tensor.size(0)
        if seq_len <= 5:
            len_bin = 0    # SHORT (Hầu hết VQA v2.0 sẽ rơi vào đây)
        elif seq_len <= 14:
            len_bin = 1    # MEDIUM
        else:
            len_bin = 2    # LONG
        len_bin_tensor = torch.tensor(len_bin, dtype=torch.long)

        img_name   = f"COCO_{self.split}_{img_id:012d}.jpg"
        img_path   = os.path.join(self.image_dir, img_name)
        image      = Image.open(img_path).convert("RGB")

        image      = _apply_flip_guard(image, q_text, self.augment)
        img_tensor = self.transform(image)

        # RETURN 4 ELEMENTS
        return img_tensor, q_tensor, a_tensor, len_bin_tensor


def build_mixed_sampler(vqa_v2_dataset: Dataset, vqae_dataset: Dataset,
                        vqa_fraction: float = 0.7,
                        num_samples: int = None):
    """
    Build a ConcatDataset + WeightedRandomSampler that mixes VQA v2.0 and VQA-E
    at a controlled ratio to prevent length bias during Phase 1 pretraining.
    """
    assert 0.0 < vqa_fraction < 1.0, "vqa_fraction must be in (0, 1)"
    vqae_fraction = 1.0 - vqa_fraction

    n_vqa  = len(vqa_v2_dataset)
    n_vqae = len(vqae_dataset)

    w_vqa  = vqa_fraction  / n_vqa
    w_vqae = vqae_fraction / n_vqae

    weights = [w_vqa]  * n_vqa + [w_vqae] * n_vqae

    total = num_samples or (n_vqa + n_vqae)
    concat = ConcatDataset([vqa_v2_dataset, vqae_dataset])
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=total,
        replacement=True,
    )
    return concat, sampler


class BUTDDataset(VQAEDataset):
    """
    Tier-3B: Dataset that loads pre-extracted Faster R-CNN RoI features from disk
    instead of raw images.
    """
    def __init__(self, feat_dir, vqa_e_json_path, vocab,
                 split='train2014', max_samples=None):
        super().__init__(image_dir='', vqa_e_json_path=vqa_e_json_path,
                         vocab=vocab, split=split,
                         max_samples=max_samples, augment=False)
        self.feat_dir = feat_dir

    def __getitem__(self, index):
        ann    = self.annotations[index]
        img_id = ann['img_id']
        q_text = ann['question']

        feat_path = os.path.join(self.feat_dir, f"{img_id}.pt")
        data      = torch.load(feat_path, map_location='cpu', weights_only=True)
        feat_tensor = data['feat']   # (k, feat_dim)

        answer   = ann.get('multiple_choice_answer', '')
        exp_list = ann.get('explanation', [])
        valid_exps  = [e for e in exp_list if isinstance(e, str) and e.strip()]
        explanation = random.choice(valid_exps) if valid_exps else ''
        a_text   = f"{answer} because {explanation}" if explanation else answer

        q_tensor = torch.tensor(self.vocab.numericalize(q_text), dtype=torch.long)
        a_tensor = torch.tensor(self.vocab.numericalize(a_text), dtype=torch.long)
        
        # --- G5 FIX: TÍNH TOÁN LENGTH BIN ĐỘNG ---
        seq_len = a_tensor.size(0)
        if seq_len <= 5:
            len_bin = 0    # SHORT
        elif seq_len <= 14:
            len_bin = 1    # MEDIUM
        else:
            len_bin = 2    # LONG
        len_bin_tensor = torch.tensor(len_bin, dtype=torch.long)

        # RETURN 4 ELEMENTS
        return feat_tensor, q_tensor, a_tensor, len_bin_tensor


def butd_collate_fn(batch):
    """
    Collate for BUTDDataset. Pads the region (k) dimension across the batch.
    NOW UNPACKS 4 ELEMENTS (including length_bins).
    """
    feats, questions, answers, length_bins = zip(*batch)
    
    feats_padded = pad_sequence(feats,     batch_first=True)  # (B, max_k, feat_dim)
    q_padded     = pad_sequence(questions, batch_first=True)
    a_padded     = pad_sequence(answers,   batch_first=True)
    
    img_mask = feats_padded.abs().sum(dim=-1) > 0             # (B, max_k) bool
    length_bins_stacked = torch.stack(length_bins, dim=0)     # (B,)
    
    # TRẢ VỀ 5 ELEMENTS
    return feats_padded, q_padded, a_padded, img_mask, length_bins_stacked