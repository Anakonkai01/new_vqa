"""
Dataset class:
  Input : index
  Output: 1 sample (image_tensor, question_tensor, answer_tensor)

Loads raw images and encodes questions/answers to index tensors.

answer_sampling: if True, randomly pick 1 of the 10 human annotations
  instead of always using the majority vote (multiple_choice_answer).
  Enabled for training only — val always uses majority vote.
  Increases answer diversity and acts as a form of data augmentation.
"""


import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from torchvision import transforms
import json
import os


# helper function
# collate function
def vqa_collate_fn(batch):
    """
    Handle batches with variable-length sequences.
    pad_sequence finds the longest sequence in the batch and zero-pads shorter ones.
    """

    imgs, questions, answers = zip(*batch)  # unzip batch

    imgs_stacked     = torch.stack(imgs, dim=0)
    questions_padded = pad_sequence(questions, batch_first=True)
    answer_padded    = pad_sequence(answers, batch_first=True)

    return imgs_stacked, questions_padded, answer_padded


class VQADataset(Dataset):
    def __init__(self, image_dir, question_json_path,
                 annotations_json_path, vocab_q, vocab_a,
                 split='train2014', max_samples=None,
                 augment=False, answer_sampling=False):
        """
        split           : 'train2014' or 'val2014'
        max_samples     : cap samples for quick pipeline tests
        augment         : image augmentation (train only)
        answer_sampling : randomly pick 1 of 10 human annotations per sample
                          instead of always using majority vote (train only)
        """
        self.image_dir       = image_dir
        self.vocab_q         = vocab_q
        self.vocab_a         = vocab_a
        self.split           = split
        self.answer_sampling = answer_sampling

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                       saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        with open(question_json_path, 'r') as f:
            self.questions = json.load(f)['questions']

        if max_samples is not None:
            self.questions = self.questions[:max_samples]

        with open(annotations_json_path, 'r') as f:
            annotations = json.load(f)['annotations']

        # Build answer dict:
        # answer_sampling=True  → store list of all 10 annotations per question
        # answer_sampling=False → store only the majority vote string
        self.qid2ans = {}
        for ann in annotations:
            qid = ann['question_id']
            if answer_sampling:
                self.qid2ans[qid] = [a['answer'] for a in ann['answers']]
            else:
                self.qid2ans[qid] = ann['multiple_choice_answer']

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        q_info = self.questions[index]
        q_text = q_info['question']
        q_id   = q_info['question_id']
        img_id = q_info['image_id']

        # 1. question → indices tensor
        q_indices = self.vocab_q.numericalize(q_text)
        q_tensor  = torch.tensor(q_indices, dtype=torch.long)

        # 2. answer → indices tensor
        ans_data = self.qid2ans.get(q_id, "")
        if self.answer_sampling and isinstance(ans_data, list):
            a_text = random.choice(ans_data)   # randomly pick 1 of 10 per epoch
        else:
            a_text = ans_data
        a_indices = self.vocab_a.numericalize(a_text)
        a_tensor  = torch.tensor(a_indices, dtype=torch.long)

        # 3. image
        img_name = f"COCO_{self.split}_{img_id:012d}.jpg"
        img_path = os.path.join(self.image_dir, img_name)
        image    = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(image)

        return img_tensor, q_tensor, a_tensor
