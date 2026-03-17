"""
Tier-D4: Task Progressive Curriculum Learning (TPCL)
======================================================
Sorts training samples by answer complexity so the LSTM sees easy examples
first and gradually encounters harder, longer sequences.

LSTMs converge more reliably when simple patterns are learned before complex
ones — avoids getting stuck in degenerate local optima early in training.

Complexity score = answer word count + 3 × clause count
  (clause indicators: because, since, although, however, which, that, ...)

Pacing function (3 stages keyed to training progress):
  progress < 0.2  →  stage 1: short answers   (bottom 20% by complexity)
  progress < 0.5  →  stage 2: medium answers  (bottom 50%)
  progress ≥ 0.5  →  stage 3: all samples

Usage in train.py
-----------------
  from training.curriculum import CurriculumSampler, compute_complexity_scores

  # After building dataset:
  if use_curriculum:
      scores = compute_complexity_scores(dataset.annotations)
      sampler = CurriculumSampler(scores, epoch=0, total_epochs=10)
      train_loader = DataLoader(dataset, batch_size=..., sampler=sampler, ...)

  # Update sampler each epoch (before iterating):
  sampler.set_epoch(epoch)
"""

import random
from torch.utils.data import Sampler


# Clause words that indicate sentence complexity
_CLAUSE_WORDS = frozenset({
    'because', 'since', 'although', 'however', 'therefore', 'thus',
    'which', 'that', 'who', 'whom', 'while', 'when', 'where',
    'whereas', 'despite', 'unless', 'until', 'whether', 'both',
})


def compute_complexity_scores(annotations):
    """
    Compute a complexity score for each annotation.

    Works for both VQAEDataset annotations (with 'explanation' key)
    and VQADataset annotations (multiple_choice_answer only).

    Returns list[int] — one score per annotation, same order.
    """
    scores = []
    for ann in annotations:
        answer      = ann.get('multiple_choice_answer', '')
        exp_list    = ann.get('explanation', [])
        explanation = exp_list[0] if exp_list and isinstance(exp_list[0], str) else ''

        if explanation:
            full_text = f"{answer} because {explanation}"
        else:
            full_text = answer

        words   = full_text.lower().split()
        clauses = sum(1 for w in words if w in _CLAUSE_WORDS)
        scores.append(len(words) + clauses * 3)

    return scores


class CurriculumSampler(Sampler):
    """
    Yields training indices in curriculum order.

    Stage boundaries (by fraction of total samples, sorted by complexity):
      stage 1: 0%–20%   (short/simple answers)
      stage 2: 0%–50%   (includes stage 1)
      stage 3: 0%–100%  (all samples, fully random)

    After each epoch call set_epoch(epoch) to advance the pacing function.

    Args:
        complexity_scores : list[int] — one per sample
        epoch             : current epoch (0-indexed)
        total_epochs      : total epochs in this training run
    """

    def __init__(self, complexity_scores, epoch=0, total_epochs=10):
        self.scores       = complexity_scores
        self.epoch        = epoch
        self.total_epochs = total_epochs
        self._build_sorted_indices()

    def _build_sorted_indices(self):
        """Sort by score ascending (easy → hard) once."""
        paired = sorted(enumerate(self.scores), key=lambda x: x[1])
        self.sorted_indices = [i for i, _ in paired]
        n = len(self.sorted_indices)
        self._s1_end = max(1, int(0.20 * n))  # 20% boundary
        self._s2_end = max(2, int(0.50 * n))  # 50% boundary

    def set_epoch(self, epoch):
        """Call before each epoch's DataLoader iteration."""
        self.epoch = epoch

    def _active_pool(self):
        progress = self.epoch / max(self.total_epochs - 1, 1)
        if progress < 0.2:
            return self.sorted_indices[:self._s1_end]
        elif progress < 0.5:
            return self.sorted_indices[:self._s2_end]
        else:
            return self.sorted_indices   # full dataset

    def __iter__(self):
        pool = self._active_pool().copy()
        random.shuffle(pool)
        return iter(pool)

    def __len__(self):
        return len(self._active_pool())


if __name__ == "__main__":
    # Quick test
    mock_annotations = [
        {'multiple_choice_answer': 'yes'},
        {'multiple_choice_answer': 'a red apple because the object is bright red in color'},
        {'multiple_choice_answer': 'two'},
        {'multiple_choice_answer': 'a dog',
         'explanation': ['it is a brown dog which is sitting on the mat because it is resting']},
        {'multiple_choice_answer': 'no'},
    ]
    scores = compute_complexity_scores(mock_annotations)
    print("Complexity scores:", scores)

    sampler = CurriculumSampler(scores, epoch=0, total_epochs=10)
    print("Epoch 0 (20% pool):", list(sampler))
    sampler.set_epoch(4)
    print("Epoch 4 (50% pool):", list(sampler))
    sampler.set_epoch(9)
    print("Epoch 9 (100% pool):", list(sampler))
    print("CurriculumSampler test PASSED")
