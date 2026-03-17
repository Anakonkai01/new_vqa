"""
Tier-D4: Task Progressive Curriculum Learning (TPCL) — v2
==========================================================
Revised per Gemini architectural review v3:

OLD approach: sort by answer length / clause count (answer-side heuristic)
NEW approach: sort by QUESTION TYPE — the question type directly encodes the
reasoning demand the model must satisfy:

  Phase 1  (progress < 0.25) — Binary questions (Yes/No)
    Model learns to produce the simplest explanations: short, factual.
    Examples: "Is it raining?" → "no because the sky is blue"

  Phase 2  (progress < 0.50) — Color + Count questions
    Model encounters numeric and perceptual answers.
    Examples: "How many cats?" → "two because there are two orange cats"

  Phase 3  (progress < 0.75) — What + Where questions
    Object identification and spatial reasoning.
    Examples: "What is on the table?" → "a book because a red book is resting..."

  Phase 4  (progress ≥ 0.75) — Why + How questions  (full dataset)
    Causal and procedural reasoning — longest, most complex explanations.
    Examples: "Why is the man smiling?" → "he is happy because he just won..."

Rationale:
  Difficulty correlates strongly with answer vocabulary breadth and explanation
  length.  Binary → 2 possible answers, simple grammar.  Why/How → open-ended,
  long explanations with causal connectives ("because", "since", "therefore").
  Introducing hard types early destabilizes the LSTM before it has learned the
  basic sequence structure.

Usage
-----
  from training.curriculum import CurriculumSampler, compute_question_type_scores

  scores  = compute_question_type_scores(dataset.annotations)
  sampler = CurriculumSampler(scores, epoch=0, total_epochs=10)
  loader  = DataLoader(dataset, batch_size=..., sampler=sampler)

  # Start of each epoch:
  sampler.set_epoch(epoch)
"""

import random
from torch.utils.data import Sampler


# ── Question type classifier ────────────────────────────────────────────────────

# Ordered difficulty tiers (int 0 = easiest → 5 = hardest)
_BINARY_PREFIXES = (
    'is ', 'are ', 'does ', 'do ', 'did ', 'was ', 'were ',
    'has ', 'have ', 'had ', 'can ', 'could ', 'will ', 'would ',
    'should ', 'shall ', 'may ', 'might ',
)

def classify_question_type(question: str) -> int:
    """
    Assign an integer complexity tier based on question type.

    0 — Binary   : Yes/No questions (is, are, does, do, did, was, were ...)
    1 — Color    : perceptual attribute (what color / what colour)
    2 — Count    : numeric (how many, how much)
    3 — What     : object/attribute identification
    4 — Where    : spatial reasoning
    5 — Why/How  : causal and procedural — hardest, longest explanations

    Unknown question forms default to tier 3 (What-level difficulty).
    """
    q = question.lower().strip()

    # Binary: yes/no (easiest)
    if q.startswith(_BINARY_PREFIXES):
        return 0

    # Color perception
    if 'color' in q or 'colour' in q:
        return 1

    # Numeric
    if q.startswith(('how many ', 'how much ')):
        return 2

    # What: object / attribute (medium)
    if q.startswith('what '):
        return 3

    # Spatial
    if q.startswith(('where ', 'which ')):
        return 4

    # Causal / procedural (hardest)
    if q.startswith(('why ', 'how ')):
        return 5

    # Default: treat as What-level
    return 3


def compute_question_type_scores(annotations) -> list:
    """
    Compute question-type complexity scores for a list of annotations.

    Accepts two annotation formats:
      • VQAEDataset / BUTDDataset: dicts with 'question' key
      • VQADataset:                dicts with 'question' key (same key name)

    Returns list[int] — one score per annotation, same order.
    Scores are in [0, 5]; lower = simpler.
    """
    scores = []
    for ann in annotations:
        q_text = ann.get('question', '')
        scores.append(classify_question_type(q_text))
    return scores


# Legacy: kept for backward compat with existing train.py calls that use
# compute_complexity_scores.  New code should call compute_question_type_scores.
def compute_complexity_scores(annotations) -> list:
    """
    Alias → compute_question_type_scores.

    Previous implementation sorted by answer word count + clause count
    (answer-side heuristic).  Revised per Gemini v3 directive to sort by
    question type complexity (question-side), which more accurately reflects
    the reasoning demand and aligns with standard VQA curriculum literature.
    """
    return compute_question_type_scores(annotations)


# ── Sampler ─────────────────────────────────────────────────────────────────────

class CurriculumSampler(Sampler):
    """
    Yields training indices in question-type curriculum order.

    Stage boundaries (by fraction of training progress):
      stage 1  0% → 25%  : Binary only          (type 0)
      stage 2 25% → 50%  : Binary + Color/Count  (types 0–2)
      stage 3 50% → 75%  : All but Why/How       (types 0–4)
      stage 4 75% → 100% : Full dataset           (types 0–5)

    Within each stage the active pool is shuffled, preserving randomness
    while respecting the curriculum order across epochs.

    Args:
        complexity_scores : list[int] — one score per sample (from compute_question_type_scores)
        epoch             : current epoch (0-indexed)
        total_epochs      : total epochs for this training run
    """

    def __init__(self, complexity_scores, epoch=0, total_epochs=10):
        self.scores       = complexity_scores
        self.epoch        = epoch
        self.total_epochs = total_epochs
        self._build_sorted_buckets()

    def _build_sorted_buckets(self):
        """Group indices by question-type tier once at init."""
        # Sort all indices by their complexity score ascending
        paired = sorted(enumerate(self.scores), key=lambda x: x[1])
        self.sorted_indices = [i for i, _ in paired]

        # Precompute stage boundaries (cumulative)
        n = len(self.sorted_indices)
        # Find where types > 0, > 2, > 4 begin
        scores_sorted = [self.scores[i] for i in self.sorted_indices]

        self._stage_ends = []
        for threshold in (0, 2, 4):
            # Last index where score <= threshold
            end = sum(1 for s in scores_sorted if s <= threshold)
            self._stage_ends.append(max(1, end))
        # Stage 4: full dataset (no boundary needed)

    def set_epoch(self, epoch: int):
        """Call before each epoch's DataLoader iteration."""
        self.epoch = epoch

    def _active_pool(self):
        progress = self.epoch / max(self.total_epochs - 1, 1)
        if progress < 0.25:
            # Stage 1: Binary only
            return self.sorted_indices[:self._stage_ends[0]]
        elif progress < 0.50:
            # Stage 2: Binary + Color/Count
            return self.sorted_indices[:self._stage_ends[1]]
        elif progress < 0.75:
            # Stage 3: Binary + Color/Count + What/Where
            return self.sorted_indices[:self._stage_ends[2]]
        else:
            # Stage 4: full dataset
            return self.sorted_indices

    def __iter__(self):
        pool = list(self._active_pool())
        random.shuffle(pool)
        return iter(pool)

    def __len__(self):
        return len(self._active_pool())


if __name__ == "__main__":
    mock_annotations = [
        {'question': 'Is the dog running?'},               # binary → 0
        {'question': 'What color is the car?'},            # color  → 1
        {'question': 'How many people are there?'},        # count  → 2
        {'question': 'What is on the table?'},             # what   → 3
        {'question': 'Where is the cat sitting?'},         # where  → 4
        {'question': 'Why is the man smiling?'},           # why    → 5
        {'question': 'How was the photo taken?'},          # how    → 5
        {'question': 'Are there any trees?'},              # binary → 0
    ]

    scores = compute_question_type_scores(mock_annotations)
    print("Question type scores:", scores)
    assert scores == [0, 1, 2, 3, 4, 5, 5, 0], f"Unexpected: {scores}"

    sampler = CurriculumSampler(scores, epoch=0, total_epochs=10)
    print(f"Epoch 0 (binary only, stage 1): pool size = {len(sampler)}")

    sampler.set_epoch(3)
    print(f"Epoch 3 (binary+color/count, stage 2): pool size = {len(sampler)}")

    sampler.set_epoch(6)
    print(f"Epoch 6 (all but why/how, stage 3): pool size = {len(sampler)}")

    sampler.set_epoch(9)
    print(f"Epoch 9 (full dataset, stage 4): pool size = {len(sampler)}")

    print("CurriculumSampler v2 test PASSED")
