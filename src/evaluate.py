"""Evaluation script for VQA models — automatic NLP metrics.

Runs a trained model on the validation set and computes:

* **BLEU-1/2/3/4** (n-gram precision, smoothed with SmoothingFunction.method1)
* **METEOR** (word-overlap + WordNet synonym matching)
* **BERTScore F1** (semantic similarity via contextual BERT embeddings,
  optional — requires ``pip install bert-score``)
* **Exact Match** (case-insensitive, whitespace-normalised)

The primary headline metrics are **BLEU-4** and **METEOR** since they best
reflect the quality of generated explanations.

Usage
-----
    # Greedy decode, all validation samples
    python src/evaluate.py --model_type E --checkpoint checkpoints/model_e_best.pth

    # Beam search (width 5)
    python src/evaluate.py --model_type E --checkpoint checkpoints/model_e_best.pth \\
        --beam_width 5 --no_repeat_ngram 3

    # Quick sanity check on 500 samples
    python src/evaluate.py --model_type E --checkpoint checkpoints/model_e_best.pth \\
        --num_samples 500
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Optional, Union

import nltk
import torch
import tqdm
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# BERTScore is optional — not in the base conda env.
try:
    from bert_score import score as bert_score_fn
    _HAS_BERTSCORE = True
except ImportError:
    _HAS_BERTSCORE = False

from dataset import VQAEDataset, vqa_collate_fn
from inference import (
    batch_beam_search_decode,
    batch_beam_search_decode_with_attention,
    batch_greedy_decode,
    batch_greedy_decode_with_attention,
    load_model_from_checkpoint,
    strip_compiled_prefix,
)
from vocab import Vocabulary


# ── Configuration ─────────────────────────────────────────────────────────────

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_IMAGE_DIR  = "data/raw/val2014"
VAL_VQA_E_JSON = "data/vqa_e/VQA-E_val_set.json"
VOCAB_Q_PATH   = "data/processed/vocab_questions.json"
VOCAB_A_PATH   = "data/processed/vocab_answers.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decode_answer_tensor(a_tensor: torch.Tensor, vocab: Vocabulary) -> str:
    """Convert a padded answer tensor to a plain string, skipping special tokens.

    Args:
        a_tensor: ``LongTensor (T,)`` — token indices for a single answer.
        vocab: Answer vocabulary with ``pad_idx``, ``start_idx``, ``end_idx``.

    Returns:
        Space-joined string of content words.
    """
    skip = {vocab.pad_idx, vocab.start_idx, vocab.end_idx}
    words = [vocab.idx2word[int(i)] for i in a_tensor if int(i) not in skip]
    return " ".join(words)


# ── Main evaluation function ──────────────────────────────────────────────────

def evaluate(
    model_type: str = "A",
    checkpoint: Optional[str] = None,
    num_samples: Optional[int] = None,
    beam_width: int = 1,
    no_repeat_ngram_size: int = 3,
) -> Dict[str, float]:
    """Run evaluation on the validation set and print + return metric scores.

    Args:
        model_type: One of ``'A'``, ``'B'``, ``'C'``, ``'D'``, ``'E'``.
        checkpoint: Path to ``.pth`` checkpoint.  Defaults to
            ``checkpoints/model_X_best.pth``.
        num_samples: Limit evaluation to this many samples (``None`` = all).
        beam_width: ``1`` for greedy decode; ``> 1`` for beam search.
        no_repeat_ngram_size: N-gram blocking size for beam search (``0`` = off).

    Returns:
        Dict with keys ``bleu1``, ``bleu2``, ``bleu3``, ``bleu4``, ``meteor``,
        ``bertscore``, ``exact_match`` (exact match as percentage).
    """
    if checkpoint is None:
        checkpoint = f"checkpoints/model_{model_type.lower()}_best.pth"

    vocab_q = Vocabulary()
    vocab_q.load(VOCAB_Q_PATH)
    vocab_a = Vocabulary()
    vocab_a.load(VOCAB_A_PATH)

    val_dataset = VQAEDataset(
        image_dir=VAL_IMAGE_DIR,
        vqa_e_json_path=VAL_VQA_E_JSON,
        vocab_q=vocab_q,
        vocab_a=vocab_a,
        split="val2014",
        max_samples=num_samples,
    )

    model = load_model_from_checkpoint(
        model_type, checkpoint, len(vocab_q), len(vocab_a), device=DEVICE
    )

    # ── Decode function routing ───────────────────────────────────────────────
    # Spatial-attention models (C, D, E) use the attention-aware decode path;
    # global-vector models (A, B) use the simpler path.
    use_attention = model_type in ("C", "D", "E")

    if beam_width > 1:
        decode_fn = (
            batch_beam_search_decode_with_attention if use_attention
            else batch_beam_search_decode
        )
        decode_kwargs: dict = dict(
            beam_width=beam_width,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
    else:
        decode_fn = (
            batch_greedy_decode_with_attention if use_attention
            else batch_greedy_decode
        )
        decode_kwargs = {}

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=vqa_collate_fn,
        num_workers=2,
    )

    smoothie        = SmoothingFunction().method1
    all_predictions: list[str] = []
    all_gt_strings:  list[str] = []
    n = len(val_dataset)

    print(f"Evaluating Model {model_type} | checkpoint: {checkpoint} | samples: {n}")

    with torch.no_grad():
        for imgs, questions, answers in tqdm.tqdm(val_loader, desc="Evaluating"):
            preds = decode_fn(model, imgs, questions, vocab_a, device=DEVICE, **decode_kwargs)
            all_predictions.extend(preds)
            for a_tensor in answers:
                all_gt_strings.append(_decode_answer_tensor(a_tensor, vocab_a))

    # ── Metric accumulation ───────────────────────────────────────────────────
    exact_match  = 0
    bleu1_total  = 0.0
    bleu2_total  = 0.0
    bleu3_total  = 0.0
    bleu4_total  = 0.0
    meteor_total = 0.0

    for pred_str, gt_str in zip(all_predictions, all_gt_strings):
        pred_clean = pred_str.strip().lower()
        gt_clean   = gt_str.strip().lower()

        if pred_clean == gt_clean:
            exact_match += 1

        gt_words   = gt_str.split()   or ["<unk>"]
        pred_words = pred_str.split() or ["<unk>"]

        bleu1_total  += sentence_bleu([gt_words], pred_words,
                                      weights=(1, 0, 0, 0), smoothing_function=smoothie)
        bleu2_total  += sentence_bleu([gt_words], pred_words,
                                      weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        bleu3_total  += sentence_bleu([gt_words], pred_words,
                                      weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothie)
        bleu4_total  += sentence_bleu([gt_words], pred_words,
                                      weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        meteor_total += meteor_score([gt_words], pred_words)

    # ── BERTScore (optional) ──────────────────────────────────────────────────
    bertscore_f1 = 0.0
    if _HAS_BERTSCORE:
        print("Computing BERTScore (may take a few minutes) …")
        _, _, F1 = bert_score_fn(all_predictions, all_gt_strings, lang="en", verbose=False)
        bertscore_f1 = F1.mean().item()

    # ── Results ───────────────────────────────────────────────────────────────
    decode_mode = f"beam (width={beam_width})" if beam_width > 1 else "greedy"

    print(f"\n{'=' * 50}")
    print(f"Model        : {model_type}")
    print(f"Checkpoint   : {checkpoint}")
    print(f"Samples      : {n}")
    print(f"Decode Mode  : {decode_mode}")
    print(f"{'-' * 50}")
    print(f"BLEU-4  [★]  : {bleu4_total / n:.4f}")
    print(f"METEOR  [★]  : {meteor_total / n:.4f}")
    if _HAS_BERTSCORE:
        print(f"BERTScore[★] : {bertscore_f1:.4f}")
    else:
        print("BERTScore    : N/A  (pip install bert-score to enable)")
    print(f"BLEU-1       : {bleu1_total / n:.4f}")
    print(f"BLEU-2       : {bleu2_total / n:.4f}")
    print(f"BLEU-3       : {bleu3_total / n:.4f}")
    print(f"Exact Match  : {exact_match / n * 100:.2f}%")
    print(f"{'=' * 50}\n")

    return {
        "model_type":  model_type,
        "bleu1":       bleu1_total  / n,
        "bleu2":       bleu2_total  / n,
        "bleu3":       bleu3_total  / n,
        "bleu4":       bleu4_total  / n,
        "meteor":      meteor_total / n,
        "bertscore":   bertscore_f1,
        "exact_match": exact_match  / n * 100,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a VQA checkpoint on NLP metrics."
    )
    parser.add_argument("--model_type",    type=str, default="A",
                        choices=["A", "B", "C", "D", "E"],
                        help="Model variant.")
    parser.add_argument("--checkpoint",    type=str, default=None,
                        help="Path to checkpoint. Default: checkpoints/model_X_best.pth")
    parser.add_argument("--num_samples",   type=int, default=None,
                        help="Limit evaluation to N samples (default: all).")
    parser.add_argument("--beam_width",    type=int, default=1,
                        help="Beam width. 1 = greedy (default), >1 = beam search.")
    parser.add_argument("--no_repeat_ngram", type=int, default=3,
                        help="Block repeated n-grams in beam search (0=disabled, default=3).")
    args = parser.parse_args()

    evaluate(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        num_samples=args.num_samples,
        beam_width=args.beam_width,
        no_repeat_ngram_size=args.no_repeat_ngram,
    )
