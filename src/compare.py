"""compare.py — Evaluate multiple VQA models on the same validation split.

Loads each requested model from its checkpoint, runs inference on the shared
validation set, and prints a side-by-side metric comparison table with
BLEU-1/2/3/4, METEOR, BERTScore, and Exact Match.

Usage
-----
    # Compare all 5 models, greedy decode
    python src/compare.py

    # Specific models + beam search
    python src/compare.py --models C,D,E --beam_width 5

    # Quick sanity check on 100 samples
    python src/compare.py --models E --num_samples 100
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import nltk
import torch
import tqdm
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score as nltk_meteor
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

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
        vocab: Answer vocabulary.

    Returns:
        Space-joined string of content words.
    """
    skip = {vocab.pad_idx, vocab.start_idx, vocab.end_idx}
    words = [vocab.idx2word[int(i)] for i in a_tensor if int(i) not in skip]
    return " ".join(words)


# ── Per-model evaluation ──────────────────────────────────────────────────────

def evaluate_one_model(
    model_type: str,
    epoch: int,
    vocab_q: Vocabulary,
    vocab_a: Vocabulary,
    val_dataset: VQAEDataset,
    beam_width: int = 1,
    no_repeat_ngram_size: int = 3,
) -> Optional[Dict]:
    """Evaluate a single model and return its metric dict.

    Tries ``checkpoints/model_X_epoch{epoch}.pth`` first, then falls back to
    ``checkpoints/model_X_best.pth``.

    Args:
        model_type: One of ``'A'``–``'E'``.
        epoch: Epoch number for checkpoint filename.
        vocab_q: Question vocabulary.
        vocab_a: Answer vocabulary.
        val_dataset: Validation dataset instance (shared across all models).
        beam_width: ``1`` = greedy; ``> 1`` = beam search.
        no_repeat_ngram_size: N-gram blocking size for beam search.

    Returns:
        Dict of metric scores, or ``None`` if no checkpoint was found.
    """
    checkpoint = f"checkpoints/model_{model_type.lower()}_epoch{epoch}.pth"

    if not os.path.exists(checkpoint):
        best_ckpt = f"checkpoints/model_{model_type.lower()}_best.pth"
        if os.path.exists(best_ckpt):
            print(f"  [INFO] {checkpoint} not found — using {best_ckpt}")
            checkpoint = best_ckpt
        else:
            print(f"  [SKIP] {checkpoint} not found (no best checkpoint either).")
            return None

    model = load_model_from_checkpoint(
        model_type, checkpoint, len(vocab_q), len(vocab_a), device=DEVICE
    )

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
        val_dataset, batch_size=64, shuffle=False,
        collate_fn=vqa_collate_fn, num_workers=2,
    )

    smoothie        = SmoothingFunction().method1
    all_predictions: List[str] = []
    all_gt_strings:  List[str] = []

    with torch.no_grad():
        for imgs, questions, answers in tqdm.tqdm(val_loader, desc=f"Model {model_type}", leave=False):
            preds = decode_fn(model, imgs, questions, vocab_a, device=DEVICE, **decode_kwargs)
            all_predictions.extend(preds)
            for a_tensor in answers:
                all_gt_strings.append(_decode_answer_tensor(a_tensor, vocab_a))

    n            = len(all_predictions)
    exact_match  = 0
    bleu1_total  = 0.0
    bleu2_total  = 0.0
    bleu3_total  = 0.0
    bleu4_total  = 0.0
    meteor_total = 0.0

    for pred_str, gt_str in zip(all_predictions, all_gt_strings):
        if pred_str.strip().lower() == gt_str.strip().lower():
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
        meteor_total += nltk_meteor([gt_words], pred_words)

    bertscore_f1 = 0.0
    if _HAS_BERTSCORE:
        _, _, F1 = bert_score_fn(all_predictions, all_gt_strings, lang="en", verbose=False)
        bertscore_f1 = F1.mean().item()

    return {
        "exact_match": exact_match  / n * 100,
        "bleu1":       bleu1_total  / n,
        "bleu2":       bleu2_total  / n,
        "bleu3":       bleu3_total  / n,
        "bleu4":       bleu4_total  / n,
        "meteor":      meteor_total / n,
        "bertscore":   bertscore_f1,
        "checkpoint":  checkpoint,
        "n":           n,
    }


# ── Table printing ─────────────────────────────────────────────────────────────

def print_table(results: Dict[str, Optional[Dict]]) -> None:
    """Print a formatted comparison table to stdout.

    Args:
        results: Dict mapping model type string to metric dict (or ``None``
            if the checkpoint was missing).
    """
    header = f"{'Model':<8} {'BLEU-4':>8} {'METEOR':>8}"
    if _HAS_BERTSCORE:
        header += f" {'BERTScr':>8}"
    header += f" {'BLEU-1':>8} {'BLEU-2':>8} {'BLEU-3':>8} {'Exact':>8}  Checkpoint"

    print()
    print(header)
    print("-" * len(header))

    for model_type, r in sorted(results.items()):
        if r is None:
            n_cols = 7 if _HAS_BERTSCORE else 6
            print(f"{model_type:<8} {'N/A':>8} " * n_cols + " (checkpoint missing)")
        else:
            line = (
                f"{model_type:<8}"
                f" {r['bleu4']:>8.4f}"
                f" {r['meteor']:>8.4f}"
            )
            if _HAS_BERTSCORE:
                line += f" {r['bertscore']:>8.4f}"
            line += (
                f" {r['bleu1']:>8.4f}"
                f" {r['bleu2']:>8.4f}"
                f" {r['bleu3']:>8.4f}"
                f" {r['exact_match']:>7.2f}%"
                f"  {r['checkpoint']}"
            )
            print(line)
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments and run the multi-model comparison."""
    parser = argparse.ArgumentParser(
        description="Compare multiple VQA models on the validation set."
    )
    parser.add_argument("--epoch",       type=int, default=10,
                        help="Epoch checkpoint to load (default: 10).")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit evaluation to N samples for speed.")
    parser.add_argument("--models",      type=str, default="A,B,C,D,E",
                        help="Comma-separated models to compare (default: A,B,C,D,E). "
                             "Missing checkpoints are skipped automatically.")
    parser.add_argument("--beam_width",  type=int, default=1,
                        help="Beam width. 1 = greedy (default), >1 = beam search.")
    parser.add_argument("--no_repeat_ngram", type=int, default=3,
                        help="N-gram blocking size for beam search (0=disabled, default=3).")
    args = parser.parse_args()

    model_types = [m.strip().upper() for m in args.models.split(",")]

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
        max_samples=args.num_samples,
    )

    decode_mode = f"beam (w={args.beam_width})" if args.beam_width > 1 else "greedy"
    print(
        f"Comparing: {model_types} | epoch={args.epoch} | "
        f"samples={len(val_dataset)} | decode={decode_mode}"
    )

    results: Dict[str, Optional[Dict]] = {}
    for model_type in model_types:
        results[model_type] = evaluate_one_model(
            model_type, args.epoch, vocab_q, vocab_a, val_dataset,
            beam_width=args.beam_width,
            no_repeat_ngram_size=args.no_repeat_ngram,
        )

    print_table(results)


if __name__ == "__main__":
    main()
