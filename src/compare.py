"""
compare.py — Evaluate all 4 models on the same val split, print comparison table.

Usage:
    python src/compare.py
    python src/compare.py --epoch 5           # use a specific epoch
    python src/compare.py --num_samples 50    # only run 50 samples for speed
"""

import torch
import os, sys, argparse, tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as nltk_meteor
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(__file__))

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

try:
    from rouge_score import rouge_scorer as _rouge_scorer
    _rscorer = _rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

try:
    from bert_score import score as bert_score_fn
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False

from dataset import VQAEDataset, vqa_collate_fn
from vocab import Vocabulary
from inference import get_model, load_model_from_checkpoint, \
    batch_greedy_decode, batch_greedy_decode_with_attention, \
    batch_beam_search_decode, batch_beam_search_decode_with_attention, strip_compiled_prefix

# ── Config ─────────────────────────────────────────────────────────
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VAL_IMAGE_DIR  = "data/raw/images/val2014"
VAL_VQA_E_JSON = "data/raw/vqa_e_json/VQA-E_val_set.json"
VOCAB_Q_PATH   = "data/processed/vocab_questions.json"
VOCAB_A_PATH   = "data/processed/vocab_answers.json"


def decode_tensor(a_tensor, vocab_a):
    special = {
        vocab_a.word2idx['<pad>'],
        vocab_a.word2idx['<start>'],
        vocab_a.word2idx['<end>']
    }
    words = [vocab_a.idx2word[int(i)] for i in a_tensor if int(i) not in special]
    return ' '.join(words)


def evaluate_one_model(model_type, epoch, vocab_q, vocab_a, val_dataset, beam_width=1,
                       no_repeat_ngram_size=3):
    checkpoint = f"checkpoints/model_{model_type.lower()}_epoch{epoch}.pth"

    # Fallback to best checkpoint if epoch-specific one doesn't exist
    if not os.path.exists(checkpoint):
        best_ckpt = f"checkpoints/model_{model_type.lower()}_best.pth"
        if os.path.exists(best_ckpt):
            print(f"  [INFO] {checkpoint} not found, using {best_ckpt} (early stopping fallback)")
            checkpoint = best_ckpt
        else:
            print(f"  [SKIP] {checkpoint} not found (no best checkpoint either).")
            return None

    model = load_model_from_checkpoint(
        model_type, checkpoint, len(vocab_q), len(vocab_a), device=DEVICE
    )

    use_attention = model_type in ('C', 'D')
    if beam_width > 1:
        decode_fn     = batch_beam_search_decode_with_attention if use_attention else batch_beam_search_decode
        decode_kwargs = dict(beam_width=beam_width,
                             no_repeat_ngram_size=no_repeat_ngram_size)
    else:
        decode_fn     = batch_greedy_decode_with_attention if use_attention else batch_greedy_decode
        decode_kwargs = {}

    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False,
        collate_fn=vqa_collate_fn, num_workers=2
    )

    smoothie        = SmoothingFunction().method1
    all_predictions = []
    all_gt_strings  = []

    with torch.no_grad():
        for imgs, questions, answers in tqdm.tqdm(val_loader, desc=f"Model {model_type}", leave=False):
            preds = decode_fn(model, imgs, questions, vocab_a, device=DEVICE, **decode_kwargs)
            all_predictions.extend(preds)
            for a_tensor in answers:
                all_gt_strings.append(decode_tensor(a_tensor, vocab_a))

    n             = len(all_predictions)
    exact_match   = 0
    bleu1_total   = 0.0
    bleu2_total   = 0.0
    bleu3_total   = 0.0
    bleu4_total   = 0.0
    meteor_total  = 0.0   # NLTK with WordNet (internal comparison)
    meteor_std    = 0.0   # No WordNet — comparable to Li et al. 2018
    rougeL_total  = 0.0

    for pred_str, gt_str in zip(all_predictions, all_gt_strings):
        pred_clean = pred_str.strip().lower()
        gt_clean   = gt_str.strip().lower()

        if pred_clean == gt_clean:
            exact_match += 1

        gt_words   = gt_str.split() or ['<unk>']
        pred_words = pred_str.split() or ['<unk>']

        bleu1_total  += sentence_bleu([gt_words], pred_words,
                                      weights=(1, 0, 0, 0), smoothing_function=smoothie)
        bleu2_total  += sentence_bleu([gt_words], pred_words,
                                      weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        bleu3_total  += sentence_bleu([gt_words], pred_words,
                                      weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
        bleu4_total  += sentence_bleu([gt_words], pred_words,
                                      weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

        meteor_total += nltk_meteor([gt_words], pred_words)
        meteor_std   += nltk_meteor([gt_words], pred_words, wordnet=None)

        if HAS_ROUGE:
            rougeL_total += _rscorer.score(gt_str, pred_str)['rougeL'].fmeasure

    # BERTScore (computed once per model on full prediction list)
    bertscore_f1 = 0.0
    if HAS_BERTSCORE:
        _, _, F1 = bert_score_fn(all_predictions, all_gt_strings, lang='en', verbose=False)
        bertscore_f1 = F1.mean().item()

    return {
        'exact_match':  exact_match / n * 100,
        'bleu1':        bleu1_total / n,
        'bleu2':        bleu2_total / n,
        'bleu3':        bleu3_total / n,
        'bleu4':        bleu4_total / n,
        'meteor_nltk':  meteor_total / n,
        'meteor':       meteor_std / n,     # primary — no WordNet
        'rougeL':       rougeL_total / n if HAS_ROUGE else None,
        'bertscore':    bertscore_f1,
        'checkpoint':   checkpoint,
        'n':            n,
    }


def print_table(results):
    print()
    # Header
    hdr = f"{'Model':<8} {'BLEU-4':>8} {'METEOR*':>9} {'ROUGE-L':>9}"
    if HAS_BERTSCORE:
        hdr += f" {'BERTScr':>8}"
    hdr += f" {'BLEU-1':>8} {'BLEU-2':>8} {'BLEU-3':>8} {'M-NLTK':>8} {'Exact':>8}  Checkpoint"
    print(hdr)
    print("-" * len(hdr))
    for model_type, r in sorted(results.items()):
        if r is None:
            print(f"{model_type:<8}  (checkpoint missing)")
        else:
            line = (
                f"{model_type:<8}"
                f" {r['bleu4']:>8.4f}"
                f" {r['meteor']:>9.4f}"   # standard METEOR (no WordNet)
                f" {r['rougeL']:>9.4f}" if r['rougeL'] is not None else f" {'N/A':>9}"
            )
            if HAS_BERTSCORE:
                line += f" {r['bertscore']:>8.4f}"
            line += (
                f" {r['bleu1']:>8.4f}"
                f" {r['bleu2']:>8.4f}"
                f" {r['bleu3']:>8.4f}"
                f" {r['meteor_nltk']:>8.4f}"   # NLTK METEOR for reference
                f" {r['exact_match']:>7.2f}%"
                f"  {r['checkpoint']}"
            )
            print(line)
    print()
    print("* METEOR = standard (no WordNet synonyms) — comparable to Li et al. 2018")
    print("  M-NLTK = NLTK METEOR with WordNet — use only for internal model comparison")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',       type=int, default=10,
                        help='Which epoch checkpoint to load (default: 10)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Limit evaluation to N samples for speed')
    parser.add_argument('--models',      type=str, default='A,B,C,D',
                        help='Comma-separated list of models to compare (default: A,B,C,D)')
    parser.add_argument('--beam_width',  type=int, default=1,
                        help='Beam width. 1 = greedy (default), >1 = beam search')
    parser.add_argument('--no_repeat_ngram', type=int, default=3,
                        help='Block repeated n-grams of this size in beam search (0=disabled, default=3)')
    args = parser.parse_args()

    model_types = [m.strip().upper() for m in args.models.split(',')]

    vocab_q = Vocabulary(); vocab_q.load(VOCAB_Q_PATH)
    vocab_a = Vocabulary(); vocab_a.load(VOCAB_A_PATH)

    val_dataset = VQAEDataset(
        image_dir=VAL_IMAGE_DIR,
        vqa_e_json_path=VAL_VQA_E_JSON,
        vocab_q=vocab_q,
        vocab_a=vocab_a,
        split='val2014',
        max_samples=args.num_samples
    )

    n = len(val_dataset)
    print(f"Comparing models: {model_types} | epoch={args.epoch} | samples={n} | decode={'beam (w=' + str(args.beam_width) + ')' if args.beam_width > 1 else 'greedy'}")

    results = {}
    for model_type in model_types:
        results[model_type] = evaluate_one_model(
            model_type, args.epoch, vocab_q, vocab_a, val_dataset,
            beam_width=args.beam_width,
            no_repeat_ngram_size=args.no_repeat_ngram
        )

    print_table(results)


if __name__ == "__main__":
    main()
