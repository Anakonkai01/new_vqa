"""
compare.py — Evaluate all 4 models on the same val split, print comparison table.

Usage:
    python src/compare.py
    python src/compare.py --epoch 5           # dùng epoch cụ thể
    python src/compare.py --num_samples 50    # chỉ chạy 50 samples cho nhanh
"""

import torch
from torch.utils.data import random_split
import os, sys, argparse, tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
sys.path.append(os.path.dirname(__file__))

from dataset import VQADatasetA
from vocab import Vocabulary
from inference import greedy_decode, greedy_decode_with_attention, get_model

# ── Config (phải khớp train.py) ───────────────────────────────
DEVICE          = 'cpu'
IMAGE_DIR       = "data/raw/images/train2014"
QUESTION_JSON   = "data/raw/vqa_json/v2_OpenEnded_mscoco_train2014_questions.json"
ANNOTATION_JSON = "data/raw/vqa_json/v2_mscoco_train2014_annotations.json"
VOCAB_Q_PATH    = "data/processed/vocab_questions.json"
VOCAB_A_PATH    = "data/processed/vocab_answers.json"
SPLIT_SEED      = 42
VAL_RATIO       = 0.1


def decode_tensor(a_tensor, vocab_a):
    special = {
        vocab_a.word2idx['<pad>'],
        vocab_a.word2idx['<start>'],
        vocab_a.word2idx['<end>']
    }
    words = [vocab_a.idx2word[int(i)] for i in a_tensor if int(i) not in special]
    return ' '.join(words)


def evaluate_one_model(model_type, epoch, vocab_q, vocab_a, val_dataset, num_samples):
    checkpoint = f"checkpoints/model_{model_type.lower()}_epoch{epoch}.pth"

    if not os.path.exists(checkpoint):
        print(f"  [SKIP] {checkpoint} not found.")
        return None

    model = get_model(model_type, len(vocab_q), len(vocab_a))
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()

    decode_fn = greedy_decode_with_attention if model_type in ('C', 'D') else greedy_decode

    smoothie    = SmoothingFunction().method1
    exact_match = 0
    bleu1_total = 0.0
    n           = num_samples or len(val_dataset)

    for i in tqdm.tqdm(range(n), desc=f"Model {model_type}", leave=False):
        img_tensor, q_tensor, a_tensor = val_dataset[i]

        gt_str   = decode_tensor(a_tensor, vocab_a)
        pred_str = decode_fn(model, img_tensor, q_tensor, vocab_a, device=DEVICE)

        if pred_str.strip() == gt_str.strip():
            exact_match += 1

        gt_words   = gt_str.split() or ['<unk>']
        pred_words = pred_str.split() or ['<unk>']
        bleu1_total += sentence_bleu([gt_words], pred_words,
                                     weights=(1, 0, 0, 0),
                                     smoothing_function=smoothie)

    return {
        'exact_match': exact_match / n * 100,
        'bleu1':       bleu1_total / n,
        'checkpoint':  checkpoint,
        'n':           n,
    }


def print_table(results):
    print()
    print(f"{'Model':<8} {'Exact Match':>12} {'BLEU-1':>10}  Checkpoint")
    print("-" * 70)
    for model_type, r in sorted(results.items()):
        if r is None:
            print(f"{model_type:<8} {'N/A':>12} {'N/A':>10}  (checkpoint missing)")
        else:
            print(f"{model_type:<8} {r['exact_match']:>11.2f}% {r['bleu1']:>10.4f}  {r['checkpoint']}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',       type=int, default=10,
                        help='Which epoch checkpoint to load (default: 10)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Limit evaluation to N samples for speed')
    parser.add_argument('--models',      type=str, default='A,B,C,D',
                        help='Comma-separated list of models to compare (default: A,B,C,D)')
    args = parser.parse_args()

    model_types = [m.strip().upper() for m in args.models.split(',')]

    # load vocab
    vocab_q = Vocabulary(); vocab_q.load(VOCAB_Q_PATH)
    vocab_a = Vocabulary(); vocab_a.load(VOCAB_A_PATH)

    # Dùng val set chính thức VQA 2.0 (val2014)
    val_dataset = VQADatasetA(
        image_dir=VAL_IMAGE_DIR,
        question_json_path=VAL_QUESTION_JSON,
        annotations_json_path=VAL_ANNOTATION_JSON,
        vocab_q=vocab_q,
        vocab_a=vocab_a,
        split='val2014',
        max_samples=args.num_samples
    )

    n = len(val_dataset)
    print(f"Comparing models: {model_types} | epoch={args.epoch} | samples={n}")

    results = {}
    for model_type in model_types:
        results[model_type] = evaluate_one_model(
            model_type, args.epoch, vocab_q, vocab_a, val_dataset, args.num_samples
        )

    print_table(results)


if __name__ == "__main__":
    main()
