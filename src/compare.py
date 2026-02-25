"""
compare.py — Evaluate all 4 models on the same val split, print comparison table.

Usage:
    python src/compare.py
    python src/compare.py --epoch 5           # use a specific epoch
    python src/compare.py --num_samples 50    # only run 50 samples for speed
"""

import torch
import os, sys, json, argparse, tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as nltk_meteor
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(__file__))

from dataset import VQADataset, vqa_collate_fn
from vocab import Vocabulary
from inference import get_model, batch_greedy_decode, batch_greedy_decode_with_attention

# ── Config (must match evaluate.py paths) ─────────────────────
DEVICE             = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VAL_IMAGE_DIR      = "data/raw/images/val2014"
VAL_QUESTION_JSON  = "data/raw/vqa_json/v2_OpenEnded_mscoco_val2014_questions.json"
VAL_ANNOTATION_JSON= "data/raw/vqa_json/v2_mscoco_val2014_annotations.json"
VOCAB_Q_PATH       = "data/processed/vocab_questions.json"
VOCAB_A_PATH       = "data/processed/vocab_answers.json"


def decode_tensor(a_tensor, vocab_a):
    special = {
        vocab_a.word2idx['<pad>'],
        vocab_a.word2idx['<start>'],
        vocab_a.word2idx['<end>']
    }
    words = [vocab_a.idx2word[int(i)] for i in a_tensor if int(i) not in special]
    return ' '.join(words)


def evaluate_one_model(model_type, epoch, vocab_q, vocab_a, val_dataset,
                       qid_to_all_answers, question_ids, num_samples):
    checkpoint = f"checkpoints/model_{model_type.lower()}_epoch{epoch}.pth"

    if not os.path.exists(checkpoint):
        print(f"  [SKIP] {checkpoint} not found.")
        return None

    model = get_model(model_type, len(vocab_q), len(vocab_a))
    model.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage))
    model.to(DEVICE)
    model.eval()

    decode_fn = batch_greedy_decode_with_attention if model_type in ('C', 'D') else batch_greedy_decode

    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False,
        collate_fn=vqa_collate_fn, num_workers=2
    )

    smoothie        = SmoothingFunction().method1
    all_predictions = []
    all_gt_strings  = []

    with torch.no_grad():
        for imgs, questions, answers in tqdm.tqdm(val_loader, desc=f"Model {model_type}", leave=False):
            preds = decode_fn(model, imgs, questions, vocab_a, device=DEVICE)
            all_predictions.extend(preds)
            for a_tensor in answers:
                all_gt_strings.append(decode_tensor(a_tensor, vocab_a))

    n           = len(all_predictions)
    exact_match = 0
    vqa_acc     = 0.0
    bleu1_total = 0.0
    bleu2_total = 0.0
    bleu3_total = 0.0
    bleu4_total = 0.0
    meteor_total = 0.0

    for idx, (pred_str, gt_str) in enumerate(zip(all_predictions, all_gt_strings)):
        pred_clean = pred_str.strip().lower()
        gt_clean   = gt_str.strip().lower()

        if pred_clean == gt_clean:
            exact_match += 1

        qid         = question_ids[idx]
        all_answers = qid_to_all_answers.get(qid, [gt_clean])
        match_count = sum(1 for a in all_answers if a == pred_clean)
        vqa_acc += min(match_count / 3.0, 1.0)

        gt_words   = gt_str.split() or ['<unk>']
        pred_words = pred_str.split() or ['<unk>']
        bleu1_total  += sentence_bleu([gt_words], pred_words,
                                      weights=(1, 0, 0, 0),
                                      smoothing_function=smoothie)
        bleu2_total  += sentence_bleu([gt_words], pred_words,
                                      weights=(0.5, 0.5, 0, 0),
                                      smoothing_function=smoothie)
        bleu3_total  += sentence_bleu([gt_words], pred_words,
                                      weights=(0.33, 0.33, 0.33, 0),
                                      smoothing_function=smoothie)
        bleu4_total  += sentence_bleu([gt_words], pred_words,
                                      weights=(0.25, 0.25, 0.25, 0.25),
                                      smoothing_function=smoothie)
        meteor_total += nltk_meteor([gt_words], pred_words)

    return {
        'vqa_accuracy': vqa_acc / n * 100,
        'exact_match':  exact_match / n * 100,
        'bleu1':        bleu1_total / n,
        'bleu2':        bleu2_total / n,
        'bleu3':        bleu3_total / n,
        'bleu4':        bleu4_total / n,
        'meteor':       meteor_total / n,
        'checkpoint':   checkpoint,
        'n':            n,
    }


def print_table(results):
    header = f"{'Model':<8} {'VQA Acc':>9} {'Exact':>8} {'BLEU-1':>8} {'BLEU-2':>8} {'BLEU-3':>8} {'BLEU-4':>8} {'METEOR':>8}  Checkpoint"
    print()
    print(header)
    print("-" * len(header))
    for model_type, r in sorted(results.items()):
        if r is None:
            print(f"{model_type:<8} {'N/A':>9} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}  (checkpoint missing)")
        else:
            print(
                f"{model_type:<8}"
                f" {r['vqa_accuracy']:>8.2f}%"
                f" {r['exact_match']:>7.2f}%"
                f" {r['bleu1']:>8.4f}"
                f" {r['bleu2']:>8.4f}"
                f" {r['bleu3']:>8.4f}"
                f" {r['bleu4']:>8.4f}"
                f" {r['meteor']:>8.4f}"
                f"  {r['checkpoint']}"
            )
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

    # Load vocab
    vocab_q = Vocabulary(); vocab_q.load(VOCAB_Q_PATH)
    vocab_a = Vocabulary(); vocab_a.load(VOCAB_A_PATH)

    # Official VQA 2.0 val split
    val_dataset = VQADataset(
        image_dir=VAL_IMAGE_DIR,
        question_json_path=VAL_QUESTION_JSON,
        annotations_json_path=VAL_ANNOTATION_JSON,
        vocab_q=vocab_q,
        vocab_a=vocab_a,
        split='val2014',
        max_samples=args.num_samples
    )

    # Load all 10 human annotations per question for VQA accuracy
    with open(VAL_ANNOTATION_JSON, 'r') as f:
        raw_annotations = json.load(f)['annotations']
    qid_to_all_answers = {
        ann['question_id']: [a['answer'].lower().strip() for a in ann['answers']]
        for ann in raw_annotations
    }
    question_ids = [q['question_id'] for q in val_dataset.questions]

    n = len(val_dataset)
    print(f"Comparing models: {model_types} | epoch={args.epoch} | samples={n}")

    results = {}
    for model_type in model_types:
        results[model_type] = evaluate_one_model(
            model_type, args.epoch, vocab_q, vocab_a, val_dataset,
            qid_to_all_answers, question_ids, args.num_samples
        )

    print_table(results)


if __name__ == "__main__":
    main()
