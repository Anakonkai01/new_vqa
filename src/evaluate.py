import torch
import os, sys, json, argparse, tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(__file__))

# Download required NLTK data for METEOR
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from dataset import VQADataset, vqa_collate_fn
from vocab import Vocabulary
from inference import (
    get_model,
    batch_greedy_decode,
    batch_greedy_decode_with_attention,
)

# CONFIGURATION
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

def evaluate(model_type='A', checkpoint=None, num_samples=None):
    if checkpoint is None:
        checkpoint = f"checkpoints/model_{model_type.lower()}_epoch10.pth"

    vocab_q = Vocabulary(); vocab_q.load(VOCAB_Q_PATH)
    vocab_a = Vocabulary(); vocab_a.load(VOCAB_A_PATH)

    val_dataset = VQADataset(
        image_dir=VAL_IMAGE_DIR,
        question_json_path=VAL_QUESTION_JSON,
        annotations_json_path=VAL_ANNOTATION_JSON,
        vocab_q=vocab_q,
        vocab_a=vocab_a,
        split='val2014',
        max_samples=num_samples
    )

    # Load all 10 human annotations per question for proper VQA accuracy:
    # VQA accuracy = min(number_of_matching_annotations / 3, 1.0)
    with open(VAL_ANNOTATION_JSON, 'r') as f:
        raw_annotations = json.load(f)['annotations']
    qid_to_all_answers = {
        ann['question_id']: [a['answer'].lower().strip() for a in ann['answers']]
        for ann in raw_annotations
    }
    # question_ids in dataset order (shuffle=False preserves alignment with DataLoader)
    question_ids = [q['question_id'] for q in val_dataset.questions]

    model = get_model(model_type, len(vocab_q), len(vocab_a))
    model.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage))
    model.to(DEVICE)
    model.eval()

    use_attention = model_type in ('C', 'D')
    decode_fn     = batch_greedy_decode_with_attention if use_attention else batch_greedy_decode

    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False,
        collate_fn=vqa_collate_fn, num_workers=2
    )

    smoothie        = SmoothingFunction().method1
    all_predictions = []
    all_gt_strings  = []

    n = len(val_dataset)
    print(f"Evaluating Model {model_type} | checkpoint: {checkpoint} | samples: {n}")

    with torch.no_grad():
        for imgs, questions, answers in tqdm.tqdm(val_loader, desc="Evaluating"):
            preds = decode_fn(model, imgs, questions, vocab_a, device=DEVICE)
            all_predictions.extend(preds)
            for a_tensor in answers:
                all_gt_strings.append(decode_tensor(a_tensor, vocab_a))

    # Compute all metrics
    exact_match    = 0
    vqa_acc_total  = 0.0
    bleu1_total    = 0.0
    bleu2_total    = 0.0
    bleu3_total    = 0.0
    bleu4_total    = 0.0
    meteor_total   = 0.0

    for idx, (pred_str, gt_str) in enumerate(zip(all_predictions, all_gt_strings)):
        pred_clean = pred_str.strip().lower()
        gt_clean   = gt_str.strip().lower()

        if pred_clean == gt_clean:
            exact_match += 1

        # VQA accuracy: compare prediction against all 10 human annotations
        qid         = question_ids[idx]
        all_answers = qid_to_all_answers.get(qid, [gt_clean])
        match_count = sum(1 for a in all_answers if a == pred_clean)
        vqa_acc_total += min(match_count / 3.0, 1.0)

        gt_words   = gt_str.split() or ['<unk>']
        pred_words = pred_str.split() or ['<unk>']
        bleu1_total  += sentence_bleu([gt_words], pred_words, weights=(1, 0, 0, 0),             smoothing_function=smoothie)
        bleu2_total  += sentence_bleu([gt_words], pred_words, weights=(0.5, 0.5, 0, 0),         smoothing_function=smoothie)
        bleu3_total  += sentence_bleu([gt_words], pred_words, weights=(1/3, 1/3, 1/3, 0),       smoothing_function=smoothie)
        bleu4_total  += sentence_bleu([gt_words], pred_words, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        meteor_total += meteor_score([gt_words], pred_words)

    print(f"\n{'='*50}")
    print(f"Model        : {model_type}")
    print(f"Checkpoint   : {checkpoint}")
    print(f"Samples      : {n}")
    print(f"{'-'*50}")
    print(f"VQA Accuracy : {vqa_acc_total/n*100:.2f}%")
    print(f"Exact Match  : {exact_match/n*100:.2f}%")
    print(f"BLEU-1       : {bleu1_total/n:.4f}")
    print(f"BLEU-2       : {bleu2_total/n:.4f}")
    print(f"BLEU-3       : {bleu3_total/n:.4f}")
    print(f"BLEU-4       : {bleu4_total/n:.4f}")
    print(f"METEOR       : {meteor_total/n:.4f}")
    print(f"{'='*50}\n")

    return {
        'model_type':   model_type,
        'vqa_accuracy': vqa_acc_total / n * 100,
        'exact_match':  exact_match / n * 100,
        'bleu1':        bleu1_total / n,
        'bleu2':        bleu2_total / n,
        'bleu3':        bleu3_total / n,
        'bleu4':        bleu4_total / n,
        'meteor':       meteor_total / n,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type',  type=str, default='A', choices=['A', 'B', 'C', 'D'],
                        help='Model architecture to evaluate (A/B/C/D)')
    parser.add_argument('--checkpoint',  type=str, default=None,
                        help='Path to checkpoint. Default: checkpoints/model_X_epoch10.pth')
    parser.add_argument('--num_samples', type=int, default=None)
    args = parser.parse_args()

    evaluate(model_type=args.model_type, checkpoint=args.checkpoint, num_samples=args.num_samples)

