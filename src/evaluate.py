import torch
from torch.utils.data import random_split
import os, sys, argparse, tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
sys.path.append(os.path.dirname(__file__))

from dataset import VQADatasetA
from models.vqa_models import VQAmodelA
from vocab import Vocabulary
from inference import greedy_decode

# ── Config (giống train.py) ──────────────────────────
DEVICE          = 'cpu'
CHECKPOINT      = "checkpoints/model_a_epoch10.pth"
IMAGE_DIR       = "data/raw/images/train2014"
QUESTION_JSON   = "data/raw/vqa_json/v2_OpenEnded_mscoco_train2014_questions.json"
ANNOTATION_JSON = "data/raw/vqa_json/v2_mscoco_train2014_annotations.json"
VOCAB_Q_PATH    = "data/processed/vocab_questions.json"
VOCAB_A_PATH    = "data/processed/vocab_answers.json"
SPLIT_SEED      = 42   # must be the same with train.py
VAL_RATIO       = 0.1  # must be the same with train.py

def decode_tensor(a_tensor, vocab_a):
    special = {
        vocab_a.word2idx['<pad>'],
        vocab_a.word2idx['<start>'],
        vocab_a.word2idx['<end>']
    }
    words = [vocab_a.idx2word[int(i)] for i in a_tensor if int(i) not in special]
    return ' '.join(words)

def evaluate(checkpoint=CHECKPOINT, num_samples=None):
    vocab_q = Vocabulary(); vocab_q.load(VOCAB_Q_PATH)
    vocab_a = Vocabulary(); vocab_a.load(VOCAB_A_PATH)

    # Recreate val split — PHẢI cùng seed với train.py
    dataset    = VQADatasetA(IMAGE_DIR, QUESTION_JSON, ANNOTATION_JSON, vocab_q, vocab_a)
    val_size   = int(VAL_RATIO * len(dataset))
    train_size = len(dataset) - val_size
    generator  = torch.Generator().manual_seed(SPLIT_SEED)
    _, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    model = VQAmodelA(vocab_size=len(vocab_q), answer_vocab_size=len(vocab_a))
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()

    smoothie    = SmoothingFunction().method1
    exact_match = 0
    bleu1_total = 0.0
    n           = num_samples or len(val_dataset)

    for i in tqdm.tqdm(range(n)):
        img_tensor, q_tensor, a_tensor = val_dataset[i]

        gt_str     = decode_tensor(a_tensor, vocab_a)
        pred_str   = greedy_decode(model, img_tensor, q_tensor, vocab_a, device=DEVICE)

        if pred_str.strip() == gt_str.strip():
            exact_match += 1

        gt_words   = gt_str.split() or ['<unk>']
        pred_words = pred_str.split() or ['<unk>']
        bleu1_total += sentence_bleu([gt_words], pred_words,
                                     weights=(1,0,0,0),
                                     smoothing_function=smoothie)

    print(f"Exact Match : {exact_match/n*100:.2f}%")
    print(f"BLEU-1      : {bleu1_total/n:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',  type=str, default=CHECKPOINT)
    parser.add_argument('--num_samples', type=int, default=None)
    args = parser.parse_args()

    evaluate(checkpoint=args.checkpoint, num_samples=args.num_samples)

