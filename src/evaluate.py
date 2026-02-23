import torch
import os, sys, argparse, tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
sys.path.append(os.path.dirname(__file__))

# download wordnet nếu chưa có (METEOR cần)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from dataset import VQADatasetA
from models.vqa_models import VQAmodelA, VQAModelB, VQAModelC, VQAModelD
from vocab import Vocabulary
from inference import greedy_decode, greedy_decode_with_attention, get_model

# ── Config ────────────────────────────────────────────────
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
    # checkpoint tự động theo model_type nếu không chỉ định
    if checkpoint is None:
        checkpoint = f"checkpoints/model_{model_type.lower()}_epoch10.pth"

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
        max_samples=num_samples   # giới hạn ở đây — tiết kiệm RAM khi load
    )

    model = get_model(model_type, len(vocab_q), len(vocab_a))
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()

    # chọn decode function theo model type
    use_attention = model_type in ('C', 'D')
    decode_fn = greedy_decode_with_attention if use_attention else greedy_decode

    smoothie      = SmoothingFunction().method1
    exact_match   = 0
    bleu1_total   = 0.0
    bleu2_total   = 0.0
    bleu3_total   = 0.0
    bleu4_total   = 0.0
    meteor_total  = 0.0
    n             = len(val_dataset)   # đã giới hạn bằng max_samples khi khởi tạo dataset

    print(f"Evaluating Model {model_type} | checkpoint: {checkpoint}")
    for i in tqdm.tqdm(range(n)):
        img_tensor, q_tensor, a_tensor = val_dataset[i]

        gt_str   = decode_tensor(a_tensor, vocab_a)
        pred_str = decode_fn(model, img_tensor, q_tensor, vocab_a, device=DEVICE)

        if pred_str.strip() == gt_str.strip():
            exact_match += 1

        gt_words   = gt_str.split() or ['<unk>']
        pred_words = pred_str.split() or ['<unk>']

        bleu1_total  += sentence_bleu([gt_words], pred_words, weights=(1, 0, 0, 0),    smoothing_function=smoothie)
        bleu2_total  += sentence_bleu([gt_words], pred_words, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        bleu3_total  += sentence_bleu([gt_words], pred_words, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothie)
        bleu4_total  += sentence_bleu([gt_words], pred_words, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        meteor_total += meteor_score([gt_words], pred_words)

    print(f"\n{'='*45}")
    print(f"Model       : {model_type}")
    print(f"Checkpoint  : {checkpoint}")
    print(f"Samples     : {n}")
    print(f"{'-'*45}")
    print(f"Exact Match : {exact_match/n*100:.2f}%")
    print(f"BLEU-1      : {bleu1_total/n:.4f}")
    print(f"BLEU-2      : {bleu2_total/n:.4f}")
    print(f"BLEU-3      : {bleu3_total/n:.4f}")
    print(f"BLEU-4      : {bleu4_total/n:.4f}")
    print(f"METEOR      : {meteor_total/n:.4f}")
    print(f"{'='*45}\n")

    return {
        'model_type':  model_type,
        'exact_match': exact_match / n * 100,
        'bleu1':       bleu1_total / n,
        'bleu2':       bleu2_total / n,
        'bleu3':       bleu3_total / n,
        'bleu4':       bleu4_total / n,
        'meteor':      meteor_total / n,
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

