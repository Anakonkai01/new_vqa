import torch
import os, sys, argparse, tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(__file__))

# Download required NLTK data for METEOR
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# BERTScore (semantic similarity via BERT embeddings)
try:
    from bert_score import score as bert_score_fn
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False

from dataset import VQAEDataset, vqa_collate_fn
from vocab import Vocabulary
from inference import (
    get_model,
    batch_greedy_decode,
    batch_greedy_decode_with_attention,
    batch_beam_search_decode,
    batch_beam_search_decode_with_attention,
    strip_compiled_prefix,
)

# CONFIGURATION
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VAL_IMAGE_DIR  = "data/raw/val2014"
VAL_VQA_E_JSON = "data/vqa_e/VQA-E_val_set.json"
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

def evaluate(model_type='A', checkpoint=None, num_samples=None, beam_width=1,
             no_repeat_ngram_size=3):
    if checkpoint is None:
        checkpoint = f"checkpoints/model_{model_type.lower()}_epoch10.pth"

    vocab_q = Vocabulary(); vocab_q.load(VOCAB_Q_PATH)
    vocab_a = Vocabulary(); vocab_a.load(VOCAB_A_PATH)

    val_dataset = VQAEDataset(
        image_dir=VAL_IMAGE_DIR,
        vqa_e_json_path=VAL_VQA_E_JSON,
        vocab_q=vocab_q,
        vocab_a=vocab_a,
        split='val2014',
        max_samples=num_samples
    )

    model = get_model(model_type, len(vocab_q), len(vocab_a))
    state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    model.load_state_dict(strip_compiled_prefix(state_dict))
    model.to(DEVICE)
    model.eval()

    use_attention = model_type in ('C', 'D')
    if beam_width > 1:
        decode_fn = (
            batch_beam_search_decode_with_attention if use_attention
            else batch_beam_search_decode
        )
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

    n = len(val_dataset)
    print(f"Evaluating Model {model_type} | checkpoint: {checkpoint} | samples: {n}")

    with torch.no_grad():
        for imgs, questions, answers in tqdm.tqdm(val_loader, desc="Evaluating"):
            preds = decode_fn(model, imgs, questions, vocab_a, device=DEVICE, **decode_kwargs)
            all_predictions.extend(preds)
            for a_tensor in answers:
                all_gt_strings.append(decode_tensor(a_tensor, vocab_a))

    # Compute metrics — primary: BLEU-4 and METEOR (meaningful for generative output)
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
    print(f"Decode Mode  : {'beam (width=' + str(beam_width) + ')' if beam_width > 1 else 'greedy'}")
    print(f"{'-'*50}")
    print(f"BLEU-4  [★]  : {bleu4_total/n:.4f}")
    print(f"METEOR  [★]  : {meteor_total/n:.4f}")

    # BERTScore — semantic similarity via BERT embeddings
    bertscore_f1 = 0.0
    if HAS_BERTSCORE:
        print("Computing BERTScore (this may take a few minutes) ...")
        _, _, F1 = bert_score_fn(all_predictions, all_gt_strings, lang='en', verbose=False)
        bertscore_f1 = F1.mean().item()
        print(f"BERTScore[★]  : {bertscore_f1:.4f}")
    else:
        print("BERTScore     : N/A (pip install bert-score to enable)")

    print(f"BLEU-1       : {bleu1_total/n:.4f}")
    print(f"BLEU-2       : {bleu2_total/n:.4f}")
    print(f"BLEU-3       : {bleu3_total/n:.4f}")
    print(f"Exact Match  : {exact_match/n*100:.2f}%")
    print(f"{'='*50}\n")

    return {
        'model_type':  model_type,
        'bleu1':       bleu1_total / n,
        'bleu2':       bleu2_total / n,
        'bleu3':       bleu3_total / n,
        'bleu4':       bleu4_total / n,
        'meteor':      meteor_total / n,
        'bertscore':   bertscore_f1,
        'exact_match': exact_match / n * 100,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type',  type=str, default='A', choices=['A', 'B', 'C', 'D'])
    parser.add_argument('--checkpoint',  type=str, default=None,
                        help='Path to checkpoint. Default: checkpoints/model_X_epoch10.pth')
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--beam_width',  type=int, default=1,
                        help='Beam width. 1 = greedy (default), >1 = beam search')
    parser.add_argument('--no_repeat_ngram', type=int, default=3,
                        help='Block repeated n-grams of this size in beam search (0=disabled, default=3)')
    args = parser.parse_args()

    evaluate(model_type=args.model_type, checkpoint=args.checkpoint,
             num_samples=args.num_samples, beam_width=args.beam_width,
             no_repeat_ngram_size=args.no_repeat_ngram)
