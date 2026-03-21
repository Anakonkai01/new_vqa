import torch
import os, sys, argparse, tqdm
import nltk
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(__file__))

# Download required NLTK data for METEOR
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ROUGE-L
try:
    from rouge_score import rouge_scorer as _rouge_scorer
    _rscorer = _rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    print("[WARN] rouge-score not installed. Run: pip install rouge-score")

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
    load_model_from_checkpoint,
    batch_greedy_decode,
    batch_greedy_decode_with_attention,
    batch_beam_search_decode,
    batch_beam_search_decode_with_attention,
    true_batched_beam_search_with_attention,
    true_batched_beam_search,
    strip_compiled_prefix,
)

# CONFIGURATION
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VAL_IMAGE_DIR  = "data/images/val2014"
VAL_VQA_E_JSON = "data/annotations/vqa_e/VQA-E_val_set.json"
VOCAB_JOINT_PATH   = "data/processed/vocab_joint.json"

def decode_tensor(a_tensor, vocab):
    special = {
        vocab.word2idx['<pad>'],
        vocab.word2idx['<start>'],
        vocab.word2idx['<end>']
    }
    words = [vocab.idx2word[int(i)] for i in a_tensor if int(i) not in special]
    return ' '.join(words)

# ── Parallel metric worker ────────────────────────────────────────────────────
def _metric_worker(chunk):
    """Compute partial BLEU/METEOR/ROUGE sums for a chunk of (pred, gt) pairs."""
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    try:
        from rouge_score import rouge_scorer as _rs
        rscorer = _rs.RougeScorer(['rougeL'], use_stemmer=True)
        has_rouge = True
    except ImportError:
        has_rouge = False

    smoothie = SmoothingFunction().method1
    exact = b1 = b2 = b3 = b4 = met = rl = 0.0
    for pred_str, gt_str in chunk:
        pred_clean = pred_str.strip().lower()
        gt_clean   = gt_str.strip().lower()
        if pred_clean == gt_clean:
            exact += 1
        gt_w   = gt_str.split()   or ['<unk>']
        pred_w = pred_str.split() or ['<unk>']
        b1  += sentence_bleu([gt_w], pred_w, weights=(1,0,0,0),           smoothing_function=smoothie)
        b2  += sentence_bleu([gt_w], pred_w, weights=(.5,.5,0,0),         smoothing_function=smoothie)
        b3  += sentence_bleu([gt_w], pred_w, weights=(1/3,1/3,1/3,0),     smoothing_function=smoothie)
        b4  += sentence_bleu([gt_w], pred_w, weights=(.25,.25,.25,.25),   smoothing_function=smoothie)
        met += meteor_score([gt_w], pred_w)
        if has_rouge:
            rl += rscorer.score(gt_str, pred_str)['rougeL'].fmeasure
    return exact, b1, b2, b3, b4, met, rl


def evaluate(model_type='A', checkpoint=None, num_samples=None, beam_width=1,
             no_repeat_ngram_size=3, batch_size=256):
    if checkpoint is None:
        checkpoint = f"checkpoints/model_{model_type.lower()}_epoch10.pth"

    # Models A-D use separate question/answer vocabs; E/F/G use the unified joint vocab
    if model_type in ('E', 'F', 'G'):
        vocab = Vocabulary(); vocab.load(VOCAB_JOINT_PATH)
        dataset_vocab = vocab
        decode_vocab  = vocab
    else:
        # A-D: use original separate vocabs to avoid embedding index out-of-bounds
        dataset_vocab = Vocabulary(); dataset_vocab.load("data/processed/vocab_questions.json")
        decode_vocab  = Vocabulary(); decode_vocab.load("data/processed/vocab_answers.json")
        vocab = dataset_vocab  # for metadata only

    val_dataset = VQAEDataset(
        image_dir=VAL_IMAGE_DIR,
        vqa_e_json_path=VAL_VQA_E_JSON,
        vocab=dataset_vocab,
        split='val2014',
        max_samples=num_samples
    )

    model = load_model_from_checkpoint(
        model_type, checkpoint, len(dataset_vocab), device=DEVICE, vocab=dataset_vocab
    )

    # Select decode function — true batched beam search when beam_width > 1
    use_attention = model_type in ('C', 'D', 'E', 'F', 'G')
    if beam_width > 1:
        if use_attention:
            decode_fn = true_batched_beam_search_with_attention
        else:
            decode_fn = true_batched_beam_search
        decode_kwargs = dict(beam_width=beam_width,
                             no_repeat_ngram_size=no_repeat_ngram_size)
    else:
        decode_fn     = batch_greedy_decode_with_attention if use_attention else batch_greedy_decode
        decode_kwargs = {}

    # Maximize DataLoader throughput
    num_workers = min(mp.cpu_count(), 16)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=vqa_collate_fn, num_workers=num_workers,
        pin_memory=(DEVICE.type == 'cuda'), prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )

    all_predictions = []
    all_gt_strings  = []

    n = len(val_dataset)
    print(f"Evaluating Model {model_type} | checkpoint: {checkpoint} | samples: {n}")

    amp_dtype = torch.bfloat16 if (DEVICE.type == 'cuda' and
                                    torch.cuda.is_bf16_supported()) else torch.float16
    use_amp   = DEVICE.type == 'cuda'

    with torch.no_grad():
        for imgs, questions, answers in tqdm.tqdm(val_loader, desc="Evaluating"):
            with torch.autocast(device_type=DEVICE.type, dtype=amp_dtype, enabled=use_amp):
                preds = decode_fn(model, imgs, questions, decode_vocab,
                                  device=DEVICE, **decode_kwargs)
            all_predictions.extend(preds)
            for a_tensor in answers:
                all_gt_strings.append(decode_tensor(a_tensor, decode_vocab))

    # ── Parallel metric computation ───────────────────────────────────────────
    print("Computing metrics in parallel ...")
    pairs      = list(zip(all_predictions, all_gt_strings))
    n_workers  = min(mp.cpu_count(), 28)
    chunk_size = max(1, len(pairs) // n_workers)
    chunks     = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]

    exact_match  = 0
    bleu1_total  = bleu2_total = bleu3_total = bleu4_total = 0.0
    meteor_total = rougeL_total = 0.0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for ex, b1, b2, b3, b4, met, rl in pool.map(_metric_worker, chunks):
            exact_match  += ex
            bleu1_total  += b1
            bleu2_total  += b2
            bleu3_total  += b3
            bleu4_total  += b4
            meteor_total += met
            rougeL_total += rl

    # BERTScore — semantic similarity via BERT embeddings
    bertscore_f1 = 0.0
    if HAS_BERTSCORE:
        print("Computing BERTScore (this may take a few minutes) ...")
        _, _, F1 = bert_score_fn(all_predictions, all_gt_strings, lang='en', verbose=False)
        bertscore_f1 = F1.mean().item()

    # CIDEr
    cider_score = 0.0
    try:
        from training.cider import compute_cider
        print("Computing CIDEr ...")
        gts = {str(i): [r] for i, r in enumerate(all_gt_strings)}
        res = {str(i): [h] for i, h in enumerate(all_predictions)}
        cider_score, _ = compute_cider(gts, res)
    except Exception as e:
        print(f"[WARN] CIDEr calculation skipped: {e}")

    # Text length & OOV
    total_tokens = sum(len(p.split()) for p in all_predictions)
    unk_tokens   = sum(p.split().count('<unk>') for p in all_predictions)
    avg_len      = total_tokens / max(len(all_predictions), 1)
    oov_rate     = unk_tokens / max(total_tokens, 1)

    print(f"\n{'='*56}")
    print(f"Model        : {model_type}")
    print(f"Checkpoint   : {checkpoint}")
    print(f"Samples      : {n}")
    print(f"Decode Mode  : {'beam (width=' + str(beam_width) + ')' if beam_width > 1 else 'greedy'}")
    print(f"{'-'*56}")
    print(f"BLEU-4  [★]  : {bleu4_total/n:.4f}")
    print(f"METEOR  [★]  : {meteor_total/n:.4f}  (sentence-level; see report §16.9.3 for cross-paper caveat)")
    print(f"CIDEr-D [★]  : {cider_score:.4f}")
    if HAS_ROUGE:
        print(f"ROUGE-L [★]  : {rougeL_total/n:.4f}")
    if HAS_BERTSCORE:
        print(f"BERTScore[★] : {bertscore_f1:.4f}")
    print(f"BLEU-1       : {bleu1_total/n:.4f}")
    print(f"BLEU-2       : {bleu2_total/n:.4f}")
    print(f"BLEU-3       : {bleu3_total/n:.4f}")
    print(f"Exact Match  : {exact_match/n*100:.2f}%")
    print(f"Avg Length   : {avg_len:.2f}")
    print(f"OOV Rate     : {oov_rate*100:.2f}%")
    print(f"{'='*56}\n")

    return {
        'model_type':  model_type,
        'bleu1':       bleu1_total / n,
        'bleu2':       bleu2_total / n,
        'bleu3':       bleu3_total / n,
        'bleu4':       bleu4_total / n,
        'meteor':      meteor_total / n,
        'cider':       cider_score,
        'rougeL':      rougeL_total / n if HAS_ROUGE else None,
        'bertscore':   bertscore_f1,
        'exact_match': exact_match / n * 100,
        'avg_length':  avg_len,
        'oov_rate':    oov_rate,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type',  type=str, default='A', choices=['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    parser.add_argument('--checkpoint',  type=str, default=None,
                        help='Path to checkpoint. Default: checkpoints/model_X_epoch10.pth')
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--beam_width',  type=int, default=1,
                        help='Beam width. 1 = greedy (default), >1 = beam search')
    parser.add_argument('--no_repeat_ngram', type=int, default=3,
                        help='Block repeated n-grams of this size in beam search (0=disabled, default=3)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='DataLoader batch size (default 256)')
    args = parser.parse_args()

    evaluate(model_type=args.model_type, checkpoint=args.checkpoint,
             num_samples=args.num_samples, beam_width=args.beam_width,
             no_repeat_ngram_size=args.no_repeat_ngram, batch_size=args.batch_size)
