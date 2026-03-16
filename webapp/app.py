#!/usr/bin/env python3
"""
VQA Interactive Web Demo
Run from project root:
    python webapp/app.py
Then open: http://localhost:5000
"""

import sys, os, io, base64, json, time, random, threading, glob
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm

from flask import Flask, request, jsonify, render_template, send_file, abort

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from vocab import Vocabulary
from inference import load_model_from_checkpoint

CHECKPOINTS = {
    'A': PROJECT_ROOT / 'checkpoints' / 'model_a_best.pth',
    'B': PROJECT_ROOT / 'checkpoints' / 'model_b_best.pth',
    'C': PROJECT_ROOT / 'checkpoints' / 'model_c_best.pth',
    'D': PROJECT_ROOT / 'checkpoints' / 'model_d_best.pth',
}
VAL_IMAGE_DIR  = PROJECT_ROOT / 'data' / 'raw' / 'val2014'
VOCAB_Q_PATH   = PROJECT_ROOT / 'data' / 'processed' / 'vocab_questions.json'
VOCAB_A_PATH   = PROJECT_ROOT / 'data' / 'processed' / 'vocab_answers.json'

MODEL_META = {
    'A': {'name': 'Model A',  'desc': 'Scratch CNN · No Attention',    'color': '#ef4444'},
    'B': {'name': 'Model B',  'desc': 'ResNet101 · No Attention',      'color': '#f97316'},
    'C': {'name': 'Model C',  'desc': 'Scratch CNN · Dual Attention',  'color': '#3b82f6'},
    'D': {'name': 'Model D',  'desc': 'ResNet101 · Dual Attention',    'color': '#8b5cf6'},
}

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
IMG_DENORM = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

# ── Globals ───────────────────────────────────────────────────────────────────
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_q       = None
vocab_a       = None
models        = {}
load_status   = {}   # 'loading' | 'loaded' | 'missing' | 'error:<msg>'
_sample_cache = []   # list of (filename, base64_thumb)

app = Flask(__name__, template_folder='templates')


# ── Model Loading ─────────────────────────────────────────────────────────────
def _load_all():
    global vocab_q, vocab_a, models, load_status
    try:
        vocab_q = Vocabulary(); vocab_q.load(str(VOCAB_Q_PATH))
        vocab_a = Vocabulary(); vocab_a.load(str(VOCAB_A_PATH))
    except Exception as e:
        print(f"[ERROR] Vocab load failed: {e}")
        return

    for mt, ckpt in CHECKPOINTS.items():
        load_status[mt] = 'loading'
        if not ckpt.exists():
            load_status[mt] = 'missing'
            print(f"[WARN] {mt}: checkpoint not found at {ckpt}")
            continue
        try:
            t0 = time.time()
            m = load_model_from_checkpoint(
                model_type=mt, checkpoint=str(ckpt),
                vocab_q_size=len(vocab_q), vocab_a_size=len(vocab_a),
                device=device, glove_dim=300
            )
            m.eval()
            models[mt] = m
            load_status[mt] = 'loaded'
            print(f"[OK] Model {mt} loaded in {time.time()-t0:.1f}s on {device}")
        except Exception as e:
            load_status[mt] = f'error:{e}'
            print(f"[ERROR] Model {mt}: {e}")


threading.Thread(target=_load_all, daemon=True).start()


# ── Inference Helpers ─────────────────────────────────────────────────────────
def _pil_to_b64(pil_img, fmt='JPEG', quality=85):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _make_heatmap(orig_pil_224, alpha_49, blend=0.55, cmap_name='inferno'):
    """Overlay a (49,) attention array as a heatmap on the 224×224 PIL image."""
    arr = np.array(alpha_49, dtype=np.float32).reshape(7, 7)
    # Upsample 7×7 → 224×224 via PIL bilinear
    arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr_u8   = (arr_norm * 255).astype(np.uint8)
    heat_small = Image.fromarray(arr_u8, mode='L')
    heat_large = heat_small.resize((224, 224), Image.BILINEAR)
    # Apply colormap
    cmap    = cm.get_cmap(cmap_name)
    heat_np = np.array(heat_large, dtype=np.float32) / 255.0
    heat_rgb = (cmap(heat_np)[:, :, :3] * 255).astype(np.uint8)
    heat_pil = Image.fromarray(heat_rgb, mode='RGB')
    # Alpha blend
    orig_rgb = orig_pil_224.convert('RGB')
    blended  = Image.blend(orig_rgb, heat_pil, alpha=blend)
    return _pil_to_b64(blended)


def _decode_ab(model, img_t, q_t, max_len=30):
    """Greedy decode for Model A/B. Returns tokens, probs, top5_last."""
    model.eval()
    with torch.no_grad():
        img      = img_t.unsqueeze(0).to(device)
        question = q_t.unsqueeze(0).to(device)
        img_feat = F.normalize(model.i_encoder(img), p=2, dim=1)
        q_feat, _= model.q_encoder(question)
        fusion   = model.fusion(img_feat, q_feat)
        h0       = fusion.unsqueeze(0).repeat(model.num_layers, 1, 1)
        c0       = torch.zeros_like(h0)
        hidden   = (h0, c0)

        start = vocab_a.word2idx['<start>']
        end   = vocab_a.word2idx['<end>']
        token = torch.tensor([[start]], dtype=torch.long, device=device)

        tokens, probs, top5_last = [], [], []
        for step in range(max_len):
            emb = model.decoder.embedding(token)
            if model.decoder.embed_proj is not None:
                emb = model.decoder.embed_proj(emb)
            out, hidden = model.decoder.lstm(emb, hidden)
            logit = model.decoder.fc(model.decoder.out_proj(out.squeeze(1)))
            p     = torch.softmax(logit[0], dim=-1)
            pred  = p.argmax().item()
            if pred == end:
                break
            tokens.append(vocab_a.idx2word.get(pred, '<unk>'))
            probs.append(round(p[pred].item(), 4))
            if step == max_len - 1 or pred == end:
                top5_v, top5_i = p.topk(5)
                top5_last = [
                    {'word': vocab_a.idx2word.get(i.item(), '<unk>'), 'prob': round(v.item(), 4)}
                    for v, i in zip(top5_v, top5_i)
                ]
            token = torch.tensor([[pred]], dtype=torch.long, device=device)

        # top-5 at last step
        if not top5_last:
            p_last = torch.softmax(logit[0], dim=-1)
            top5_v, top5_i = p_last.topk(5)
            top5_last = [
                {'word': vocab_a.idx2word.get(i.item(), '<unk>'), 'prob': round(v.item(), 4)}
                for v, i in zip(top5_v, top5_i)
            ]
        return tokens, probs, top5_last


def _decode_cd(model, img_t, q_t, orig_pil_224, max_len=30):
    """Greedy decode for Model C/D with attention. Returns tokens, probs, heatmaps, top5_last."""
    model.eval()
    with torch.no_grad():
        img      = img_t.unsqueeze(0).to(device)
        question = q_t.unsqueeze(0).to(device)
        img_feats = F.normalize(model.i_encoder(img), p=2, dim=-1)  # (1,49,1024)
        q_feat, q_hidden = model.q_encoder(question)
        img_mean = img_feats.mean(dim=1)
        fusion   = model.fusion(img_mean, q_feat)
        h0       = fusion.unsqueeze(0).repeat(model.num_layers, 1, 1)
        c0       = torch.zeros_like(h0)
        hidden   = (h0, c0)

        start    = vocab_a.word2idx['<start>']
        end      = vocab_a.word2idx['<end>']
        token    = torch.tensor([[start]], dtype=torch.long, device=device)
        coverage = None

        tokens, probs, alphas, top5_last = [], [], [], []
        for step in range(max_len):
            logit, hidden, alpha, coverage = model.decoder.decode_step(
                token, hidden, img_feats, q_hidden, coverage)
            p    = torch.softmax(logit[0], dim=-1)
            pred = p.argmax().item()
            if pred == end:
                break
            tokens.append(vocab_a.idx2word.get(pred, '<unk>'))
            probs.append(round(p[pred].item(), 4))
            alphas.append(alpha[0].cpu().numpy())  # (49,)
            token = torch.tensor([[pred]], dtype=torch.long, device=device)

        # top-5 at last real step
        p_last = torch.softmax(logit[0], dim=-1)
        top5_v, top5_i = p_last.topk(5)
        top5_last = [
            {'word': vocab_a.idx2word.get(i.item(), '<unk>'), 'prob': round(v.item(), 4)}
            for v, i in zip(top5_v, top5_i)
        ]

    # Generate heatmaps per token
    heatmaps = [_make_heatmap(orig_pil_224, a) for a in alphas] if alphas else []

    # Average attention heatmap
    if alphas:
        avg_alpha = np.stack(alphas).mean(axis=0)
        avg_heatmap = _make_heatmap(orig_pil_224, avg_alpha)
        # Per-token raw weights for the JS radar/bar chart
        weights = [a.tolist() for a in alphas]
    else:
        avg_heatmap = None
        weights = []

    return tokens, probs, heatmaps, avg_heatmap, weights, top5_last


def _decode_beam(model_type, model, img_t, q_t, beam_width=3, max_len=30):
    """Beam search decode — returns best answer string and score."""
    from inference import (batch_beam_search_decode,
                           batch_beam_search_decode_with_attention)
    img_batch = img_t.unsqueeze(0)
    q_batch   = q_t.unsqueeze(0)
    if model_type in ('C', 'D'):
        answers = batch_beam_search_decode_with_attention(
            model, img_batch, q_batch, vocab_a,
            beam_width=beam_width, max_len=max_len,
            device=device, no_repeat_ngram_size=3)
    else:
        answers = batch_beam_search_decode(
            model, img_batch, q_batch, vocab_a,
            beam_width=beam_width, max_len=max_len,
            device=device, no_repeat_ngram_size=3)
    return answers[0] if answers else ''


# ── Image Sampling ────────────────────────────────────────────────────────────
def _get_sample_images(n=24):
    """Return up to n random val images as (filename, base64_thumbnail)."""
    global _sample_cache
    if _sample_cache:
        return _sample_cache
    pattern = str(VAL_IMAGE_DIR / '*.jpg')
    all_imgs = glob.glob(pattern)
    if not all_imgs:
        return []
    chosen = random.sample(all_imgs, min(n, len(all_imgs)))
    result = []
    for path in chosen:
        try:
            pil = Image.open(path).convert('RGB')
            pil.thumbnail((160, 120))
            result.append({'filename': os.path.basename(path), 'thumb': _pil_to_b64(pil)})
        except Exception:
            pass
    _sample_cache = result
    return result


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    return jsonify({
        'load_status': load_status,
        'device': str(device),
        'device_name': (torch.cuda.get_device_name(0)
                        if device.type == 'cuda' else 'CPU'),
        'models': {k: v['desc'] for k, v in MODEL_META.items()},
        'vocab_loaded': vocab_q is not None,
    })


@app.route('/api/samples')
def api_samples():
    imgs = _get_sample_images(24)
    return jsonify(imgs)


@app.route('/api/refresh_samples')
def api_refresh_samples():
    global _sample_cache
    _sample_cache = []
    return api_samples()


@app.route('/api/image/<path:filename>')
def api_image(filename):
    path = VAL_IMAGE_DIR / filename
    if not path.exists():
        abort(404)
    return send_file(str(path), mimetype='image/jpeg')


@app.route('/api/infer', methods=['POST'])
def api_infer():
    data = request.get_json(force=True)

    # ── Parse request ──────────────────────────────────────────────────────
    image_b64  = data.get('image_b64', '')
    question   = data.get('question', '').strip()
    sel_models = data.get('models', ['A', 'B', 'C', 'D'])
    decode_mode= data.get('decode_mode', 'greedy')   # 'greedy' | 'beam' | 'both'
    beam_width = int(data.get('beam_width', 3))
    max_len    = 30

    if not image_b64 or not question:
        return jsonify({'error': 'image and question required'}), 400
    if vocab_q is None or vocab_a is None:
        return jsonify({'error': 'Models still loading, please wait...'}), 503

    # ── Decode image ──────────────────────────────────────────────────────
    try:
        img_bytes  = base64.b64decode(image_b64.split(',')[-1])
        orig_pil   = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        orig_224   = orig_pil.resize((224, 224), Image.BILINEAR)
        img_tensor = IMG_TRANSFORM(orig_pil)
    except Exception as e:
        return jsonify({'error': f'Image decode failed: {e}'}), 400

    # ── Tokenize question ─────────────────────────────────────────────────
    try:
        q_ids   = vocab_q.numericalize(question)
        q_tensor= torch.tensor(q_ids, dtype=torch.long)
    except Exception as e:
        return jsonify({'error': f'Question tokenize failed: {e}'}), 400

    # ── Run inference per model ───────────────────────────────────────────
    results = {}
    for mt in sel_models:
        if mt not in models:
            results[mt] = {'error': load_status.get(mt, 'not loaded'), 'skipped': True}
            continue

        model = models[mt]
        t0    = time.time()
        try:
            has_attn = mt in ('C', 'D')
            # Always run greedy for detailed output
            if has_attn:
                tokens, probs, heatmaps, avg_heatmap, attn_weights, top5 = \
                    _decode_cd(model, img_tensor, q_tensor, orig_224, max_len)
            else:
                tokens, probs, top5 = _decode_ab(model, img_tensor, q_tensor, max_len)
                heatmaps, avg_heatmap, attn_weights = [], None, []

            greedy_answer = ' '.join(tokens) if tokens else '<empty>'

            # Optionally also run beam
            beam_answer = None
            if decode_mode in ('beam', 'both'):
                beam_answer = _decode_beam(mt, model, img_tensor, q_tensor, beam_width, max_len)

            # Compute attention entropy per step (measure of focus)
            attn_entropy = []
            for w in attn_weights:
                w_np = np.array(w) + 1e-8
                w_np = w_np / w_np.sum()
                entropy = float(-np.sum(w_np * np.log(w_np)))
                attn_entropy.append(round(entropy, 3))

            results[mt] = {
                'greedy_answer': greedy_answer,
                'beam_answer':   beam_answer,
                'tokens':        tokens,
                'token_probs':   probs,
                'avg_confidence': round(float(np.mean(probs)), 4) if probs else 0.0,
                'top5_last':     top5,
                'has_attention': has_attn,
                'heatmaps':      heatmaps,       # list of base64 per token
                'avg_heatmap':   avg_heatmap,    # base64
                'attn_weights':  attn_weights,   # list of 49-element lists
                'attn_entropy':  attn_entropy,   # per token
                'meta':          MODEL_META[mt],
                'inference_ms':  round((time.time() - t0) * 1000),
            }
        except Exception as e:
            import traceback
            results[mt] = {'error': str(e), 'trace': traceback.format_exc(), 'skipped': True}

    # Return the 224×224 display image
    display_b64 = _pil_to_b64(orig_224)

    return jsonify({'results': results, 'display_image': display_b64})


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"[VQA Demo] Starting on http://localhost:5000  |  device={device}")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
