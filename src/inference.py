""" 
using autoregressive 
"""



import torch 
import torch.nn.functional as F 
import os, sys, json 
sys.path.append(os.path.dirname(__file__))


from models.vqa_models import VQAModelA, VQAModelB, VQAModelC, VQAModelD, VQAModelE
from vocab import Vocabulary


def _fuse(model, img_feat, q_feat):
    """Route fusion args correctly per model type.
    GatedFusion (A/B/C/D): forward(img, q)
    MUTANFusion  (E/F):     forward(q, img)  ← different signature
    """
    if getattr(model, 'model_type', '') in ('E', 'F'):
        return model.fusion(q_feat, img_feat)
    return model.fusion(img_feat, q_feat)


def strip_compiled_prefix(state_dict):
    """Strip '_orig_mod.' prefix from keys saved by torch.compile.
    
    When a model is wrapped with torch.compile(), state_dict keys get a
    '_orig_mod.' prefix. This helper makes such checkpoints loadable
    into non-compiled models (inference, evaluate, compare, visualize).
    """
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.', '') if k.startswith('_orig_mod.') else k
        cleaned[new_key] = v
    return cleaned


def _get_ngrams(token_ids, n):
    """Extract all n-grams from a list of token IDs."""
    return {tuple(token_ids[i:i+n]) for i in range(len(token_ids) - n + 1)}


def _block_repeated_ngrams(log_probs, token_ids, no_repeat_ngram_size):
    """Set log-prob to -inf for any token that would create a repeated n-gram.

    Args:
        log_probs: (vocab,) log probability tensor — MODIFIED in-place
        token_ids: list of int — tokens generated so far
        no_repeat_ngram_size: size of n-grams to block (e.g. 3 = no repeated trigrams)
    """
    if no_repeat_ngram_size <= 0 or len(token_ids) < no_repeat_ngram_size - 1:
        return
    # The (n-1)-gram ending at the current position
    prefix = tuple(token_ids[-(no_repeat_ngram_size - 1):])
    existing = _get_ngrams(token_ids, no_repeat_ngram_size)
    for ng in existing:
        if ng[:-1] == prefix:
            # Block the token that would complete the repeated n-gram
            log_probs[ng[-1]] = float('-inf')


def _has_coverage_keys(state_dict):
    """Detect if a state_dict was trained with coverage enabled."""
    return any('W_cov' in k for k in state_dict.keys())


def get_model(model_type, vocab_size, glove_dim=300, use_coverage=False, answer_vocab_size=None):
    """Create a model with architecture matching checkpoints trained with --glove.
    
    When glove_dim > 0, dummy embeddings of that dimension are created so the
    model has the correct layers (embedding + embed_proj). The actual weights
    are overwritten when loading the checkpoint via load_state_dict().
    
    Args:
        use_coverage: If True, create C/D models with coverage layers (W_cov).
                      Set this when loading checkpoints trained with --coverage.
    """
    if answer_vocab_size is None:
        answer_vocab_size = vocab_size
    # Create dummy GloVe embeddings to match training architecture
    if glove_dim > 0:
        pretrained_q_emb = torch.zeros(vocab_size, glove_dim)
        pretrained_a_emb = torch.zeros(answer_vocab_size, glove_dim)
    else:
        pretrained_q_emb = None
        pretrained_a_emb = None

    glove_kw = dict(pretrained_q_emb=pretrained_q_emb, pretrained_a_emb=pretrained_a_emb)
    if model_type == 'A':
        return VQAModelA(vocab_size=vocab_size, answer_vocab_size=answer_vocab_size, **glove_kw)
    elif model_type == 'B':
        return VQAModelB(vocab_size=vocab_size, answer_vocab_size=answer_vocab_size, **glove_kw)
    elif model_type == 'C':
        return VQAModelC(vocab_size=vocab_size, answer_vocab_size=answer_vocab_size,
                         use_coverage=use_coverage, **glove_kw)
    elif model_type == 'D':
        return VQAModelD(vocab_size=vocab_size, answer_vocab_size=answer_vocab_size,
                         use_coverage=use_coverage, **glove_kw)
    elif model_type == 'E':
        return VQAModelE(vocab_size=vocab_size, answer_vocab_size=answer_vocab_size,
                         use_coverage=use_coverage,
                         pretrained_q_emb=pretrained_q_emb,
                         pretrained_a_emb=pretrained_a_emb,
                         use_mutan=True, use_pgn=True,
                         use_layer_norm=True, use_dropconnect=True,
                         use_q_highway=True, use_char_cnn=True, use_dcan=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from A, B, C, D, E.")


def load_model_from_checkpoint(model_type, checkpoint, vocab_size,
                               device='cpu', glove_dim=300, vocab=None,
                               answer_vocab_size=None):
    """Load a model from checkpoint, auto-detecting coverage from state_dict."""
    state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    state_dict = strip_compiled_prefix(state_dict)
    use_coverage = _has_coverage_keys(state_dict)
    # Auto-detect vocab sizes from checkpoint to avoid mismatches when the
    # saved vocab files differ slightly from the checkpoint's actual sizes.
    if 'q_encoder.embedding.weight' in state_dict:
        vocab_size = state_dict['q_encoder.embedding.weight'].shape[0]
    if answer_vocab_size is None:
        answer_vocab_size = vocab_size
    if 'decoder.embedding.weight' in state_dict:
        answer_vocab_size = state_dict['decoder.embedding.weight'].shape[0]
    model = get_model(model_type, vocab_size,
                      answer_vocab_size=answer_vocab_size,
                      glove_dim=glove_dim, use_coverage=use_coverage)
    model.load_state_dict(state_dict, strict=False)
    # char_table is a non-parameter buffer that must be rebuilt from vocab after load
    if vocab is not None and hasattr(model, 'q_encoder') and \
            hasattr(model.q_encoder, 'char_cnn') and model.q_encoder.char_cnn is not None:
        model.q_encoder.char_cnn.build_char_table(vocab)
    model.to(device)
    model.eval()
    return model


def greedy_decode(model, image_tensor, question_tensor, vocab,
                  max_len=50, device='cpu'):
    """
    image_tensor (3, 224, 224)
    question_tensor (max_q_len)
    return: string answer
    """
    model.eval()
    with torch.no_grad():
        # add batch dim 
        img = image_tensor.unsqueeze(0).to(device) # (1, 3, 224, 224)
        question = question_tensor.unsqueeze(0).to(device) # (1, max_q_len)

        
        # encode 
        img_feat = model.i_encoder(img) # (1, 1024) 1024 is hidden size
        img_feat = F.normalize(img_feat, p=2, dim=1)
        question_feat, _ = model.q_encoder(question) # (1, 1024)

        fusion = _fuse(model, img_feat, question_feat) # (1, 1024)

        # prepare h_0 and c_0 for decoder 
        h_0 = fusion.unsqueeze(0).repeat(model.num_layers, 1, 1) # (num_layers, 1, 1024)
        c_0 = torch.zeros_like(h_0)
        
        hidden = (h_0, c_0)
        
        
        # autogression decode 
        start_idx = vocab.word2idx['<start>']
        end_idx = vocab.word2idx['<end>']

        # input shape (1, 1) first is <start>
        token = torch.tensor([[start_idx]], dtype=torch.long).to(device)

        result = []

        for _ in range(max_len):
            embed = model.decoder.embedding(token) # (1, 1, embed_size or glove_dim)
            if model.decoder.embed_proj is not None:
                embed = model.decoder.embed_proj(embed)
            output, hidden = model.decoder.lstm(embed, hidden) # output (1, 1, hidden_size)
            logit = model.decoder.fc(model.decoder.out_proj(output.squeeze(1))) # (1, vocab_size)
            pred = logit.argmax(dim=-1).item() # greedy 
            
            if pred == end_idx:
                break 
            
            result.append(pred)
            token = torch.tensor([[pred]], dtype=torch.long).to(device)

            
        words = [vocab.idx2word.get(i, '<unk>') for i in result]

        return ' '.join(words)


def greedy_decode_with_attention(model, image_tensor, question_tensor, vocab,
                                 max_len=50, device='cpu'):
    """
    For Model C and D (with Bahdanau attention).
    image_tensor : (3, 224, 224)
    question_tensor: (max_q_len)
    return: string answer
    """
    model.eval()
    with torch.no_grad():
        img      = image_tensor.unsqueeze(0).to(device)       # (1, 3, 224, 224)
        question = question_tensor.unsqueeze(0).to(device)    # (1, max_q_len)

        # encode
        img_features  = model.i_encoder(img)                  # (1, 49, 1024) -- keeps spatial
        img_features  = F.normalize(img_features, p=2, dim=-1)
        question_feat, q_hidden = model.q_encoder(question)   # (1, 1024), (1, qlen, 1024)

        # build image representation as mean of spatial regions
        img_mean = img_features.mean(dim=1)                   # (1, 1024)

        fusion = _fuse(model, img_mean, question_feat)        # (1, 1024)

        # initialize decoder hidden state
        h_0 = fusion.unsqueeze(0).repeat(model.num_layers, 1, 1)  # (num_layers, 1, 1024)
        c_0 = torch.zeros_like(h_0)
        hidden = (h_0, c_0)

        start_idx = vocab.word2idx['<start>']
        end_idx   = vocab.word2idx['<end>']

        token  = torch.tensor([[start_idx]], dtype=torch.long).to(device)  # (1, 1)
        result = []
        coverage = None  # coverage tracking (used if model has coverage enabled)

        for _ in range(max_len):
            # decode_step returns (logit, new_hidden, alpha, coverage)
            logit, hidden, alpha, coverage = model.decoder.decode_step(token, hidden, img_features, q_hidden, coverage)
            pred = logit.argmax(dim=-1).item()

            if pred == end_idx:
                break

            result.append(pred)
            token = torch.tensor([[pred]], dtype=torch.long).to(device)

        words = [vocab.idx2word.get(i, '<unk>') for i in result]
        return ' '.join(words)


def batch_greedy_decode(model, img_tensors, q_tensors, vocab,
                        max_len=50, device='cpu'):
    """
    Batch greedy decode for models A/B (no attention).

    img_tensors : (B, 3, 224, 224)
    q_tensors   : (B, max_q_len)
    returns     : list of B answer strings
    """
    model.eval()
    with torch.no_grad():
        B    = img_tensors.size(0)
        imgs = img_tensors.to(device)
        qs   = q_tensors.to(device)

        img_feat = model.i_encoder(imgs)               # (B, hidden)
        img_feat = F.normalize(img_feat, p=2, dim=1)
        q_feat, _ = model.q_encoder(qs)                  # (B, hidden)
        fusion   = _fuse(model, img_feat, q_feat)

        h = fusion.unsqueeze(0).repeat(model.num_layers, 1, 1)  # (layers, B, hidden)
        c = torch.zeros_like(h)
        hidden = (h, c)

        start_idx = vocab.word2idx['<start>']
        end_idx   = vocab.word2idx['<end>']

        token    = torch.full((B, 1), start_idx, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        results  = [[] for _ in range(B)]

        for _ in range(max_len):
            embed  = model.decoder.embedding(token)             # (B, 1, embed)
            if model.decoder.embed_proj is not None:
                embed = model.decoder.embed_proj(embed)
            output, hidden = model.decoder.lstm(embed, hidden)  # (B, 1, hidden)
            logit  = model.decoder.fc(model.decoder.out_proj(output.squeeze(1)))  # (B, vocab)
            preds  = logit.argmax(dim=-1)                       # (B,)

            for i in range(B):
                if not finished[i]:
                    p = preds[i].item()
                    if p == end_idx:
                        finished[i] = True
                    else:
                        results[i].append(p)

            if finished.all():
                break

            token = preds.unsqueeze(1)  # (B, 1)

        return [' '.join(vocab.idx2word.get(i, '<unk>') for i in r) for r in results]


def batch_greedy_decode_with_attention(model, img_tensors, q_tensors, vocab,
                                       max_len=50, device='cpu'):
    """
    Batch greedy decode for models C/D (Bahdanau attention).

    img_tensors : (B, 3, 224, 224)
    q_tensors   : (B, max_q_len)
    returns     : list of B answer strings
    """
    model.eval()
    with torch.no_grad():
        B    = img_tensors.size(0)
        imgs = img_tensors.to(device)
        qs   = q_tensors.to(device)

        img_features = model.i_encoder(imgs)                    # (B, 49, hidden)
        img_features = F.normalize(img_features, p=2, dim=-1)
        q_feat, q_hidden = model.q_encoder(qs)                   # (B, hidden), (B, qlen, hidden)
        img_mean     = img_features.mean(dim=1)                 # (B, hidden)
        fusion       = _fuse(model, img_mean, q_feat)

        h = fusion.unsqueeze(0).repeat(model.num_layers, 1, 1)
        c = torch.zeros_like(h)
        hidden = (h, c)

        start_idx = vocab.word2idx['<start>']
        end_idx   = vocab.word2idx['<end>']

        token    = torch.full((B, 1), start_idx, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        results  = [[] for _ in range(B)]
        coverage = None  # coverage tracking

        for _ in range(max_len):
            logit, hidden, _, coverage = model.decoder.decode_step(
                token, hidden, img_features, q_hidden, coverage,
                q_token_ids=qs)
            preds = logit.argmax(dim=-1)  # (B,)

            for i in range(B):
                if not finished[i]:
                    p = preds[i].item()
                    if p == end_idx:
                        finished[i] = True
                    else:
                        results[i].append(p)

            if finished.all():
                break

            token = preds.unsqueeze(1)  # (B, 1)

        return [' '.join(vocab.idx2word.get(i, '<unk>') for i in r) for r in results]


# ── Beam Search ────────────────────────────────────────────────────────────────
#
# Greedy decode always picks the single highest-prob token at each step.
# Beam search keeps the top-k (beam_width) candidate sequences at every step, then
# returns the one with the highest length-normalised log-probability.
#
# Why it helps:
#   Greedy: "yes" → score = log P("yes")
#   Beam-3: keeps {"yes", "no", "there is"} → compares P(full sequence) of all 3
#
# length normalisation: score / len(sequence)  – prevents short answers winning by default
# ──────────────────────────────────────────────────────────────────────────────

def beam_search_decode(model, img_tensor, q_tensor, vocab,
                       beam_width=5, max_len=50, device='cpu',
                       no_repeat_ngram_size=3, min_len=5, length_penalty=0.7):
    """
    Single-sample beam search for models A/B (no attention).
    Supports n-gram blocking to prevent repetitive output.

    img_tensor           : (3, 224, 224)
    q_tensor             : (max_q_len,)
    no_repeat_ngram_size : block repeated n-grams of this size (0=disabled)
    returns              : best answer string
    """
    model.eval()
    with torch.no_grad():
        img = img_tensor.unsqueeze(0).to(device)
        q   = q_tensor.unsqueeze(0).to(device)

        img_feat = F.normalize(model.i_encoder(img), p=2, dim=1)  # (1, hidden)
        q_feat, _ = model.q_encoder(q)                              # (1, hidden)
        fusion   = _fuse(model, img_feat, q_feat)

        h0 = fusion.unsqueeze(0).repeat(model.num_layers, 1, 1)   # (L, 1, hidden)
        c0 = torch.zeros_like(h0)

        start_idx = vocab.word2idx['<start>']
        end_idx   = vocab.word2idx['<end>']

        # Each beam: (cumulative_log_score, token_ids, h, c)
        beams     = [(0.0, [start_idx], h0, c0)]
        completed = []

        for step in range(max_len):
            if not beams:
                break
            candidates = []
            for log_score, tokens, bh, bc in beams:
                if tokens[-1] == end_idx:
                    completed.append((log_score, tokens))
                    continue
                tok  = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
                emb  = model.decoder.embedding(tok)                    # (1, 1, embed)
                if model.decoder.embed_proj is not None:
                    emb = model.decoder.embed_proj(emb)
                out, (nh, nc) = model.decoder.lstm(emb, (bh, bc))     # (1, 1, hidden)
                lp   = F.log_softmax(model.decoder.fc(model.decoder.out_proj(out.squeeze(1)))[0], dim=-1)  # (vocab,)
                # Min-length: block <end> for first min_len steps
                if step < min_len:
                    lp[end_idx] = float('-inf')
                # N-gram blocking: prevent repeated trigrams (or other n-grams)
                _block_repeated_ngrams(lp, tokens, no_repeat_ngram_size)
                topk_vals, topk_ids = lp.topk(beam_width)
                for v, idx in zip(topk_vals.tolist(), topk_ids.tolist()):
                    candidates.append((log_score + v, tokens + [idx], nh, nc))

            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_width]

        for log_score, tokens, _, _ in beams:
            completed.append((log_score, tokens))

        if not completed:
            return ''

        # Length penalty: score / len^alpha — stronger than /len, rewards fluent longer outputs
        completed.sort(key=lambda x: x[0] / (max(len(x[1]) - 1, 1) ** length_penalty), reverse=True)
        best  = completed[0][1][1:]   # strip <start>
        words = [vocab.idx2word.get(t, '<unk>') for t in best if t != end_idx]
        return ' '.join(words)


def beam_search_decode_with_attention(model, img_tensor, q_tensor, vocab,
                                      beam_width=5, max_len=50, device='cpu',
                                      no_repeat_ngram_size=3, min_len=5, length_penalty=0.7):
    """
    Single-sample beam search for models C/D (Bahdanau attention).
    Supports n-gram blocking to prevent repetitive output.

    img_tensor           : (3, 224, 224)
    q_tensor             : (max_q_len,)
    no_repeat_ngram_size : block repeated n-grams of this size (0=disabled)
    returns              : best answer string
    """
    model.eval()
    with torch.no_grad():
        img = img_tensor.unsqueeze(0).to(device)
        q   = q_tensor.unsqueeze(0).to(device)

        img_features = F.normalize(model.i_encoder(img), p=2, dim=-1)  # (1, 49, hidden)
        q_feat, q_hidden = model.q_encoder(q)                           # (1, hidden), (1, qlen, hidden)
        img_mean     = img_features.mean(dim=1)                         # (1, hidden)
        fusion       = _fuse(model, img_mean, q_feat)

        h0 = fusion.unsqueeze(0).repeat(model.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)

        start_idx = vocab.word2idx['<start>']
        end_idx   = vocab.word2idx['<end>']

        beams     = [(0.0, [start_idx], h0, c0, None)]  # last element = coverage
        completed = []

        for step in range(max_len):
            if not beams:
                break
            candidates = []
            for log_score, tokens, bh, bc, b_cov in beams:
                if tokens[-1] == end_idx:
                    completed.append((log_score, tokens))
                    continue
                tok            = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
                logit, (nh, nc), _, new_cov = model.decoder.decode_step(tok, (bh, bc), img_features, q_hidden, b_cov)
                lp             = F.log_softmax(logit[0], dim=-1)  # (vocab,)
                # Min-length: block <end> for first min_len steps
                if step < min_len:
                    lp[end_idx] = float('-inf')
                # N-gram blocking: prevent repeated trigrams (or other n-grams)
                _block_repeated_ngrams(lp, tokens, no_repeat_ngram_size)
                topk_vals, topk_ids = lp.topk(beam_width)
                for v, idx in zip(topk_vals.tolist(), topk_ids.tolist()):
                    candidates.append((log_score + v, tokens + [idx], nh, nc, new_cov))

            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_width]

        for log_score, tokens, _, _, _ in beams:
            completed.append((log_score, tokens))

        if not completed:
            return ''

        # Length penalty: score / len^alpha
        completed.sort(key=lambda x: x[0] / (max(len(x[1]) - 1, 1) ** length_penalty), reverse=True)
        best  = completed[0][1][1:]   # strip <start>
        words = [vocab.idx2word.get(t, '<unk>') for t in best if t != end_idx]
        return ' '.join(words)


def batch_beam_search_decode(model, img_tensors, q_tensors, vocab,
                             beam_width=5, max_len=50, device='cpu',
                             no_repeat_ngram_size=3, min_len=5, length_penalty=0.7):
    """Batch wrapper for beam_search_decode (models A/B)."""
    return [
        beam_search_decode(
            model, img_tensors[i], q_tensors[i], vocab,
            beam_width=beam_width, max_len=max_len, device=device,
            no_repeat_ngram_size=no_repeat_ngram_size,
            min_len=min_len, length_penalty=length_penalty,
        )
        for i in range(img_tensors.size(0))
    ]


def batch_beam_search_decode_with_attention(model, img_tensors, q_tensors, vocab,
                                            beam_width=5, max_len=50, device='cpu',
                                            no_repeat_ngram_size=3, min_len=5, length_penalty=0.7):
    """Batch wrapper for beam_search_decode_with_attention (models C/D)."""
    return [
        beam_search_decode_with_attention(
            model, img_tensors[i], q_tensors[i], vocab,
            beam_width=beam_width, max_len=max_len, device=device,
            no_repeat_ngram_size=no_repeat_ngram_size,
            min_len=min_len, length_penalty=length_penalty,
        )
        for i in range(img_tensors.size(0))
    ]


# ── True Batched Beam Search ────────────────────────────────────────────────────
#
# Processes B samples × K beams in ONE GPU call per decode step.
# Old approach: B × K × max_steps individual calls (e.g. 64×3×50 = 9600 calls).
# New approach: max_steps calls, each on a batch of B×K samples (e.g. 50 calls).
#
# Speed-up comes from:
#   1. Fewer CUDA kernel launches (9600 → 50)
#   2. Better GPU SM utilization (192 samples vs 1 per kernel)
#   3. Memory coalescing across the full B×K batch
# ────────────────────────────────────────────────────────────────────────────────

def true_batched_beam_search_with_attention(
        model, img_tensors, q_tensors, vocab,
        beam_width=3, max_len=50, device='cpu',
        no_repeat_ngram_size=3, min_len=5, length_penalty=0.7):
    """
    True batched beam search for attention models (C/D/E/F).

    All B samples and K beams are decoded in a single decoder forward call per
    time step.  Equivalent output to looping beam_search_decode_with_attention
    B times but substantially faster on GPU.

    img_tensors : (B, 3, H, W)
    q_tensors   : (B, max_q_len)
    returns     : list[str] of length B
    """
    B  = img_tensors.size(0)
    K  = beam_width
    start_idx = vocab.word2idx['<start>']
    end_idx   = vocab.word2idx['<end>']

    imgs = img_tensors.to(device)
    qs   = q_tensors.to(device)

    model.eval()
    with torch.no_grad():
        # ── Encode (single forward pass for all B samples) ───────────────────
        img_feats      = F.normalize(model.i_encoder(imgs), p=2, dim=-1)  # (B, S, H)
        q_feat, q_hid  = model.q_encoder(qs)                              # (B, H), (B, Q, H)
        img_mean       = img_feats.mean(1)                                 # (B, H)
        fusion         = _fuse(model, img_mean, q_feat)                    # (B, H)

        L = model.num_layers
        h = fusion.unsqueeze(0).repeat(L, 1, 1)   # (L, B, H)
        c = torch.zeros_like(h)                    # (L, B, H)

        # Expand encodings to B*K (repeat_interleave keeps b0k0,b0k1..b1k0,b1k1...)
        h           = h.repeat_interleave(K, dim=1)            # (L, B*K, H)
        c           = c.repeat_interleave(K, dim=1)            # (L, B*K, H)
        img_feats_k = img_feats.repeat_interleave(K, dim=0)    # (B*K, S, H)
        q_hid_k     = q_hid.repeat_interleave(K, dim=0)        # (B*K, Q, H)
        q_tok_k     = qs.repeat_interleave(K, dim=0)           # (B*K, q_len)

        # ── Beam state ───────────────────────────────────────────────────────
        # token_seqs[bk, t] = t-th token of beam bk; pre-filled with end_idx as padding
        token_seqs = torch.full((B * K, max_len + 1), end_idx,
                                dtype=torch.long, device=device)
        token_seqs[:, 0] = start_idx

        # scores[b, k] = cumulative log-prob; only beam-0 is active at step 0
        scores = torch.full((B, K), float('-inf'), device=device)
        scores[:, 0] = 0.0

        coverage  = None
        # finished_bk[bk] = True once beam bk has generated <end>
        finished_bk = torch.zeros(B * K, dtype=torch.bool, device=device)
        completed   = [[] for _ in range(B)]   # (score, token_list) per batch item

        # Pre-compute batch offset (avoids re-allocating every step)
        batch_off = torch.arange(B, device=device).unsqueeze(1) * K  # (B, 1)
        V = -1  # will be set on first step

        for step in range(max_len):
            # ── Single GPU decode step for all B*K beams ─────────────────────
            cur = token_seqs[:, step].unsqueeze(1)   # (B*K, 1)
            logit, (h, c), _, coverage = model.decoder.decode_step(
                cur, (h, c), img_feats_k, q_hid_k, coverage,
                q_token_ids=q_tok_k
            )   # logit: (B*K, V)

            if V < 0:
                V = logit.size(-1)
            lp = F.log_softmax(logit, dim=-1)        # (B*K, V)

            # Finished beams: set all log-probs to -inf FIRST
            # (also makes per-beam n-gram blocking unnecessary for finished beams)
            if finished_bk.any():
                lp[finished_bk] = float('-inf')

            # Min-length: block <end> for first min_len steps
            if step < min_len:
                lp[:, end_idx] = float('-inf')

            # N-gram blocking — only for active (non-finished) beams
            if no_repeat_ngram_size > 0:
                tok_cpu    = token_seqs[:, :step + 1].cpu().tolist()
                active_bks = (~finished_bk).nonzero(as_tuple=False).view(-1).tolist()
                for bk in active_bks:
                    _block_repeated_ngrams(lp[bk], tok_cpu[bk], no_repeat_ngram_size)

            # candidate scores: scores[b,k] + lp[bk, v]  → (B, K*V)
            cand = (scores.view(B * K, 1) + lp).reshape(B, K * V)

            # Top-K per batch item
            topk_scores, topk_ids = cand.topk(K, dim=-1)   # (B, K)
            topk_beam  = topk_ids // V                       # (B, K) source beam
            topk_token = topk_ids %  V                       # (B, K) next token

            # ── Reorder hidden states to match selected beams ────────────────
            beam_idx = (batch_off + topk_beam).reshape(-1)   # (B*K,)

            h = h[:, beam_idx, :]
            c = c[:, beam_idx, :]
            if coverage is not None:
                coverage = coverage[beam_idx]

            # ── Update token sequences: reindex rows, write new token in-place ─
            # Replaces the expensive 3-part torch.cat from the naive implementation.
            # token_seqs was pre-filled with end_idx, so positions > step+1 stay valid.
            token_seqs = token_seqs[beam_idx]                # reindex rows (B*K, max_len+1)
            token_seqs[:, step + 1] = topk_token.reshape(B * K)  # write new token

            scores = topk_scores                             # (B, K)

            # ── Track completed beams ─────────────────────────────────────────
            new_end   = topk_token.reshape(-1) == end_idx   # (B*K,)
            just_done = new_end & ~finished_bk
            if just_done.any():
                done_list = just_done.nonzero(as_tuple=False).view(-1).tolist()
                # Single CPU transfer for all completed sequences this step
                seqs_cpu  = token_seqs[just_done, :step + 2].cpu().tolist()
                for i, bk in enumerate(done_list):
                    b  = bk // K
                    k  = bk %  K
                    sc = topk_scores[b, k].item()
                    completed[b].append((sc, seqs_cpu[i]))

            finished_bk = finished_bk | new_end

            if finished_bk.all():
                break

        # ── Add unfinished beams to completed ────────────────────────────────
        unfinished = (~finished_bk).nonzero(as_tuple=False).view(-1).tolist()
        if unfinished:
            seqs_cpu = token_seqs[~finished_bk].cpu().tolist()
            for i, bk in enumerate(unfinished):
                b  = bk // K
                k  = bk %  K
                sc = scores[b, k].item()
                completed[b].append((sc, seqs_cpu[i]))

        # ── Select best sequence per batch item ──────────────────────────────
        results = []
        for b in range(B):
            cands = completed[b]
            if not cands:
                results.append('')
                continue
            best = max(cands,
                       key=lambda x: x[0] / (max(len(x[1]) - 1, 1) ** length_penalty))
            seq   = best[1][1:]   # strip <start>
            words = [vocab.idx2word.get(t, '<unk>') for t in seq if t != end_idx]
            results.append(' '.join(words))

        return results


def true_batched_beam_search(
        model, img_tensors, q_tensors, vocab,
        beam_width=3, max_len=50, device='cpu',
        no_repeat_ngram_size=3, min_len=5, length_penalty=0.7):
    """
    True batched beam search for non-attention models (A/B).

    img_tensors : (B, 3, H, W)
    q_tensors   : (B, max_q_len)
    returns     : list[str] of length B
    """
    B  = img_tensors.size(0)
    K  = beam_width
    start_idx = vocab.word2idx['<start>']
    end_idx   = vocab.word2idx['<end>']

    imgs = img_tensors.to(device)
    qs   = q_tensors.to(device)

    model.eval()
    with torch.no_grad():
        # Encode
        img_feat          = F.normalize(model.i_encoder(imgs), p=2, dim=1)  # (B, H)
        q_feat, _         = model.q_encoder(qs)                              # (B, H)
        fusion            = _fuse(model, img_feat, q_feat)                   # (B, H)

        L = model.num_layers
        h = fusion.unsqueeze(0).repeat(L, 1, 1)   # (L, B, H)
        c = torch.zeros_like(h)

        h = h.repeat_interleave(K, dim=1)          # (L, B*K, H)
        c = c.repeat_interleave(K, dim=1)

        token_seqs  = torch.full((B * K, max_len + 1), end_idx,
                                 dtype=torch.long, device=device)
        token_seqs[:, 0] = start_idx

        scores = torch.full((B, K), float('-inf'), device=device)
        scores[:, 0] = 0.0

        finished_bk = torch.zeros(B * K, dtype=torch.bool, device=device)
        completed   = [[] for _ in range(B)]
        batch_off   = torch.arange(B, device=device).unsqueeze(1) * K  # pre-compute
        V = -1

        for step in range(max_len):
            cur  = token_seqs[:, step].unsqueeze(1)   # (B*K, 1)
            emb  = model.decoder.embedding(cur)        # (B*K, 1, E)
            if model.decoder.embed_proj is not None:
                emb = model.decoder.embed_proj(emb)
            out, (h, c) = model.decoder.lstm(emb, (h, c))  # (B*K, 1, H)
            logit = model.decoder.fc(
                model.decoder.out_proj(out.squeeze(1)))     # (B*K, V)

            if V < 0:
                V = logit.size(-1)
            lp = F.log_softmax(logit, dim=-1)

            if finished_bk.any():
                lp[finished_bk] = float('-inf')
            if step < min_len:
                lp[:, end_idx] = float('-inf')

            if no_repeat_ngram_size > 0:
                tok_cpu    = token_seqs[:, :step + 1].cpu().tolist()
                active_bks = (~finished_bk).nonzero(as_tuple=False).view(-1).tolist()
                for bk in active_bks:
                    _block_repeated_ngrams(lp[bk], tok_cpu[bk], no_repeat_ngram_size)

            cand       = (scores.view(B * K, 1) + lp).reshape(B, K * V)
            topk_scores, topk_ids = cand.topk(K, dim=-1)
            topk_beam  = topk_ids // V
            topk_token = topk_ids %  V

            beam_idx = (batch_off + topk_beam).reshape(-1)
            h = h[:, beam_idx, :]
            c = c[:, beam_idx, :]

            token_seqs = token_seqs[beam_idx]
            token_seqs[:, step + 1] = topk_token.reshape(B * K)

            scores = topk_scores

            new_end   = topk_token.reshape(-1) == end_idx
            just_done = new_end & ~finished_bk
            if just_done.any():
                done_list = just_done.nonzero(as_tuple=False).view(-1).tolist()
                seqs_cpu  = token_seqs[just_done, :step + 2].cpu().tolist()
                for i, bk in enumerate(done_list):
                    b  = bk // K
                    k  = bk %  K
                    sc = topk_scores[b, k].item()
                    completed[b].append((sc, seqs_cpu[i]))

            finished_bk = finished_bk | new_end
            if finished_bk.all():
                break

        unfinished = (~finished_bk).nonzero(as_tuple=False).view(-1).tolist()
        if unfinished:
            seqs_cpu = token_seqs[~finished_bk].cpu().tolist()
            for i, bk in enumerate(unfinished):
                b  = bk // K
                k  = bk %  K
                sc = scores[b, k].item()
                completed[b].append((sc, seqs_cpu[i]))

        results = []
        for b in range(B):
            cands = completed[b]
            if not cands:
                results.append('')
                continue
            best  = max(cands,
                        key=lambda x: x[0] / (max(len(x[1]) - 1, 1) ** length_penalty))
            seq   = best[1][1:]
            words = [vocab.idx2word.get(t, '<unk>') for t in seq if t != end_idx]
            results.append(' '.join(words))

        return results


# ── G5 Beam Search for VQAModel (Model G) ──────────────────────────────────────
#
# Uses VQAModel.encode() + VQAModel.decode_step() — the clean unified API.
# Supports:
#   G2: label_tokens passed through decode_step → ThreeWayPGNHead visual copy
#   G5: min_decode_len enforces LONG-bin minimum length (default 8 per spec)
#       length_bin=2 (LONG) is forced at inference — decoder always sees LONG
# ────────────────────────────────────────────────────────────────────────────────

def beam_search_decode_g(
    model,
    feats: torch.Tensor,
    q_tensor: torch.Tensor,
    vocab,
    beam_width: int = 5,
    max_len: int = 50,
    device: str = 'cpu',
    no_repeat_ngram_size: int = 3,
    min_decode_len: int = 8,
    length_penalty: float = 0.7,
    img_mask=None,
    label_tokens=None,
) -> str:
    """
    Single-sample beam search for VQAModel (Model G).

    Uses the unified VQAModel.encode() → _fuse_and_init() → decode_step() API.
    G5: blocks <end> for first `min_decode_len` steps (default 8, LONG bin).
    G2: passes label_tokens through to ThreeWayPGNHead for visual copy distribution.

    Args:
        model          : VQAModel instance (Model G)
        feats          : (k, feat_dim) BUTD features for one sample
        q_tensor       : (q_len,) question token ids
        vocab          : answer Vocabulary
        beam_width     : K beam width
        max_len        : max decode steps
        device         : torch device string
        no_repeat_ngram_size : block repeated n-grams (0=disabled)
        min_decode_len : G5 — block <end> for this many steps (default 8)
        length_penalty : score / len^alpha normalization
        img_mask       : (k,) bool — valid regions; None = all valid
        label_tokens   : (k, max_t) int64 — G2 visual label token ids; None = skip

    Returns:
        best answer string
    """
    model.eval()
    with torch.no_grad():
        # Add batch dimension
        feats_b   = feats.unsqueeze(0).to(device)       # (1, k, feat_dim)
        q_b       = q_tensor.unsqueeze(0).to(device)    # (1, q_len)
        mask_b    = img_mask.unsqueeze(0).to(device) if img_mask is not None else None
        ltok_b    = label_tokens.unsqueeze(0).to(device) if label_tokens is not None else None
        # G5: always decode as LONG at inference
        len_bin_b = torch.full((1,), 2, dtype=torch.long, device=device)

        # Encode
        V, q_feat, Q_H = model.encode(feats_b, q_b, img_mask=mask_b)
        h_0, c_0 = model._fuse_and_init(V, q_feat, mask_b)

        start_idx = vocab.word2idx['<start>']
        end_idx   = vocab.word2idx['<end>']

        # Beam state: (cumulative_log_score, token_ids, h, c, coverage)
        beams     = [(0.0, [start_idx], h_0, c_0, None)]
        completed = []

        for step in range(max_len):
            if not beams:
                break
            candidates = []
            for log_score, tokens, bh, bc, b_cov in beams:
                if tokens[-1] == end_idx:
                    completed.append((log_score, tokens))
                    continue

                tok = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
                logit, h_new, c_new, img_alpha, cov_new = model.decode_step(
                    tok, bh, bc, V, Q_H,
                    coverage=b_cov,
                    img_mask=mask_b,
                    q_token_ids=q_b,
                    length_bin=len_bin_b,
                    label_tokens=ltok_b,
                )
                lp = F.log_softmax(logit[0], dim=-1)    # (V,)

                # G5 min length: block <end> for first min_decode_len steps
                if step < min_decode_len:
                    lp[end_idx] = float('-inf')

                _block_repeated_ngrams(lp, tokens, no_repeat_ngram_size)
                topk_vals, topk_ids = lp.topk(beam_width)
                for v, idx in zip(topk_vals.tolist(), topk_ids.tolist()):
                    candidates.append((log_score + v, tokens + [idx], h_new, c_new, cov_new))

            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_width]

        for log_score, tokens, _, _, _ in beams:
            completed.append((log_score, tokens))

        if not completed:
            return ''

        completed.sort(
            key=lambda x: x[0] / (max(len(x[1]) - 1, 1) ** length_penalty),
            reverse=True)
        best  = completed[0][1][1:]     # strip <start>
        words = [vocab.idx2word.get(t, '<unk>') for t in best if t != end_idx]
        return ' '.join(words)


def batch_beam_search_decode_g(
    model,
    feats_batch: torch.Tensor,
    q_batch: torch.Tensor,
    vocab,
    beam_width: int = 5,
    max_len: int = 50,
    device: str = 'cpu',
    no_repeat_ngram_size: int = 3,
    min_decode_len: int = 8,
    length_penalty: float = 0.7,
    img_mask_batch=None,
    label_tokens_batch=None,
) -> list:
    """
    Batch wrapper for beam_search_decode_g (Model G).

    feats_batch : (B, k, feat_dim)
    q_batch     : (B, q_len)
    Returns     : list[str] of length B
    """
    B = feats_batch.size(0)
    return [
        beam_search_decode_g(
            model,
            feats_batch[i],
            q_batch[i],
            vocab,
            beam_width=beam_width,
            max_len=max_len,
            device=device,
            no_repeat_ngram_size=no_repeat_ngram_size,
            min_decode_len=min_decode_len,
            length_penalty=length_penalty,
            img_mask=img_mask_batch[i] if img_mask_batch is not None else None,
            label_tokens=label_tokens_batch[i] if label_tokens_batch is not None else None,
        )
        for i in range(B)
    ]


if __name__ == "__main__":
    from PIL import Image
    from torchvision import transforms

    MODEL_TYPE    = 'A'   # change to 'B', 'C', 'D' to run other models
    DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    VOCAB_JOINT_PATH = "data/processed/vocab_joint.json"
    
    CHECKPOINT    = f"checkpoints/model_{MODEL_TYPE.lower()}_epoch10.pth"
    IMAGE_DIR     = "data/images/train2014"
    VQA_E_JSON    = "data/annotations/vqa_e/VQA-E_train_set.json"

    # Load vocab
    vocab = Vocabulary(); vocab.load(VOCAB_JOINT_PATH)
    

    # Load model
    model = load_model_from_checkpoint(
        MODEL_TYPE, CHECKPOINT, len(vocab), device=DEVICE
    )

    # Load 1 sample
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # VQA-E format: root is a list of dicts
    with open(VQA_E_JSON, 'r') as f:
        annotations = json.load(f)

    sample = annotations[0]
    q_text = sample['question']
    img_id = sample['img_id']

    img_path   = os.path.join(IMAGE_DIR, f"COCO_train2014_{img_id:012d}.jpg")
    img_tensor = transform(Image.open(img_path).convert("RGB"))
    q_tensor   = torch.tensor(vocab.numericalize(q_text), dtype=torch.long)

    # select decode function based on model type
    if MODEL_TYPE in ('A', 'B'):
        answer = greedy_decode(model, img_tensor, q_tensor, vocab, device=DEVICE)
    else:
        answer = greedy_decode_with_attention(model, img_tensor, q_tensor, vocab, device=DEVICE)

    print(f"Model    : {MODEL_TYPE}")
    print(f"Question : {q_text}")
    print(f"Predicted: {answer}")