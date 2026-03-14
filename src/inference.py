"""Autoregressive decoding utilities for all VQA models (A–H).

Provides greedy decode and beam-search decode for both global-vector models
(A, B) and spatial-attention models (C–H).  Batch variants run each item
in the batch through a shared encoder then decode in parallel, which is much
faster than calling the single-sample functions in a loop.

Model routing
-------------
* **A / B** — global image vector, no attention.  Use ``batch_greedy_decode``
  or ``batch_beam_search_decode``.  Question input: ``LongTensor (B, Q)``.
* **C / D** — spatial features, dual Bahdanau attention, GatedFusion init.
  Use ``batch_greedy_decode_with_attention``.  Question: ``LongTensor (B, Q)``.
* **E** — same as C/D but FiLM.  Detected via ``hasattr(model, 'init_h_proj')``.
  Question: ``LongTensor (B, Q)``.
* **F / G** — CLIP text encoder replaces BiLSTM.  Same ``_with_attention``
  decode functions; question input is ``LongTensor (B, 77)`` CLIP BPE ids.
  F uses Bahdanau; G uses MHA. Decoder hidden state is LSTM (h, c).
* **H** — CLIP encoders + Transformer decoder.  Same ``_with_attention``
  functions auto-dispatch via ``_is_transformer(model.decoder)``.  Hidden
  state is the KV cache (accumulated embedding buffer), not an LSTM state.

Checkpoint compatibility
------------------------
``load_model_from_checkpoint`` handles three checkpoint formats:

1. Wrapped: ``{'model_state_dict': ..., 'epoch': ...}`` (saved by ``train.py``).
2. Raw state dict (legacy format).
3. ``torch.compile``-prefixed keys (``_orig_mod.*``) stripped automatically.

Dummy GloVe embeddings
-----------------------
When a checkpoint was trained with ``--glove``, the embedding layers have shape
``(vocab_size, 300)`` instead of ``(vocab_size, embed_size)``.  ``get_model``
creates ``torch.zeros`` tensors of the correct GloVe dimension so the model
architecture matches the checkpoint before ``load_state_dict`` overwrites the
weights.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

sys.path.append(os.path.dirname(__file__))

from train import get_model as _get_model_base
from vocab import Vocabulary


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def strip_compiled_prefix(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Strip the ``_orig_mod.`` prefix added by ``torch.compile``.

    When a model is wrapped with ``torch.compile()``, its ``state_dict`` keys
    gain an ``_orig_mod.`` prefix.  This helper removes the prefix so such
    checkpoints can be loaded into non-compiled models at inference time.

    Args:
        state_dict: Raw state dict, possibly with ``_orig_mod.`` prefixes.

    Returns:
        New state dict with the prefix stripped from all keys.
    """
    return {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()
    }


def _has_coverage_keys(state_dict: Dict[str, Tensor]) -> bool:
    """Return True if *state_dict* contains coverage-mechanism weights.

    Coverage weights (``W_cov``) are only present in checkpoints trained with
    ``--coverage``.  Auto-detecting this avoids the caller having to track
    whether coverage was used during training.

    Args:
        state_dict: Model state dict (already stripped of compiled prefixes).

    Returns:
        ``True`` if any key contains ``'W_cov'``.
    """
    return any("W_cov" in k for k in state_dict.keys())


def get_model(
    model_type: str,
    vocab_q_size: int,
    vocab_a_size: int,
    glove_dim: int = 300,
    use_coverage: bool = False,
) -> nn.Module:
    """Create an unloaded model with architecture matching a GloVe-trained checkpoint.

    When *glove_dim* > 0, dummy zero embeddings of the specified dimension are
    created so that the model's embedding + ``embed_proj`` layers have the same
    shape as a checkpoint trained with ``--glove``.  Actual weight values are
    then overwritten by ``load_state_dict``.

    Args:
        model_type: One of ``'A'``, ``'B'``, ``'C'``, ``'D'``, ``'E'``.
        vocab_q_size: Question vocabulary size.
        vocab_a_size: Answer vocabulary size.
        glove_dim: GloVe embedding dimension used during training (default 300).
            Pass ``0`` to create a model without GloVe layers (standard path).
        use_coverage: Enable coverage mechanism for C/D/E checkpoints trained
            with ``--coverage``.

    Returns:
        Un-initialised ``nn.Module`` ready for ``load_state_dict``.
    """
    if glove_dim > 0:
        pretrained_q_emb: Optional[Tensor] = torch.zeros(vocab_q_size, glove_dim)
        pretrained_a_emb: Optional[Tensor] = torch.zeros(vocab_a_size, glove_dim)
    else:
        pretrained_q_emb = None
        pretrained_a_emb = None

    return _get_model_base(
        model_type=model_type,
        vocab_q_size=vocab_q_size,
        vocab_a_size=vocab_a_size,
        pretrained_q_emb=pretrained_q_emb,
        pretrained_a_emb=pretrained_a_emb,
        use_coverage=use_coverage,
    )


def load_model_from_checkpoint(
    model_type: str,
    checkpoint: str,
    vocab_q_size: int,
    vocab_a_size: int,
    device: Union[torch.device, str] = "cpu",
    glove_dim: int = 300,
) -> nn.Module:
    """Load a VQA model from a checkpoint file.

    Handles both the wrapped format saved by ``train.py``
    (``{'model_state_dict': ...}``) and legacy raw state dicts.  Coverage
    layers are auto-detected from the keys.

    Args:
        model_type: One of ``'A'``–``'E'``.
        checkpoint: Path to the ``.pth`` file.
        vocab_q_size: Question vocabulary size.
        vocab_a_size: Answer vocabulary size.
        device: Target device for the loaded model.
        glove_dim: GloVe dimension used at training time (default 300).

    Returns:
        Model in ``eval()`` mode on *device*.
    """
    raw = torch.load(checkpoint, map_location=device)

    # Handle wrapped vs raw checkpoint format.
    state_dict: Dict[str, Tensor] = raw.get("model_state_dict", raw)

    state_dict    = strip_compiled_prefix(state_dict)
    use_coverage  = _has_coverage_keys(state_dict)

    model = get_model(model_type, vocab_q_size, vocab_a_size,
                      glove_dim=glove_dim, use_coverage=use_coverage)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ── N-gram blocking helpers ───────────────────────────────────────────────────

def _get_ngrams(token_ids: List[int], n: int) -> Set[Tuple[int, ...]]:
    """Return all n-grams from *token_ids* as a set of tuples.

    Args:
        token_ids: Sequence of token index integers.
        n: n-gram size.

    Returns:
        Set of n-gram tuples.
    """
    return {tuple(token_ids[i: i + n]) for i in range(len(token_ids) - n + 1)}


def _block_repeated_ngrams(
    log_probs: Tensor,
    token_ids: List[int],
    no_repeat_ngram_size: int,
) -> None:
    """Set log-probabilities to ``-inf`` for tokens that would repeat an n-gram.

    Applied in beam search to prevent the decoder from repeating the same
    n-gram (e.g., trigram) within a single sequence.

    **In-place**: modifies *log_probs* directly.

    Args:
        log_probs: ``FloatTensor (vocab_size,)`` — log-probs for the next token.
        token_ids: Token indices generated so far (including ``<start>``).
        no_repeat_ngram_size: Size of the n-gram to block (``0`` = disabled).
    """
    if no_repeat_ngram_size <= 0 or len(token_ids) < no_repeat_ngram_size - 1:
        return

    prefix   = tuple(token_ids[-(no_repeat_ngram_size - 1):])
    existing = _get_ngrams(token_ids, no_repeat_ngram_size)

    for ng in existing:
        if ng[:-1] == prefix:
            log_probs[ng[-1]] = float("-inf")


# ── Encoder helpers ───────────────────────────────────────────────────────────

def _encode_global(
    model: nn.Module,
    imgs: Tensor,
    questions: Tensor,
    device: Union[torch.device, str],
) -> Tuple[Tensor, Tensor]:
    """Encode images and questions for global-vector models (A, B).

    Fuses the global image feature and question feature into an initial LSTM
    hidden state.  Cell state is zero-initialised.

    Args:
        model: ``VQAModelA`` or ``VQAModelB`` instance.
        imgs: ``FloatTensor (B, 3, 224, 224)``.
        questions: ``LongTensor (B, Q)``.
        device: Device to move tensors to.

    Returns:
        Tuple ``(h_0, c_0)``, each ``FloatTensor (num_layers, B, hidden_size)``.
    """
    imgs      = imgs.to(device)
    questions = questions.to(device)

    img_feat = F.normalize(model.i_encoder(imgs), p=2, dim=1)  # (B, H)
    q_feat, _ = model.q_encoder(questions)                      # (B, H)
    fusion    = model.fusion(img_feat, q_feat)                  # (B, H)

    h = fusion.unsqueeze(0).repeat(model.num_layers, 1, 1)      # (L, B, H)
    c = torch.zeros_like(h)                                     # (L, B, H)
    return h, c


def _is_transformer(decoder: nn.Module) -> bool:
    """Return True if *decoder* is a ``TransformerDecoder`` (Model H).

    Detected by the presence of ``final_norm`` (LayerNorm after the last
    Transformer layer), which is absent in all LSTM-based decoders.
    """
    return hasattr(decoder, "final_norm")


def _encode_spatial(
    model: nn.Module,
    imgs: Tensor,
    q_input: Tensor,
    device: Union[torch.device, str],
) -> Tuple[Tuple[Tensor, Tensor], Tensor, Tensor]:
    """Encode images and questions for spatial-attention models (C, D, E, F, G, H).

    Handles all three fusion variants:

    * **C / D**: GatedFusion, ``h_0 = repeat(fusion_global)``, ``c_0 = zeros``,
      spatial features un-modulated.
    * **E / F / G / H**: FiLM, ``h_0`` and ``c_0`` from
      ``init_h_proj``/``init_c_proj``; spatial features FiLM-modulated.
      Detected via ``hasattr(model, 'init_h_proj')``.

    For **Model H** (TransformerDecoder), the returned ``encoder_hidden`` tuple
    is computed but ignored by the decoder — state is maintained via KV cache.

    Args:
        model: Any spatial-attention VQA model (C–H).
        imgs: ``FloatTensor (B, 3, 224, 224)``.
        q_input: ``LongTensor (B, Q)`` vocab tokens (C/D/E) **or**
            ``LongTensor (B, 77)`` CLIP BPE ids (F/G/H).
        device: Device to move tensors to.

    Returns:
        Tuple of:
            encoder_hidden – ``(h_0, c_0)``, each ``(num_layers, B, H)``
                             (ignored by TransformerDecoder).
            spatial_feats  – ``FloatTensor (B, 49, H)`` — FiLM-modulated or raw.
            q_hidden       – ``FloatTensor (B, Q_or_77, H)`` question states.
    """
    imgs    = imgs.to(device)
    q_input = q_input.to(device)

    img_features         = F.normalize(model.i_encoder(imgs), p=2, dim=-1)
    # img_features: (B, 49, H)

    q_feat, q_hidden     = model.q_encoder(q_input)
    # q_feat: (B, H) | q_hidden: (B, Q_or_77, H)

    if hasattr(model, "q_norm"):
        q_feat = model.q_norm(q_feat)

    img_mean = img_features.mean(dim=1)  # (B, H)

    if hasattr(model, "init_h_proj"):
        # ── Models E / F / G / H: FiLM-based fusion ──────────────────────────
        fusion_global  = model.fusion(img_mean, q_feat)          # (B, H)
        h_0 = (model.init_h_proj(fusion_global)
               .unsqueeze(0).repeat(model.num_layers, 1, 1))     # (L, B, H)
        c_0 = (model.init_c_proj(fusion_global)
               .unsqueeze(0).repeat(model.num_layers, 1, 1))     # (L, B, H)
        # FiLM-modulate all 49 spatial patches
        spatial_feats = model.fusion(img_features, q_feat)       # (B, 49, H)
    else:
        # ── Models C / D: GatedFusion ─────────────────────────────────────────
        fusion_global  = model.fusion(img_mean, q_feat)          # (B, H)
        h_0 = fusion_global.unsqueeze(0).repeat(model.num_layers, 1, 1)  # (L, B, H)
        c_0 = torch.zeros_like(h_0)
        spatial_feats  = img_features                            # (B, 49, H) un-modulated

    return (h_0, c_0), spatial_feats, q_hidden


# ── Greedy decode ─────────────────────────────────────────────────────────────

def greedy_decode(
    model: nn.Module,
    image_tensor: Tensor,
    question_tensor: Tensor,
    vocab_a: Vocabulary,
    max_len: int = 100,
    device: Union[torch.device, str] = "cpu",
) -> str:
    """Single-sample greedy decode for global-vector models (A, B).

    Args:
        model: ``VQAModelA`` or ``VQAModelB``.
        image_tensor: ``FloatTensor (3, 224, 224)`` — single image.
        question_tensor: ``LongTensor (Q,)`` — single tokenised question.
        vocab_a: Answer vocabulary.
        max_len: Maximum number of tokens to generate.
        device: Device for computation.

    Returns:
        Decoded answer string (content tokens only, space-joined).
    """
    results = batch_greedy_decode(
        model,
        image_tensor.unsqueeze(0),
        question_tensor.unsqueeze(0),
        vocab_a,
        max_len=max_len,
        device=device,
    )
    return results[0]


def greedy_decode_with_attention(
    model: nn.Module,
    image_tensor: Tensor,
    question_tensor: Tensor,
    vocab_a: Vocabulary,
    max_len: int = 100,
    device: Union[torch.device, str] = "cpu",
) -> str:
    """Single-sample greedy decode for spatial-attention models (C, D, E).

    Args:
        model: ``VQAModelC``, ``VQAModelD``, or ``VQAModelE``.
        image_tensor: ``FloatTensor (3, 224, 224)`` — single image.
        question_tensor: ``LongTensor (Q,)`` — single tokenised question.
        vocab_a: Answer vocabulary.
        max_len: Maximum number of tokens to generate.
        device: Device for computation.

    Returns:
        Decoded answer string (content tokens only, space-joined).
    """
    results = batch_greedy_decode_with_attention(
        model,
        image_tensor.unsqueeze(0),
        question_tensor.unsqueeze(0),
        vocab_a,
        max_len=max_len,
        device=device,
    )
    return results[0]


def batch_greedy_decode(
    model: nn.Module,
    img_tensors: Tensor,
    q_tensors: Tensor,
    vocab_a: Vocabulary,
    max_len: int = 100,
    device: Union[torch.device, str] = "cpu",
) -> List[str]:
    """Batch greedy decode for global-vector models (A, B).

    Runs a single forward pass through the shared encoder, then generates
    tokens autoregressively using ``model.decoder.decode_step()``.

    Args:
        model: ``VQAModelA`` or ``VQAModelB``.
        img_tensors: ``FloatTensor (B, 3, 224, 224)``.
        q_tensors: ``LongTensor (B, Q)``.
        vocab_a: Answer vocabulary.
        max_len: Maximum tokens per sequence.
        device: Device for computation.

    Returns:
        List of B decoded answer strings.
    """
    model.eval()
    with torch.no_grad():
        B = img_tensors.size(0)

        h, c   = _encode_global(model, img_tensors, q_tensors, device)
        hidden: Tuple[Tensor, Tensor] = (h, c)

        token    = torch.full((B, 1), vocab_a.start_idx, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        results: List[List[int]] = [[] for _ in range(B)]

        for _ in range(max_len):
            logit, hidden = model.decoder.decode_step(token, hidden)
            # logit: (B, vocab_size)
            preds = logit.argmax(dim=-1)  # (B,)

            for i in range(B):
                if not finished[i]:
                    p = preds[i].item()
                    if p == vocab_a.end_idx:
                        finished[i] = True
                    else:
                        results[i].append(p)

            if finished.all():
                break

            token = preds.unsqueeze(1)  # (B, 1) — feed predicted token back

        return [
            " ".join(vocab_a.idx2word.get(i, "<unk>") for i in seq)
            for seq in results
        ]


def batch_greedy_decode_with_attention(
    model: nn.Module,
    img_tensors: Tensor,
    q_tensors: Tensor,
    vocab_a: Vocabulary,
    max_len: int = 100,
    device: Union[torch.device, str] = "cpu",
) -> List[str]:
    """Batch greedy decode for spatial-attention models (C–H).

    Routes automatically:
    - LSTM decoders (C/D/E/F/G): LSTM ``(h, c)`` hidden state, optional coverage.
    - Transformer decoder (H): KV-cache (accumulated embedding buffer).

    Args:
        model: Any spatial-attention VQA model (C–H).
        img_tensors: ``FloatTensor (B, 3, 224, 224)``.
        q_tensors: ``LongTensor (B, Q)`` vocab ids (C/D/E) or
            ``LongTensor (B, 77)`` CLIP ids (F/G/H).
        vocab_a: Answer vocabulary.
        max_len: Maximum tokens per sequence.
        device: Device for computation.

    Returns:
        List of B decoded answer strings.
    """
    model.eval()
    with torch.no_grad():
        B = img_tensors.size(0)

        encoder_hidden, spatial_feats, q_hidden = _encode_spatial(
            model, img_tensors, q_tensors, device
        )

        token    = torch.full((B, 1), vocab_a.start_idx, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        results: List[List[int]] = [[] for _ in range(B)]

        if _is_transformer(model.decoder):
            # ── Model H: Transformer KV-cache decode ──────────────────────────
            kv_cache: Optional[Tensor] = None
            for _ in range(max_len):
                logit, kv_cache, _alpha, _ = model.decoder.decode_step(
                    token, kv_cache, spatial_feats, q_hidden
                )
                preds = logit.argmax(dim=-1)
                for i in range(B):
                    if not finished[i]:
                        p = preds[i].item()
                        if p == vocab_a.end_idx:
                            finished[i] = True
                        else:
                            results[i].append(p)
                if finished.all():
                    break
                token = preds.unsqueeze(1)
        else:
            # ── LSTM decoders (C/D/E/F/G) ─────────────────────────────────────
            hidden   = encoder_hidden
            coverage: Optional[Tensor] = None
            for _ in range(max_len):
                logit, hidden, _alpha, coverage = model.decoder.decode_step(
                    token, hidden, spatial_feats, q_hidden, coverage
                )
                preds = logit.argmax(dim=-1)
                for i in range(B):
                    if not finished[i]:
                        p = preds[i].item()
                        if p == vocab_a.end_idx:
                            finished[i] = True
                        else:
                            results[i].append(p)
                if finished.all():
                    break
                token = preds.unsqueeze(1)

        return [
            " ".join(vocab_a.idx2word.get(i, "<unk>") for i in seq)
            for seq in results
        ]


# ── Beam search ───────────────────────────────────────────────────────────────
#
# Greedy picks the single highest-prob token at each step.
# Beam search keeps the top-K candidate sequences at every step and returns
# the one with the highest length-normalised cumulative log-probability.
#
#   length normalisation: score / max(len(seq) − 1, 1)
#     → prevents short answers from winning by default
#
# n-gram blocking: _block_repeated_ngrams sets log-prob to -inf for any token
# that would complete a repeated n-gram, penalising repetitive output.
# ─────────────────────────────────────────────────────────────────────────────

def beam_search_decode(
    model: nn.Module,
    img_tensor: Tensor,
    q_tensor: Tensor,
    vocab_a: Vocabulary,
    beam_width: int = 5,
    max_len: int = 100,
    device: Union[torch.device, str] = "cpu",
    no_repeat_ngram_size: int = 3,
) -> str:
    """Single-sample beam search for global-vector models (A, B).

    Args:
        model: ``VQAModelA`` or ``VQAModelB``.
        img_tensor: ``FloatTensor (3, 224, 224)`` — single image.
        q_tensor: ``LongTensor (Q,)`` — single tokenised question.
        vocab_a: Answer vocabulary.
        beam_width: Number of beams to maintain.
        max_len: Maximum tokens per beam.
        device: Device for computation.
        no_repeat_ngram_size: Block repeated n-grams of this size (``0`` = off).

    Returns:
        Best decoded answer string (length-normalised score).
    """
    model.eval()
    with torch.no_grad():
        h, c = _encode_global(model, img_tensor.unsqueeze(0), q_tensor.unsqueeze(0), device)

        # Each beam: (cumulative_log_score, token_ids, h, c)
        beams:     List[Tuple[float, List[int], Tensor, Tensor]] = [
            (0.0, [vocab_a.start_idx], h, c)
        ]
        completed: List[Tuple[float, List[int]]] = []

        for _ in range(max_len):
            if not beams:
                break
            candidates: List[Tuple[float, List[int], Tensor, Tensor]] = []

            for log_score, tokens, bh, bc in beams:
                if tokens[-1] == vocab_a.end_idx:
                    completed.append((log_score, tokens))
                    continue

                tok            = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
                logit, (nh, nc) = model.decoder.decode_step(tok, (bh, bc))
                # logit: (1, vocab_size)
                lp             = F.log_softmax(logit[0], dim=-1)  # (vocab_size,)
                _block_repeated_ngrams(lp, tokens, no_repeat_ngram_size)

                topk_vals, topk_ids = lp.topk(beam_width)
                for v, idx in zip(topk_vals.tolist(), topk_ids.tolist()):
                    candidates.append((log_score + v, tokens + [idx], nh, nc))

            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_width]

        # Flush remaining beams into completed
        for log_score, tokens, _, _ in beams:
            completed.append((log_score, tokens))

        if not completed:
            return ""

        # Length-normalised ranking: prevents short sequences winning by default.
        completed.sort(
            key=lambda x: x[0] / max(len(x[1]) - 1, 1), reverse=True
        )
        best  = completed[0][1][1:]  # strip <start>
        words = [vocab_a.idx2word.get(t, "<unk>") for t in best if t != vocab_a.end_idx]
        return " ".join(words)


def beam_search_decode_with_attention(
    model: nn.Module,
    img_tensor: Tensor,
    q_tensor: Tensor,
    vocab_a: Vocabulary,
    beam_width: int = 5,
    max_len: int = 100,
    device: Union[torch.device, str] = "cpu",
    no_repeat_ngram_size: int = 3,
) -> str:
    """Single-sample beam search for spatial-attention models (C–H).

    Automatically routes LSTM models (C/D/E/F/G) via ``(h, c)`` state and
    Transformer (H) via KV-cache state.  Each beam carries its own state
    independently.

    Args:
        model: Any spatial-attention VQA model (C–H).
        img_tensor: ``FloatTensor (3, 224, 224)`` — single image.
        q_tensor: ``LongTensor (Q,)`` or ``LongTensor (77,)`` — tokenised question.
        vocab_a: Answer vocabulary.
        beam_width: Number of beams to maintain.
        max_len: Maximum tokens per beam.
        device: Device for computation.
        no_repeat_ngram_size: Block repeated n-grams of this size (``0`` = off).

    Returns:
        Best decoded answer string (length-normalised score).
    """
    model.eval()
    with torch.no_grad():
        encoder_hidden, spatial_feats, q_hidden = _encode_spatial(
            model, img_tensor.unsqueeze(0), q_tensor.unsqueeze(0), device
        )
        is_tx = _is_transformer(model.decoder)

        if is_tx:
            # Each beam: (log_score, token_ids, kv_cache)
            beams_tx: List[Tuple[float, List[int], Optional[Tensor]]] = [
                (0.0, [vocab_a.start_idx], None)
            ]
            completed: List[Tuple[float, List[int]]] = []

            for _ in range(max_len):
                if not beams_tx:
                    break
                cands_tx: List[Tuple[float, List[int], Optional[Tensor]]] = []

                for log_score, tokens, b_kv in beams_tx:
                    if tokens[-1] == vocab_a.end_idx:
                        completed.append((log_score, tokens))
                        continue
                    tok   = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
                    logit, new_kv, _alpha, _ = model.decoder.decode_step(
                        tok, b_kv, spatial_feats, q_hidden
                    )
                    lp = F.log_softmax(logit[0], dim=-1)
                    _block_repeated_ngrams(lp, tokens, no_repeat_ngram_size)
                    topk_vals, topk_ids = lp.topk(beam_width)
                    for v, idx in zip(topk_vals.tolist(), topk_ids.tolist()):
                        cands_tx.append((log_score + v, tokens + [idx], new_kv))

                cands_tx.sort(key=lambda x: x[0], reverse=True)
                beams_tx = cands_tx[:beam_width]

            for log_score, tokens, _ in beams_tx:
                completed.append((log_score, tokens))
        else:
            # ── LSTM beams: (log_score, token_ids, h, c, coverage) ───────────
            h0, c0 = encoder_hidden
            beams_lstm: List[Tuple[float, List[int], Tensor, Tensor, Optional[Tensor]]] = [
                (0.0, [vocab_a.start_idx], h0, c0, None)
            ]
            completed = []

            for _ in range(max_len):
                if not beams_lstm:
                    break
                cands_lstm: List[Tuple[float, List[int], Tensor, Tensor, Optional[Tensor]]] = []

                for log_score, tokens, bh, bc, b_cov in beams_lstm:
                    if tokens[-1] == vocab_a.end_idx:
                        completed.append((log_score, tokens))
                        continue
                    tok   = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
                    logit, (nh, nc), _alpha, new_cov = model.decoder.decode_step(
                        tok, (bh, bc), spatial_feats, q_hidden, b_cov
                    )
                    lp = F.log_softmax(logit[0], dim=-1)
                    _block_repeated_ngrams(lp, tokens, no_repeat_ngram_size)
                    topk_vals, topk_ids = lp.topk(beam_width)
                    for v, idx in zip(topk_vals.tolist(), topk_ids.tolist()):
                        cands_lstm.append((log_score + v, tokens + [idx], nh, nc, new_cov))

                cands_lstm.sort(key=lambda x: x[0], reverse=True)
                beams_lstm = cands_lstm[:beam_width]

            for log_score, tokens, _, _, _ in beams_lstm:
                completed.append((log_score, tokens))

        if not completed:
            return ""

        completed.sort(key=lambda x: x[0] / max(len(x[1]) - 1, 1), reverse=True)
        best  = completed[0][1][1:]   # strip <start>
        words = [vocab_a.idx2word.get(t, "<unk>") for t in best if t != vocab_a.end_idx]
        return " ".join(words)


def batch_beam_search_decode(
    model: nn.Module,
    img_tensors: Tensor,
    q_tensors: Tensor,
    vocab_a: Vocabulary,
    beam_width: int = 5,
    max_len: int = 100,
    device: Union[torch.device, str] = "cpu",
    no_repeat_ngram_size: int = 3,
) -> List[str]:
    """Batch wrapper for beam search on global-vector models (A, B).

    Calls ``beam_search_decode`` for each sample in the batch.

    Args:
        model: ``VQAModelA`` or ``VQAModelB``.
        img_tensors: ``FloatTensor (B, 3, 224, 224)``.
        q_tensors: ``LongTensor (B, Q)``.
        vocab_a: Answer vocabulary.
        beam_width: Number of beams per sample.
        max_len: Maximum tokens per sequence.
        device: Device for computation.
        no_repeat_ngram_size: N-gram blocking size (``0`` = off).

    Returns:
        List of B decoded answer strings.
    """
    return [
        beam_search_decode(
            model, img_tensors[i], q_tensors[i], vocab_a,
            beam_width=beam_width, max_len=max_len, device=device,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        for i in range(img_tensors.size(0))
    ]


def batch_beam_search_decode_with_attention(
    model: nn.Module,
    img_tensors: Tensor,
    q_tensors: Tensor,
    vocab_a: Vocabulary,
    beam_width: int = 5,
    max_len: int = 100,
    device: Union[torch.device, str] = "cpu",
    no_repeat_ngram_size: int = 3,
) -> List[str]:
    """Batch wrapper for beam search on spatial-attention models (C, D, E).

    Calls ``beam_search_decode_with_attention`` for each sample in the batch.

    Args:
        model: ``VQAModelC``, ``VQAModelD``, or ``VQAModelE``.
        img_tensors: ``FloatTensor (B, 3, 224, 224)``.
        q_tensors: ``LongTensor (B, Q)``.
        vocab_a: Answer vocabulary.
        beam_width: Number of beams per sample.
        max_len: Maximum tokens per sequence.
        device: Device for computation.
        no_repeat_ngram_size: N-gram blocking size (``0`` = off).

    Returns:
        List of B decoded answer strings.
    """
    return [
        beam_search_decode_with_attention(
            model, img_tensors[i], q_tensors[i], vocab_a,
            beam_width=beam_width, max_len=max_len, device=device,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        for i in range(img_tensors.size(0))
    ]


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json as _json
    from PIL import Image
    from torchvision import transforms

    _MODEL_TYPE   = "E"   # change to 'A', 'B', 'C', 'D' for other models
    _DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _VOCAB_Q_PATH = "data/processed/vocab_questions.json"
    _VOCAB_A_PATH = "data/processed/vocab_answers.json"
    _CHECKPOINT   = f"checkpoints/model_{_MODEL_TYPE.lower()}_best.pth"
    _IMAGE_DIR    = "data/raw/train2014"
    _VQA_E_JSON   = "data/vqa_e/VQA-E_train_set.json"

    vocab_q = Vocabulary()
    vocab_q.load(_VOCAB_Q_PATH)
    vocab_a = Vocabulary()
    vocab_a.load(_VOCAB_A_PATH)

    model = load_model_from_checkpoint(
        _MODEL_TYPE, _CHECKPOINT, len(vocab_q), len(vocab_a), device=_DEVICE
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with open(_VQA_E_JSON) as f:
        annotations = _json.load(f)

    sample     = annotations[0]
    q_text     = sample["question"]
    img_id     = sample["img_id"]
    img_path   = os.path.join(_IMAGE_DIR, f"COCO_train2014_{img_id:012d}.jpg")
    img_tensor = transform(Image.open(img_path).convert("RGB"))
    q_tensor   = torch.tensor(vocab_q.numericalize(q_text), dtype=torch.long)

    use_attention = _MODEL_TYPE in ("C", "D", "E")
    if use_attention:
        answer = greedy_decode_with_attention(model, img_tensor, q_tensor, vocab_a, device=_DEVICE)
    else:
        answer = greedy_decode(model, img_tensor, q_tensor, vocab_a, device=_DEVICE)

    print(f"Model    : {_MODEL_TYPE}")
    print(f"Question : {q_text}")
    print(f"Predicted: {answer}")
