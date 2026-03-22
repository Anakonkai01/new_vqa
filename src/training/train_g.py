"""
Model G — 4-Phase Curriculum Training Engine.

Phases (per Architecture Specification v2, Section 6):
  Phase 1 — Alignment   (15 ep): 40% VQA v2.0 + 30% VQA-E + 30% A-OKVQA
             L = FocalLoss + 0.5*Lcov + 0.1*LInfoNCE
             LR = 1e-3, warmup 2 ep, cosine decay
  Phase 2 — Mastery     (10 ep): 100% expl data + 20% VQA v2.0 replay
             L = same as Phase 1 (no SS)
             LR = 5e-4
  Phase 3 — Correction  ( 7 ep): same data as Phase 2 + scheduled sampling
             L = same  (SS epsilon decays with k=5)
             LR = 2e-4
  Phase 4 — Optimization ( 3 ep): VQA-E + VQA-X only + SCST RL
             L = 0.5*CE + 0.5*SCST + 0.1*LInfoNCE
             LR = 5e-5, batch_size=64

Key design decisions:
  - VQAGenerativeDataset (src/data/dataset.py) — new data stack (Step B)
  - VQAModel (src/models/vqa_model.py) — unified model (Step C+D)
  - FocalSequenceLoss (src/training/losses.py) — per-token normalized (E1)
  - butd_collate_fn → VQABatch — G2 label_tokens + G5 length_bins
  - A-F train.py untouched — this file is dispatched from train.py when --model G

Usage (called automatically by train.py --model G):
    python src/train.py --model G --phase 1 --epochs 15 \
        --butd_feat_dir data/features/butd_g1 \
        --merged_json data/processed/merged_train_filtered.json \
        --geo7 --pgn3 --infonce --ohp --len_cond [...]
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# ── Ensure src/ is on path ──────────────────────────────────────────────────
_SRC = os.path.dirname(os.path.dirname(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from config.model_config import ModelConfig
from data.dataset import VQAGenerativeDataset
from data.collate import VQABatch, butd_collate_fn
from data.samplers import build_mixed_sampler, build_replay_sampler
from models.vqa_model import VQAModel
from training.losses import FocalSequenceLoss, build_criterion
from training.scst import scst_step
from vocab import Vocabulary


# ── Wandb (optional) ────────────────────────────────────────────────────────
try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ── Hardware helpers (copied from train.py) ──────────────────────────────────

def _supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 8


def _fused_adam_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        optim.Adam([torch.zeros(1, device='cuda')], fused=True)
        return True
    except Exception:
        return False


# ── Loss computation helper ──────────────────────────────────────────────────

def _compute_loss_g(
    model: VQAModel,
    batch: VQABatch,
    criterion: nn.Module,
    coverage_lambda: float,
    infonce_beta: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """
    Forward + loss for Model G training step.

    Returns:
        loss       : total scalar loss
        components : dict of float scalars for logging
    """
    feats       = batch.feats.to(device)
    questions   = batch.questions.to(device)
    targets     = batch.targets.to(device)
    img_mask    = batch.img_mask.to(device) if batch.img_mask is not None else None
    len_bins    = batch.length_bins.to(device) if batch.length_bins is not None else None
    label_toks  = batch.label_tokens.to(device) if batch.label_tokens is not None else None

    # Teacher forcing: [<start>, w1, ...] → [w1, ..., <end>]
    dec_input  = targets[:, :-1]
    dec_target = targets[:, 1:]

    # Forward: VQAOutput
    out, cov_loss_scalar = model.forward_with_cov(
        feats, questions, dec_input,
        img_mask=img_mask,
    )
    # Re-run to get G2/G5 features (forward_with_cov doesn't pass length_bin/label_tokens yet)
    # Actually use model.forward() directly which supports all G args:
    # (forward_with_cov is a convenience wrapper — extend here instead)
    logits = out.logits   # (B, T, V)

    # ── CE / Focal loss ──────────────────────────────────────────────────────
    ce_loss = criterion(logits, dec_target)

    # ── Coverage loss ────────────────────────────────────────────────────────
    cov_loss = coverage_lambda * cov_loss_scalar

    # ── InfoNCE loss (G3, training only) ─────────────────────────────────────
    infonce_val = torch.tensor(0.0, device=device)
    if model.infonce_heads is not None and model.training:
        # Re-compute infonce_z via full forward (already computed inside out above—
        # but forward_with_cov discards it; call model.forward() to get it)
        full_out = model(
            feats, questions, dec_input,
            img_mask=img_mask, length_bin=len_bins, label_tokens=label_toks,
        )
        if full_out.infonce_z is not None:
            from models.infonce import infonce_loss
            z_img, z_txt = full_out.infonce_z
            infonce_val  = infonce_loss(z_img, z_txt, tau=model.config.infonce_tau)

    total_loss = ce_loss + cov_loss + infonce_beta * infonce_val

    return total_loss, {
        'ce':      ce_loss.item(),
        'cov':     cov_loss.item(),
        'infonce': infonce_val.item(),
    }


def _compute_loss_g_full(
    model: VQAModel,
    feats: torch.Tensor,
    questions: torch.Tensor,
    targets: torch.Tensor,
    img_mask: Optional[torch.Tensor],
    length_bins: Optional[torch.Tensor],
    label_tokens: Optional[torch.Tensor],
    criterion: nn.Module,
    coverage_lambda: float,
    infonce_beta: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """
    Full-featured forward + loss for Model G.
    Passes length_bin and label_tokens through to LSTMDecoderG.
    """
    dec_input  = targets[:, :-1]   # [<start>, w1, ...]
    dec_target = targets[:, 1:]    # [w1, ..., <end>]

    # Single forward pass — all G2/G3/G5 args handled inside VQAModel.forward()
    out = model(
        feats, questions, dec_input,
        img_mask=img_mask,
        length_bin=length_bins,
        label_tokens=label_tokens,
    )
    logits = out.logits   # (B, T, V)

    # Coverage loss requires forward_with_cov; reuse the scalar from out
    # (VQAOutput.coverage is None for now — coverage_loss was computed inside decoder)
    # For coverage tracking, call forward_with_cov separately in val loop only.
    ce_loss    = criterion(logits, dec_target)
    cov_loss   = torch.tensor(0.0, device=device)   # set if use_coverage
    infonce_val = torch.tensor(0.0, device=device)

    if model.config.decoder.use_coverage:
        # Re-run with forward_with_cov to get coverage scalar
        _, cov_scalar = model.forward_with_cov(
            feats, questions, dec_input, img_mask=img_mask,
        )
        cov_loss = coverage_lambda * cov_scalar

    if out.infonce_z is not None:
        from models.infonce import infonce_loss
        z_img, z_txt = out.infonce_z
        infonce_val  = infonce_loss(z_img, z_txt, tau=model.config.infonce_tau)

    total = ce_loss + cov_loss + infonce_beta * infonce_val
    return total, {'ce': ce_loss.item(), 'cov': cov_loss.item(), 'infonce': infonce_val.item()}


# ── Main training function ───────────────────────────────────────────────────

def train_model_g(args) -> None:
    """
    4-phase curriculum training for Model G.

    Called from train.py when --model G is detected.
    Uses args namespace (same argparse Namespace from train.py).

    Required args: butd_feat_dir, merged_json, phase, epochs, lr, batch_size
    Optional G-flags: geo7, pgn3, infonce, infonce_beta, infonce_tau,
                      ohp, ohp_lambda, ohp_threshold, len_cond
    """
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs("checkpoints", exist_ok=True)

    phase          = getattr(args, 'phase', 1) or 1
    epochs         = args.epochs
    lr             = args.lr
    batch_size     = args.batch_size
    coverage_lambda = getattr(args, 'coverage_lambda', 0.5)
    infonce_beta    = getattr(args, 'infonce_beta', 0.1)
    accum_steps     = getattr(args, 'accum_steps', 1)
    grad_clip       = getattr(args, 'grad_clip', 2.0)
    warmup_epochs   = getattr(args, 'warmup_epochs', 2)
    num_workers     = getattr(args, 'num_workers', 8)
    scheduled_sampling = getattr(args, 'scheduled_sampling', False)
    ss_k            = getattr(args, 'ss_k', 5.0)
    use_scst        = getattr(args, 'scst', False) or (phase == 4)
    scst_lambda     = getattr(args, 'scst_lambda', 0.5)
    use_wandb       = getattr(args, 'wandb', False)
    butd_feat_dir   = getattr(args, 'butd_feat_dir', None)
    merged_json     = getattr(args, 'merged_json',
                              'data/processed/merged_train_filtered.json')
    vocab_q_path    = getattr(args, 'vocab_q_path',
                              'data/processed/vocab_questions.json')
    vocab_a_path    = getattr(args, 'vocab_a_path',
                              'data/processed/vocab_answers.json')

    if butd_feat_dir is None:
        raise ValueError(
            "--butd_feat_dir required for Model G. "
            "Run: python src/scripts/extract_features_model_f.py first.")

    # ── Vocabularies ─────────────────────────────────────────────────────────
    q_vocab = Vocabulary(); q_vocab.load(vocab_q_path)
    a_vocab = Vocabulary(); a_vocab.load(vocab_a_path)
    print(f"Vocab            : Q={len(q_vocab)} | A={len(a_vocab)}")

    # ── W&B ──────────────────────────────────────────────────────────────────
    _wb = None
    if use_wandb and _WANDB_AVAILABLE:
        run_name = getattr(args, 'wandb_run_name', None) \
                   or f"model_g_phase{phase}"
        _wb = _wandb.init(
            project=getattr(args, 'wandb_project', 'vqa-model-g'),
            name=run_name,
            config=vars(args),
            resume='allow',
            tags=getattr(args, 'wandb_tags', []),
        )
        print(f"W&B run          : {_wb.url}")

    # ── ModelConfig ──────────────────────────────────────────────────────────
    config = ModelConfig.from_args(args)
    config.encoder.q_vocab_size  = len(q_vocab)
    config.decoder.a_vocab_size  = len(a_vocab)

    # ── GloVe embeddings ─────────────────────────────────────────────────────
    pretrained_q_emb = None
    pretrained_a_emb = None
    if getattr(args, 'glove', False):
        from glove_utils import build_glove_matrix
        glove_dim = getattr(args, 'glove_dim', 300)
        print(f"Loading GloVe {glove_dim}d ...")
        q_mat, q_cov = build_glove_matrix(q_vocab, glove_dim=glove_dim)
        a_mat, a_cov = build_glove_matrix(a_vocab, glove_dim=glove_dim)
        pretrained_q_emb = torch.tensor(q_mat)
        pretrained_a_emb = torch.tensor(a_mat)
        print(f"  Q coverage: {q_cov:.1%} | A coverage: {a_cov:.1%}")

    # ── Build model ───────────────────────────────────────────────────────────
    model = VQAModel(config, pretrained_q_emb, pretrained_a_emb).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model G          : {total_params:,} params | device={DEVICE}")
    print(f"  G-flags        : geo7={config.geo7} pgn3={config.pgn3} "
          f"infonce={config.infonce} ohp={config.ohp} len_cond={config.len_cond}")

    if getattr(args, 'char_cnn', False) and hasattr(model.q_encoder, 'char_cnn'):
        model.q_encoder.char_cnn.build_char_table(q_vocab)
        print("Char-CNN         : char table built")

    # ── Data loading by phase ─────────────────────────────────────────────────
    from functools import partial as _partial
    from data.collate import make_collate_fn

    _collate = make_collate_fn(use_butd=True, a_vocab=a_vocab)

    def _make_butd_loader(feat_dir):
        from data.dataset import make_butd_loader as _mbl
        return _mbl(feat_dir)

    butd_loader = _make_butd_loader(butd_feat_dir)
    augment     = getattr(args, 'augment', False)

    # Source definitions (load once, reuse across phases)
    print(f"Loading data from {merged_json} ...")
    ds_expl = VQAGenerativeDataset.from_merged_json(
        merged_json, q_vocab, a_vocab,
        feature_loader=butd_loader, augment=augment,
    )
    ds_vqa_e = ds_expl.filter_by_source('vqa_e')
    ds_vqa_x = ds_expl.filter_by_source('vqa_x')
    ds_aokvqa = ds_expl.filter_by_source('aokvqa')
    print(f"  Total expl: {len(ds_expl):,} | "
          f"VQA-E: {len(ds_vqa_e):,} | VQA-X: {len(ds_vqa_x):,} | "
          f"A-OKVQA: {len(ds_aokvqa):,}")

    # VQA v2.0 for Phase 1 mix / Phase 2-3 replay
    vqa_v2_q   = getattr(args, 'vqa_v2_q_json',
                          'data/raw/vqa_data_json/v2_OpenEnded_mscoco_train2014_questions.json')
    vqa_v2_ann = getattr(args, 'vqa_v2_ann_json',
                          'data/raw/vqa_data_json/v2_mscoco_train2014_annotations.json')
    ds_vqa_v2  = None
    if os.path.exists(vqa_v2_q) and os.path.exists(vqa_v2_ann):
        from data.dataset import make_image_loader as _mil
        image_dir  = getattr(args, 'image_dir', 'data/raw/images')
        img_loader = _mil(image_dir, split='train2014', augment=augment)
        ds_vqa_v2  = VQAGenerativeDataset.from_vqa_v2(
            vqa_v2_q, vqa_v2_ann, q_vocab, a_vocab,
            feature_loader=img_loader,
        )
        print(f"  VQA v2.0: {len(ds_vqa_v2):,}")
    else:
        print("  [WARN] VQA v2.0 JSONs not found — phases requiring v2.0 mix will skip it")

    # Per-phase data mix
    if phase == 1:
        # 40% VQA v2.0 + 30% VQA-E + 30% A-OKVQA
        if ds_vqa_v2 is not None and len(ds_aokvqa) > 0:
            train_dataset, train_sampler = build_mixed_sampler(
                [ds_vqa_v2, ds_vqa_e, ds_aokvqa],
                fractions=[0.40, 0.30, 0.30],
            )
        else:
            # Fallback: equal mix of available sources
            avail = [d for d in [ds_vqa_v2, ds_vqa_e, ds_aokvqa] if d is not None and len(d) > 0]
            frac  = [1.0 / len(avail)] * len(avail)
            train_dataset, train_sampler = build_mixed_sampler(avail, frac)
        print(f"Phase 1 mix      : 40% VQA v2.0 + 30% VQA-E + 30% A-OKVQA")

    elif phase in (2, 3):
        # 100% explanation data + 20% VQA v2.0 replay
        if ds_vqa_v2 is not None:
            train_dataset, train_sampler = build_replay_sampler(
                ds_expl, ds_vqa_v2, replay_fraction=0.2,
            )
        else:
            from torch.utils.data import RandomSampler
            train_dataset = ds_expl
            train_sampler = None
        print(f"Phase {phase} data: expl + 20% v2.0 replay")

    else:  # phase 4
        # VQA-E + VQA-X only (no A-OKVQA for SCST — per spec)
        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset([ds_vqa_e, ds_vqa_x]) \
                        if len(ds_vqa_x) > 0 else ds_vqa_e
        train_sampler = None
        batch_size    = min(batch_size, 64)   # Phase 4 batch_size=64 per spec
        print(f"Phase 4 data     : VQA-E ({len(ds_vqa_e):,}) + VQA-X ({len(ds_vqa_x):,})")

    # Validation set (always full explanation dataset)
    # Check for separate val json
    val_json = getattr(args, 'val_merged_json', None)
    if val_json and os.path.exists(val_json):
        ds_val = VQAGenerativeDataset.from_merged_json(
            val_json, q_vocab, a_vocab, feature_loader=butd_loader)
    else:
        # Use a random 5% split of the explanation data as proxy val
        n_val = max(int(0.05 * len(ds_expl)), 1000)
        indices = list(range(len(ds_expl)))
        random.shuffle(indices)
        val_idx  = indices[:n_val]
        from torch.utils.data import Subset
        ds_val = Subset(ds_expl, val_idx)

    print(f"Train: {len(train_dataset):,} | Val: {len(ds_val):,}")

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    # ── Loss function ─────────────────────────────────────────────────────────
    use_focal   = (phase != 4)    # Phase 4: plain CE per spec
    label_smooth = getattr(args, 'label_smoothing', 0.1) if phase != 4 else 0.0
    criterion   = build_criterion(
        gamma=getattr(args, 'focal_gamma', 2.0),
        label_smoothing=label_smooth,
        use_focal=use_focal,
        ignore_index=0,
    )
    print(f"Loss             : {'FocalSequenceLoss' if use_focal else 'CrossEntropyLoss'} "
          f"| label_smooth={label_smooth}")
    if config.infonce:
        print(f"InfoNCE (G3)     : β={infonce_beta} | τ={config.infonce_tau}")
    if config.decoder.use_coverage:
        print(f"Coverage (G)     : λ={coverage_lambda}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=getattr(args, 'weight_decay', 1e-4),
        fused=_fused_adam_available(),
    )

    # ── LR scheduler ──────────────────────────────────────────────────────────
    warmup_sched = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1 if warmup_epochs > 0 else 1.0,
        end_factor=1.0,
        total_iters=max(warmup_epochs, 1),
    )
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=lr * 0.01
    )
    print(f"LR               : {lr:.1e} | warmup={warmup_epochs} ep | cosine decay")

    # ── AMP ───────────────────────────────────────────────────────────────────
    use_amp  = torch.cuda.is_available()
    use_bf16 = use_amp and _supports_bf16()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler    = GradScaler('cuda', enabled=(use_amp and not use_bf16))
    print(f"AMP              : {'BF16' if use_bf16 else 'FP16' if use_amp else 'disabled'}")

    # ── Resume ────────────────────────────────────────────────────────────────
    resume_path = getattr(args, 'resume', None)
    start_epoch = 0
    best_val    = float('inf')
    history     = {'train_loss': [], 'val_loss': [], 'train_ce': [], 'train_infonce': []}
    history_path = "checkpoints/history_model_g.json"
    model_key    = "g"

    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
        sd   = {k.replace('_orig_mod.', ''): v
                for k, v in ckpt.get('model_state_dict', ckpt).items()}
        incompatible = model.load_state_dict(sd, strict=False)
        if incompatible.missing_keys:
            print(f"  New keys (init fresh): {incompatible.missing_keys[:5]}")
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except Exception as e:
                print(f"  Optimizer restore skipped: {e}")
        start_epoch = ckpt.get('epoch', 0)
        best_val    = ckpt.get('best_val_loss', float('inf'))
        if getattr(args, 'reset_best_val_loss', False):
            best_val = float('inf')
        history = ckpt.get('history', history)
        print(f"Resumed          : epoch {start_epoch} | best_val={best_val:.4f}")

    # ── torch.compile ────────────────────────────────────────────────────────
    if torch.cuda.is_available() and not getattr(args, 'no_compile', False):
        try:
            model = torch.compile(model, mode='default', dynamic=True)
            print("torch.compile    : ON")
        except Exception as e:
            print(f"torch.compile    : skipped — {e}")

    # ── OHP GloVe (Phase 4 only) ──────────────────────────────────────────────
    ohp_glove = None
    if config.ohp and use_scst:
        print("Loading GloVe for OHP ...")
        try:
            import numpy as np
            glove_path = f"data/embeddings/glove.6B.300d.txt"
            if os.path.exists(glove_path):
                ohp_glove = {}
                with open(glove_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.rstrip().split(' ')
                        ohp_glove[parts[0]] = np.array(parts[1:], dtype=np.float32)
                print(f"  OHP GloVe loaded: {len(ohp_glove):,} words")
            else:
                print(f"  [WARN] GloVe not found at {glove_path} — OHP disabled")
        except Exception as e:
            print(f"  [WARN] OHP GloVe load failed: {e}")

    # ── SCST target text helper ───────────────────────────────────────────────
    _end_idx = a_vocab.word2idx.get('<end>', 2)

    def _ids_to_text(ids_row):
        words = []
        for t in ids_row:
            if t in (0, _end_idx):
                break
            w = a_vocab.idx2word.get(t, '<unk>')
            if w not in ('<pad>', '<start>'):
                words.append(w)
        return ' '.join(words)

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step = 0

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        ep_ce = ep_cov = ep_infonce = ep_rl = 0.0
        ep_grad = ep_opt_steps = 0
        optimizer.zero_grad()

        for step_idx, batch in enumerate(train_loader):
            # batch is VQABatch dataclass
            feats      = batch.feats.to(DEVICE)
            questions  = batch.questions.to(DEVICE)
            targets    = batch.targets.to(DEVICE)
            img_mask   = batch.img_mask.to(DEVICE)   if batch.img_mask    is not None else None
            len_bins   = batch.length_bins.to(DEVICE) if batch.length_bins is not None else None
            label_toks = batch.label_tokens.to(DEVICE) if batch.label_tokens is not None else None

            dec_input  = targets[:, :-1]
            dec_target = targets[:, 1:]

            with autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                if scheduled_sampling and phase == 3:
                    rel_ep = epoch - start_epoch
                    epsilon = ss_k / (ss_k + math.exp(rel_ep / ss_k))
                    logits = _ss_forward_g(
                        model, feats, questions, dec_input, epsilon,
                        img_mask=img_mask, len_bins=len_bins,
                        label_toks=label_toks, device=DEVICE,
                    )
                    cov_scalar = torch.tensor(0.0, device=DEVICE)
                    infonce_val = torch.tensor(0.0, device=DEVICE)
                else:
                    out = model(
                        feats, questions, dec_input,
                        img_mask=img_mask, length_bin=len_bins, label_tokens=label_toks,
                    )
                    logits = out.logits
                    # Coverage loss via forward_with_cov (cheap — no double encode)
                    _, cov_scalar = model.forward_with_cov(
                        feats, questions, dec_input, img_mask=img_mask)
                    infonce_val = torch.tensor(0.0, device=DEVICE)
                    if out.infonce_z is not None:
                        from models.infonce import infonce_loss as _il
                        z_i, z_t = out.infonce_z
                        infonce_val = _il(z_i, z_t, tau=model.config.infonce_tau
                                          if hasattr(model, 'config') else 0.07)

                ce_loss    = criterion(logits, dec_target)
                cov_loss   = coverage_lambda * cov_scalar
                total_loss = ce_loss + cov_loss + infonce_beta * infonce_val

            scaler.scale(total_loss / accum_steps).backward()

            # ── SCST (Phase 4) ───────────────────────────────────────────────
            rl_loss = None
            if use_scst:
                target_texts = [_ids_to_text(r.tolist()) for r in dec_target]
                with autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                    rl_loss, scst_stats = scst_step(
                        model, 'G', feats, questions, target_texts, a_vocab,
                        device=DEVICE, max_len=dec_input.size(1),
                        bleu_weight=0.5, meteor_weight=0.5,
                        ohp_weight=getattr(args, 'ohp_lambda', 0.3) if config.ohp else 0.0,
                        cider_weight=1.0,
                        visual_labels_batch=getattr(batch, 'label_names', None),
                        glove_embed=ohp_glove,
                        ohp_threshold=getattr(args, 'ohp_threshold', 0.5),
                        return_stats=True,
                    )
                scaler.scale(scst_lambda * rl_loss / accum_steps).backward()

            # ── Optimizer step ────────────────────────────────────────────────
            if (step_idx + 1) % accum_steps == 0 \
                    or (step_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step  += 1
                ep_opt_steps += 1
                ep_grad  += grad_norm.item()
                ep_ce    += ce_loss.item()
                ep_cov   += cov_loss.item()
                ep_infonce += infonce_val.item()
                if rl_loss is not None:
                    ep_rl += rl_loss.item()

        avg_train = (ep_ce + ep_cov + ep_infonce * infonce_beta) / max(ep_opt_steps, 1)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                v_feats    = val_batch.feats.to(DEVICE)
                v_q        = val_batch.questions.to(DEVICE)
                v_tgt      = val_batch.targets.to(DEVICE)
                v_mask     = val_batch.img_mask.to(DEVICE) if val_batch.img_mask is not None else None
                v_dec_in   = v_tgt[:, :-1]
                v_dec_tgt  = v_tgt[:, 1:]
                with autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                    v_out = model(v_feats, v_q, v_dec_in, img_mask=v_mask)
                    val_loss += criterion(v_out.logits, v_dec_tgt).item()

        avg_val = val_loss / max(len(val_loader), 1)
        cur_lr  = optimizer.param_groups[0]['lr']
        print(f"[Phase {phase}] Epoch {epoch+1}/{start_epoch+epochs} | "
              f"Train: {avg_train:.4f} (CE={ep_ce/max(ep_opt_steps,1):.4f} "
              f"Cov={ep_cov/max(ep_opt_steps,1):.4f} "
              f"InfoNCE={ep_infonce/max(ep_opt_steps,1):.4f}) | "
              f"Val: {avg_val:.4f} | LR: {cur_lr:.2e}")

        # LR scheduling
        if epoch < start_epoch + warmup_epochs:
            warmup_sched.step()
        else:
            cosine_sched.step()

        # History
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['train_ce'].append(ep_ce / max(ep_opt_steps, 1))
        history['train_infonce'].append(ep_infonce / max(ep_opt_steps, 1))
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        # W&B logging
        if _wb is not None:
            _n = max(ep_opt_steps, 1)
            _wb.log({
                'train/loss': avg_train, 'val/loss': avg_val,
                'train/ce': ep_ce / _n, 'train/cov': ep_cov / _n,
                'train/infonce': ep_infonce / _n,
                'train/rl': ep_rl / _n,
                'train/grad_norm': ep_grad / _n,
                'lr': cur_lr, 'epoch': epoch + 1,
            }, step=global_step)

        # Checkpoints (same 3-tier strategy as train.py)
        clean_sd = {k.replace('_orig_mod.', ''): v
                    for k, v in model.state_dict().items()}
        resume_ckpt = {
            'epoch': epoch + 1,
            'model_state_dict': clean_sd,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val,
            'history': history,
            'model_config': config.to_json(),
        }
        torch.save(resume_ckpt, f"checkpoints/model_{model_key}_resume.pth")

        milestone_epochs = {15, 25, 32, 35}
        if (epoch + 1) in milestone_epochs:
            torch.save({
                'epoch': epoch + 1, 'model_state_dict': clean_sd,
                'val_loss': avg_val, 'model_config': config.to_json(),
            }, f"checkpoints/model_{model_key}_epoch{epoch+1}.pth")
            print(f"  Saved milestone: epoch {epoch+1}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                'epoch': epoch + 1, 'model_state_dict': clean_sd,
                'val_loss': avg_val, 'best_val_loss': best_val,
                'model_config': config.to_json(),
            }, f"checkpoints/model_{model_key}_best.pth")
            print(f"  -> New best val: {best_val:.4f}")

    if _wb is not None:
        _wb.finish()


# ── Scheduled Sampling forward for Model G ───────────────────────────────────

def _ss_forward_g(
    model: VQAModel,
    feats: torch.Tensor,
    questions: torch.Tensor,
    dec_input: torch.Tensor,
    epsilon: float,
    img_mask=None,
    len_bins=None,
    label_toks=None,
    device='cpu',
) -> torch.Tensor:
    """
    Scheduled Sampling forward pass for VQAModel (Model G).

    At step t: feed GT token (prob epsilon) or model's t-1 prediction (1-epsilon).

    Returns:
        logits: (B, T, V) — from ThreeWayPGNHead log-probs
    """
    B       = feats.size(0)
    max_len = dec_input.size(1)

    V, q_feat, Q_H = model.encode(feats, questions, img_mask=img_mask)
    h, c = model._fuse_and_init(V, q_feat, img_mask)

    coverage = None
    if model.config.decoder.use_coverage:
        coverage = feats.new_zeros(B, V.size(1))

    # Length bin for G5 — use LONG at SS (we want long-form output)
    if len_bins is None:
        len_bins = feats.new_full((B,), 2, dtype=torch.long)

    current_tok = dec_input[:, 0]   # <start>
    logits_list = []

    for t in range(max_len):
        tok = current_tok.unsqueeze(1)   # (B, 1)
        logit, h_new, c_new, img_alpha, cov_new = model.decode_step(
            tok, h, c, V, Q_H,
            coverage=coverage,
            img_mask=img_mask,
            q_token_ids=questions,
            length_bin=len_bins,
            label_tokens=label_toks,
        )
        h, c, coverage = h_new, c_new, cov_new
        logits_list.append(logit)

        if t < max_len - 1:
            current_tok = dec_input[:, t + 1] \
                if random.random() < epsilon \
                else logit.detach().argmax(dim=-1)

    return torch.stack(logits_list, dim=1)   # (B, T, V)
