"""Supervised training script for VQA models (Phases 1–3).

Runs a 3-phase progressive training pipeline:

    Phase 1 — Base training     : frozen CNN backbone, pure teacher forcing
    Phase 2 — Fine-tuning       : unfreeze CNN top layers, differential LR
    Phase 3 — Scheduled Sampling: gradually replace GT tokens with model predictions

Loss
----
CrossEntropyLoss with label smoothing on teacher-forced decoder output::

    logits  : (B, T, vocab_size)  →  reshape to (B*T, vocab_size)
    targets : (B, T)              →  reshape to (B*T,)
    Loss    = CrossEntropy(logits.view(-1, V), targets.view(-1), ignore_index=0)

``ignore_index=0`` skips <pad> tokens in the loss, which are meaningless
padding positions that would otherwise dominate the gradient.

Usage
-----
    python src/train.py --model E --epochs 15 --lr 1e-3 --batch_size 256 \\
        --num_workers 12 --augment --warmup_epochs 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))

from dataset import VQAEDataset, vqa_collate_fn
from glove_utils import build_glove_matrix
from models.vqa_models import VQAModelA, VQAModelB, VQAModelC, VQAModelD, VQAModelE
from vocab import Vocabulary


# ── Device & GPU optimisations ───────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True           # auto-tune conv algorithms
    torch.backends.cuda.matmul.allow_tf32 = True    # ~2× faster on Ampere+
    torch.backends.cudnn.allow_tf32 = True


def _supports_bf16() -> bool:
    """Return True if the GPU supports BFloat16 (Ampere+ = compute cap ≥ 8.0)."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def _fused_adam_available() -> bool:
    """Return True if fused Adam is available (requires CUDA + PyTorch ≥ 2.0)."""
    if not torch.cuda.is_available():
        return False
    try:
        optim.Adam([torch.zeros(1, device="cuda")], fused=True)
        return True
    except Exception:
        return False


# ── Data paths ───────────────────────────────────────────────────────────────

TRAIN_IMAGE_DIR  = "data/raw/train2014"
TRAIN_VQA_E_JSON = "data/vqa_e/VQA-E_train_set.json"
VAL_IMAGE_DIR    = "data/raw/val2014"
VAL_VQA_E_JSON   = "data/vqa_e/VQA-E_val_set.json"
VOCAB_Q_PATH     = "data/processed/vocab_questions.json"
VOCAB_A_PATH     = "data/processed/vocab_answers.json"


# ── Model factory ─────────────────────────────────────────────────────────────

def get_model(
    model_type: str,
    vocab_q_size: int,
    vocab_a_size: int,
    pretrained_q_emb: Optional[Tensor] = None,
    pretrained_a_emb: Optional[Tensor] = None,
    use_coverage: bool = False,
    dropout: float = 0.5,
) -> nn.Module:
    """Instantiate and return the model corresponding to *model_type*.

    Args:
        model_type: One of ``'A'``, ``'B'``, ``'C'``, ``'D'``, ``'E'``.
        vocab_q_size: Size of the question vocabulary.
        vocab_a_size: Size of the answer vocabulary.
        pretrained_q_emb: Optional GloVe weight matrix for the question encoder.
        pretrained_a_emb: Optional GloVe weight matrix for the answer decoder.
        use_coverage: Enable the coverage mechanism in attention decoders (C/D/E).
        dropout: Dropout probability for embeddings and LSTM inter-layer dropout.

    Returns:
        Initialised ``nn.Module``.

    Raises:
        ValueError: If *model_type* is not one of the supported values.
    """
    kw: dict = dict(
        pretrained_q_emb=pretrained_q_emb,
        pretrained_a_emb=pretrained_a_emb,
        dropout=dropout,
    )
    if model_type == "A":
        return VQAModelA(vocab_size=vocab_q_size, answer_vocab_size=vocab_a_size, **kw)
    if model_type == "B":
        return VQAModelB(vocab_size=vocab_q_size, answer_vocab_size=vocab_a_size, **kw)
    if model_type == "C":
        return VQAModelC(vocab_size=vocab_q_size, answer_vocab_size=vocab_a_size,
                         use_coverage=use_coverage, **kw)
    if model_type == "D":
        return VQAModelD(vocab_size=vocab_q_size, answer_vocab_size=vocab_a_size,
                         use_coverage=use_coverage, **kw)
    if model_type == "E":
        return VQAModelE(vocab_size=vocab_q_size, answer_vocab_size=vocab_a_size,
                         use_coverage=use_coverage, **kw)
    raise ValueError(f"Unknown model type: '{model_type}'. Choose from A, B, C, D, E.")


# ── Scheduled Sampling forward pass ──────────────────────────────────────────

def ss_forward(
    model: nn.Module,
    model_type: str,
    imgs: Tensor,
    questions: Tensor,
    decoder_input: Tensor,
    epsilon: float,
) -> Tensor:
    """Scheduled Sampling forward pass (Bengio et al., 2015).

    At each decode step *t* the decoder receives the ground-truth token
    with probability *epsilon* and its own previous prediction with
    probability ``1 - epsilon``::

        epsilon = 1.0  →  pure teacher forcing  (start of Phase 3)
        epsilon = 0.0  →  fully autoregressive  (same as inference)

    *epsilon* follows an inverse-sigmoid decay across epochs::

        epsilon(epoch) = k / (k + exp(epoch / k))

    which starts near 1.0 and decays smoothly, forcing the model to
    learn to recover from its own prediction errors.

    Args:
        model: VQA model (must expose ``i_encoder``, ``q_encoder``,
            ``decoder``, ``fusion``, and ``num_layers``).
        model_type: ``'A'``/``'B'`` (no attention) or ``'C'``/``'D'``/``'E'``
            (spatial attention).
        imgs: ``FloatTensor (B, 3, 224, 224)``.
        questions: ``LongTensor (B, Q)``.
        decoder_input: ``LongTensor (B, T)`` — GT tokens ``[<start>, w_1, ...]``.
        epsilon: Mixing probability for GT tokens ∈ ``[0, 1]``.

    Returns:
        ``FloatTensor (B, T, vocab_size)`` — stacked logits.
    """
    B       = imgs.size(0)
    max_len = decoder_input.size(1)

    # ── Encode image + question ───────────────────────────────────────────────
    if model_type in ("C", "D", "E"):
        img_features = F.normalize(model.i_encoder(imgs), p=2, dim=-1)
        # img_features: (B, 49, H)

        q_feat, q_hidden = model.q_encoder(questions)
        # q_feat: (B, H) | q_hidden: (B, Q, H)

        if hasattr(model, "q_norm"):
            q_feat = model.q_norm(q_feat)       # Model E: stabilise FiLM generation

        if model_type == "E":
            img_mean      = img_features.mean(dim=1)             # (B, H)
            fusion_global = model.fusion(img_mean, q_feat)       # (B, H)
            h_0_base      = model.init_h_proj(fusion_global)     # (B, H)
            c_0_base      = model.init_c_proj(fusion_global)     # (B, H)
            h = h_0_base.unsqueeze(0).repeat(model.num_layers, 1, 1)  # (L, B, H)
            c = c_0_base.unsqueeze(0).repeat(model.num_layers, 1, 1)  # (L, B, H)
            # FiLM-modulate the 49 spatial features (same as VQAModelE.forward)
            img_features = model.fusion(img_features, q_feat)    # (B, 49, H)
        else:
            fusion = model.fusion(img_features.mean(dim=1), q_feat)   # (B, H)
            h = fusion.unsqueeze(0).repeat(model.num_layers, 1, 1)    # (L, B, H)
            c = torch.zeros_like(h)
    else:
        img_feat  = F.normalize(model.i_encoder(imgs), p=2, dim=1)    # (B, H)
        q_feat, _ = model.q_encoder(questions)                         # (B, H)
        fusion    = model.fusion(img_feat, q_feat)                     # (B, H)
        h = fusion.unsqueeze(0).repeat(model.num_layers, 1, 1)         # (L, B, H)
        c = torch.zeros_like(h)

    hidden: Tuple[Tensor, Tensor] = (h, c)

    # Coverage accumulator for attention models (C/D/E with use_coverage=True).
    # BUG FIX: original code excluded Model E from coverage init.
    coverage: Optional[Tensor] = None
    if model_type in ("C", "D", "E") and model.decoder.use_coverage:
        coverage = imgs.new_zeros(B, img_features.size(1))  # (B, 49)

    # ── Step-by-step decode ───────────────────────────────────────────────────
    current_token = decoder_input[:, 0]   # (B,) — always <start>
    logits_list: List[Tensor] = []

    for t in range(max_len):
        tok = current_token.unsqueeze(1)  # (B, 1)

        if model_type in ("C", "D", "E"):
            logit, hidden, _, coverage = model.decoder.decode_step(
                tok, hidden, img_features, q_hidden, coverage
            )
        else:
            # Direct LSTM cell execution for non-attention models (A/B).
            emb = model.decoder.dropout(model.decoder.embedding(tok))  # (B, 1, embed)
            if model.decoder.embed_proj is not None:
                emb = model.decoder.embed_proj(emb)                    # (B, 1, embed)
            out, hidden = model.decoder.lstm(emb, hidden)              # (B, 1, H)
            logit = model.decoder.fc(model.decoder.out_proj(out.squeeze(1)))  # (B, V)

        logits_list.append(logit)

        if t < max_len - 1:
            if random.random() < epsilon:
                current_token = decoder_input[:, t + 1]       # GT token
            else:
                current_token = logit.detach().argmax(dim=-1)  # model's prediction

    return torch.stack(logits_list, dim=1)  # (B, T, vocab_size)


# ── Main training function ────────────────────────────────────────────────────

def train(
    model_type: str = "A",
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 256,
    resume: Optional[str] = None,
    scheduled_sampling: bool = False,
    ss_k: float = 5.0,
    finetune_cnn: bool = False,
    cnn_lr_factor: float = 0.1,
    num_workers: int = 12,
    weight_decay: float = 1e-5,
    early_stopping_patience: int = 0,
    augment: bool = False,
    use_glove: bool = False,
    glove_dim: int = 300,
    use_coverage: bool = False,
    coverage_lambda: float = 1.0,
    accum_steps: int = 1,
    warmup_epochs: int = 3,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    dropout: float = 0.5,
    no_compile: bool = False,
    grad_clip: float = 5.0,
    label_smoothing: float = 0.1,
) -> None:
    """Run supervised training for *epochs* on a single VQA model.

    The scheduler is a ``SequentialLR`` that runs a linear warmup for
    *warmup_epochs* epochs followed by cosine annealing for the remainder.
    On resume, the scheduler is fast-forwarded by calling ``scheduler.step()``
    once per already-completed epoch — this is critical to avoid the PyTorch
    ``SequentialLR`` bug that causes the LR to jump on resume.

    Args:
        model_type: Architecture selector — ``'A'``–``'E'``.
        epochs: Number of epochs to train in this call (not total).
        lr: Peak learning rate (after warmup).
        batch_size: Samples per mini-batch.
        resume: Path to a ``model_X_resume.pth`` checkpoint to continue from.
        scheduled_sampling: Enable Scheduled Sampling (Phase 3).
        ss_k: Inverse-sigmoid decay rate for Scheduled Sampling.
        finetune_cnn: Unfreeze CNN/ViT top layers for Phase 2 (B/D/E only).
        cnn_lr_factor: LR multiplier for backbone params (default 0.1× base LR).
        num_workers: DataLoader worker processes.
        weight_decay: L2 regularization coefficient.
        early_stopping_patience: Stop if val loss doesn't improve for N epochs
            (0 = disabled).
        augment: Apply data augmentation (train split only).
        use_glove: Initialise embeddings with GloVe vectors.
        glove_dim: GloVe dimension (50/100/200/300).
        use_coverage: Enable the coverage penalty in attention decoders.
        coverage_lambda: Weight for the coverage loss term.
        accum_steps: Gradient accumulation steps.
        warmup_epochs: Epochs for LR warmup. Set 0 to disable (e.g. Phase 2+).
        max_train_samples: Cap training samples (None = use all).
        max_val_samples: Cap validation samples (None = use all).
        dropout: Dropout probability for embeddings and LSTM layers.
        no_compile: Disable ``torch.compile`` (recommended for C/D attention loop).
        grad_clip: Maximum gradient norm for clipping.
        label_smoothing: Label smoothing factor for CrossEntropyLoss.
    """
    os.makedirs("checkpoints", exist_ok=True)

    # ── Vocabulary ────────────────────────────────────────────────────────────
    vocab_q = Vocabulary()
    vocab_q.load(VOCAB_Q_PATH)
    vocab_a = Vocabulary()
    vocab_a.load(VOCAB_A_PATH)

    # ── Datasets & DataLoaders ────────────────────────────────────────────────
    train_dataset = VQAEDataset(
        image_dir=TRAIN_IMAGE_DIR,
        vqa_e_json_path=TRAIN_VQA_E_JSON,
        vocab_q=vocab_q,
        vocab_a=vocab_a,
        split="train2014",
        max_samples=max_train_samples,
        augment=augment,
    )
    val_dataset = VQAEDataset(
        image_dir=VAL_IMAGE_DIR,
        vqa_e_json_path=VAL_VQA_E_JSON,
        vocab_q=vocab_q,
        vocab_a=vocab_a,
        split="val2014",
        max_samples=max_val_samples,
    )
    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    pin_memory = torch.cuda.is_available()
    loader_kwargs = dict(
        collate_fn=vqa_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, **loader_kwargs)

    # ── Optional GloVe embeddings ─────────────────────────────────────────────
    pretrained_q_emb: Optional[Tensor] = None
    pretrained_a_emb: Optional[Tensor] = None
    if use_glove:
        print(f"Loading GloVe {glove_dim}d embeddings ...")
        q_matrix, q_cov = build_glove_matrix(vocab_q, glove_dim=glove_dim)
        a_matrix, a_cov = build_glove_matrix(vocab_a, glove_dim=glove_dim)
        pretrained_q_emb = torch.tensor(q_matrix)
        pretrained_a_emb = torch.tensor(a_matrix)
        print(f"  Q-vocab coverage: {q_cov:.1%} | A-vocab coverage: {a_cov:.1%}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = get_model(
        model_type, len(vocab_q), len(vocab_a),
        pretrained_q_emb=pretrained_q_emb,
        pretrained_a_emb=pretrained_a_emb,
        use_coverage=use_coverage,
        dropout=dropout,
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing)

    # ── Optimizer — differential LR when fine-tuning CNN backbone ────────────
    if finetune_cnn and model_type in ("B", "D", "E"):
        model.i_encoder.unfreeze_top_layers()
        backbone_ids  = {id(p) for p in model.i_encoder.backbone_params()}
        backbone_params = [p for p in model.parameters()
                           if id(p) in backbone_ids and p.requires_grad]
        other_params    = [p for p in model.parameters()
                           if id(p) not in backbone_ids and p.requires_grad]
        optimizer = optim.Adam(
            [{"params": other_params,    "lr": lr},
             {"params": backbone_params, "lr": lr * cnn_lr_factor}],
            weight_decay=weight_decay,
            fused=_fused_adam_available(),
        )
        print(f"CNN fine-tuning  : ON | backbone LR = {lr * cnn_lr_factor:.2e}")
        print(f"  Trainable backbone params: {sum(p.numel() for p in backbone_params):,}")
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=lr,
            weight_decay=weight_decay,
            fused=_fused_adam_available(),
        )

    # ── LR Schedule: Linear Warmup → CosineAnnealing via SequentialLR ────────
    warmup_epochs  = max(warmup_epochs, 0)
    warmup_iters   = max(warmup_epochs, 1)
    remaining      = max(epochs - warmup_epochs, 1)

    warmup_sched = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1 if warmup_epochs > 0 else 1.0,
        end_factor=1.0,
        total_iters=warmup_iters,
    )
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=remaining, eta_min=lr * 0.01
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_iters],
    )

    # ── Mixed Precision ───────────────────────────────────────────────────────
    use_amp   = torch.cuda.is_available()
    use_bf16  = use_amp and _supports_bf16()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler    = GradScaler("cuda", enabled=(use_amp and not use_bf16))

    # ── Tracking state ────────────────────────────────────────────────────────
    history:      Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    history_path  = f"checkpoints/history_model_{model_type.lower()}.json"
    best_val_loss = float("inf")
    start_epoch   = 0

    # ── Resume ────────────────────────────────────────────────────────────────
    if resume is not None:
        if not os.path.exists(resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        print(f"Resuming from: {resume}")
        ckpt = torch.load(resume, map_location=lambda storage, loc: storage)

        # Strip '_orig_mod.' prefix from torch.compile checkpoints.
        clean_sd = {k.replace("_orig_mod.", ""): v
                    for k, v in ckpt["model_state_dict"].items()}
        incompatible = model.load_state_dict(clean_sd, strict=False)
        if incompatible.missing_keys:
            print(f"  [INFO] New keys (freshly initialised): {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"  [WARN] Unexpected keys (ignored):      {incompatible.unexpected_keys}")

        # Restore optimizer only when the param group layout is unchanged.
        saved_groups   = len(ckpt["optimizer_state_dict"]["param_groups"])
        current_groups = len(optimizer.param_groups)
        model_changed  = bool(incompatible.missing_keys or incompatible.unexpected_keys)
        if saved_groups == current_groups and not model_changed:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            print("  Optimizer & scaler state restored.")
        else:
            reasons = []
            if saved_groups != current_groups:
                reasons.append(f"param groups {saved_groups} → {current_groups}")
            if model_changed:
                reasons.append(f"model arch changed ({len(incompatible.missing_keys)} new keys)")
            print(f"  Optimizer skipped ({', '.join(reasons)}) — using fresh optimizer.")

        start_epoch   = ckpt["epoch"]
        best_val_loss = ckpt["best_val_loss"]
        history       = ckpt.get("history", history)

        # ── SequentialLR fast-forward (CRITICAL — DO NOT REMOVE) ─────────────
        # PyTorch's SequentialLR does not resume correctly from its state_dict
        # alone: on resume it resets the sub-scheduler counters and re-runs
        # the warmup, causing the LR to jump by several orders of magnitude.
        # Instead, we always reconstruct a fresh SequentialLR and fast-forward
        # it by calling scheduler.step() once per already-completed epoch.
        # This guarantees the LR exactly matches the cosine schedule position.
        for _ in range(start_epoch):
            scheduler.step()
        print(f"  Resumed at epoch {start_epoch} | best_val: {best_val_loss:.4f} "
              f"| current LR: {scheduler.get_last_lr()[0]:.2e}")

    # ── Pre-training diagnostics ───────────────────────────────────────────────
    print(f"Model: {model_type} | Device: {DEVICE} | Dropout: {dropout}")
    if use_bf16:
        print("AMP: BFloat16 (Ampere+ detected)")
    elif use_amp:
        print("AMP: Float16 + GradScaler")
    if use_coverage and model_type in ("C", "D", "E"):
        print(f"Coverage         : ON | λ = {coverage_lambda}")
    if accum_steps > 1:
        print(f"Grad accumulation: {accum_steps} steps "
              f"(effective batch = {batch_size * accum_steps})")
    if augment:
        print("Data augmentation: ON (RandomHorizontalFlip + ColorJitter)")
    if early_stopping_patience > 0:
        print(f"Early stopping   : patience = {early_stopping_patience}")
    if scheduled_sampling:
        eps_start = ss_k / (ss_k + math.exp(0 / ss_k))
        eps_end   = ss_k / (ss_k + math.exp((start_epoch + epochs - 1) / ss_k))
        print(f"Scheduled Sampling: ON | k={ss_k} | ε: {eps_start:.2f} → {eps_end:.2f}")

    # ── torch.compile ─────────────────────────────────────────────────────────
    if torch.cuda.is_available() and not no_compile:
        try:
            model = torch.compile(model, mode="default", dynamic=True)
            print("torch.compile    : ON  (default | dynamic shapes)")
        except Exception as e:
            print(f"torch.compile    : skipped — {e}")
    else:
        print("torch.compile    : OFF")

    es_counter = 0  # early stopping no-improvement counter

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in tqdm.tqdm(range(start_epoch, start_epoch + epochs)):

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step_idx, (imgs, questions, answer) in enumerate(train_loader):
            imgs      = imgs.to(DEVICE)
            questions = questions.to(DEVICE)
            answer    = answer.to(DEVICE)

            # Teacher forcing layout:
            #   decoder_input  = answer[:, :-1]  →  [<start>, w_1, ..., w_{T-1}]
            #   decoder_target = answer[:,  1:]  →  [w_1, ..., w_{T-1}, <end>]
            decoder_input  = answer[:, :-1]
            decoder_target = answer[:, 1:]

            with autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                if scheduled_sampling:
                    epsilon = ss_k / (ss_k + math.exp(epoch / ss_k))
                    logits  = ss_forward(model, model_type, imgs, questions,
                                         decoder_input, epsilon)
                    coverage_loss = torch.tensor(0.0, device=DEVICE)
                else:
                    result = model(imgs, questions, decoder_input)
                    if isinstance(result, tuple):
                        logits, coverage_loss = result
                    else:
                        logits        = result
                        coverage_loss = torch.tensor(0.0, device=DEVICE)

                vocab_size = logits.size(-1)
                ce_loss = criterion(
                    logits.view(-1, vocab_size),
                    decoder_target.contiguous().view(-1),
                )
                # Total loss = CE + λ·coverage_loss (coverage_loss = 0 when disabled)
                loss = (ce_loss + coverage_lambda * coverage_loss) / accum_steps

            scaler.scale(loss).backward()

            if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps   # undo the /accum_steps for logging

        avg_train_loss = total_loss / len(train_loader)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, questions, answer in val_loader:
                imgs      = imgs.to(DEVICE)
                questions = questions.to(DEVICE)
                answer    = answer.to(DEVICE)

                decoder_input  = answer[:, :-1]
                decoder_target = answer[:, 1:]

                with autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    result = model(imgs, questions, decoder_input)
                    logits = result[0] if isinstance(result, tuple) else result
                    vocab_size = logits.size(-1)
                    loss = criterion(
                        logits.view(-1, vocab_size),
                        decoder_target.contiguous().view(-1),
                    )
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        current_lr   = optimizer.param_groups[0]["lr"]

        log_msg = (f"Epoch {epoch + 1}/{start_epoch + epochs} | "
                   f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
                   f"LR: {current_lr:.2e}")
        if scheduled_sampling:
            log_msg += f" | SS ε: {ss_k / (ss_k + math.exp(epoch / ss_k)):.3f}"
        print(log_msg)

        # ── LR Scheduler step ─────────────────────────────────────────────────
        # Call scheduler.step() on the unified SequentialLR — it automatically
        # delegates to warmup_sched or cosine_sched based on its milestone.
        scheduler.step()

        # ── History ───────────────────────────────────────────────────────────
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # ── Checkpoint saving ─────────────────────────────────────────────────
        comparison_epochs = {10, 15, 20, 25, 30}
        current_epoch     = epoch + 1

        # Strip torch.compile prefix so checkpoints work outside compile context.
        clean_sd = {k.replace("_orig_mod.", ""): v
                    for k, v in model.state_dict().items()}

        resume_ckpt = {
            "epoch":                current_epoch,
            "model_state_dict":     clean_sd,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict":    scaler.state_dict(),
            "best_val_loss":        best_val_loss,
            "history":              history,
        }
        torch.save(resume_ckpt,
                   f"checkpoints/model_{model_type.lower()}_resume.pth")

        if current_epoch in comparison_epochs:
            torch.save(clean_sd,
                       f"checkpoints/model_{model_type.lower()}_epoch{current_epoch}.pth")
            print(f"  Saved milestone checkpoint: epoch {current_epoch}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            es_counter    = 0
            torch.save(clean_sd,
                       f"checkpoints/model_{model_type.lower()}_best.pth")
            print(f"  → New best val loss: {best_val_loss:.4f} — saved.")
        else:
            # Don't count warmup epochs as no-improvement — LR ramp naturally
            # destabilises complex models (C/D/E), inflating val loss temporarily.
            if epoch >= start_epoch + warmup_epochs:
                es_counter += 1
            else:
                print("  Val loss did not improve (warmup — not counting).")
            if early_stopping_patience > 0 and es_counter > 0:
                print(f"  No improvement: {es_counter}/{early_stopping_patience}")
                if es_counter >= early_stopping_patience:
                    print(f"  Early stopping triggered at epoch {current_epoch}.")
                    # Copy best → milestone so compare.py can always find a checkpoint.
                    target_epoch   = start_epoch + epochs
                    milestone_path = (f"checkpoints/model_{model_type.lower()}"
                                      f"_epoch{target_epoch}.pth")
                    best_path      = f"checkpoints/model_{model_type.lower()}_best.pth"
                    if not os.path.exists(milestone_path) and os.path.exists(best_path):
                        shutil.copy2(best_path, milestone_path)
                        print(f"  Copied best → {milestone_path} (for compare.py)")
                    break


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VQA model (Phases 1–3).")
    parser.add_argument("--model",           type=str,   default="A",
                        choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--epochs",          type=int,   default=10)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--batch_size",      type=int,   default=128)
    parser.add_argument("--resume",          type=str,   default=None)
    parser.add_argument("--scheduled_sampling", action="store_true")
    parser.add_argument("--ss_k",            type=float, default=5.0)
    parser.add_argument("--finetune_cnn",    action="store_true")
    parser.add_argument("--cnn_lr_factor",   type=float, default=0.1)
    parser.add_argument("--num_workers",     type=int,   default=4)
    parser.add_argument("--weight_decay",    type=float, default=1e-5)
    parser.add_argument("--early_stopping",  type=int,   default=0)
    parser.add_argument("--augment",         action="store_true")
    parser.add_argument("--glove",           action="store_true")
    parser.add_argument("--glove_dim",       type=int,   default=300)
    parser.add_argument("--coverage",        action="store_true")
    parser.add_argument("--coverage_lambda", type=float, default=1.0)
    parser.add_argument("--accum_steps",     type=int,   default=1)
    parser.add_argument("--warmup_epochs",   type=int,   default=3)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples",   type=int, default=None)
    parser.add_argument("--dropout",         type=float, default=0.5)
    parser.add_argument("--no_compile",      action="store_true")
    parser.add_argument("--grad_clip",       type=float, default=5.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    args = parser.parse_args()

    train(
        model_type=args.model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        resume=args.resume,
        scheduled_sampling=args.scheduled_sampling,
        ss_k=args.ss_k,
        finetune_cnn=args.finetune_cnn,
        cnn_lr_factor=args.cnn_lr_factor,
        num_workers=args.num_workers,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping,
        augment=args.augment,
        use_glove=args.glove,
        glove_dim=args.glove_dim,
        use_coverage=args.coverage,
        coverage_lambda=args.coverage_lambda,
        accum_steps=args.accum_steps,
        warmup_epochs=args.warmup_epochs,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        dropout=args.dropout,
        no_compile=args.no_compile,
        grad_clip=args.grad_clip,
        label_smoothing=args.label_smoothing,
    )
