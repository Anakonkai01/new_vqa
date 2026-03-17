import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import os
import sys
import json
import argparse
import math
import random
import tqdm
sys.path.append(os.path.dirname(__file__))

# Import modules
from dataset import VQAEDataset, vqa_collate_fn
from models.vqa_models import VQAModelA, VQAModelB, VQAModelC, VQAModelD, VQAModelE
from glove_utils import build_glove_matrix
from vocab import Vocabulary
from training.css_augment import CSSAugmentor, css_contrastive_loss
from training.scst import scst_step




"""
CrossEntropy requires 2D predictions: (N, C) and target (N).
Reshape 3D logits to 2D before computing loss:

  logits:  (batch, seq_len, vocab_size)
  targets: (batch, seq_len)

  logits  = logits.view(-1, vocab_size)   # (batch*seq_len, vocab_size)
  targets = targets.view(-1)              # (batch*seq_len)
  loss = criterion(logits, targets)
"""



# Configurations / Hyperparameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── A100 / Ampere+ optimizations ──────────────────────────────────────
if torch.cuda.is_available():
    # Auto-tune conv algorithms for fixed input sizes (faster after warm-up)
    torch.backends.cudnn.benchmark = True
    # TF32 for matmul & convolutions — near-FP32 accuracy, ~2× faster on Ampere+
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _supports_bf16():
    """Check if GPU supports BFloat16 (Ampere+ = compute capability >= 8.0)."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 8


def _fused_adam_available():
    """Check if fused Adam is available (CUDA + PyTorch >= 2.0)."""
    if not torch.cuda.is_available():
        return False
    try:
        # fused=True requires CUDA tensors and PyTorch 2.0+
        optim.Adam([torch.zeros(1, device='cuda')], fused=True)
        return True
    except Exception:
        return False


def get_model(model_type, vocab_q_size, vocab_a_size,
              pretrained_q_emb=None, pretrained_a_emb=None,
              use_coverage=False, dropout=0.5,
              use_layer_norm=False, use_dropconnect=False,
              use_dcan=False, use_mutan=False, use_pgn=False,
              use_q_highway=False, use_char_cnn=False):
    """Factory function: return the model corresponding to model_type."""
    kw = dict(pretrained_q_emb=pretrained_q_emb, pretrained_a_emb=pretrained_a_emb,
              dropout=dropout, use_q_highway=use_q_highway, use_char_cnn=use_char_cnn)
    if model_type == 'A':
        return VQAModelA(vocab_size=vocab_q_size, answer_vocab_size=vocab_a_size, **kw)
    elif model_type == 'B':
        return VQAModelB(vocab_size=vocab_q_size, answer_vocab_size=vocab_a_size, **kw)
    elif model_type == 'C':
        return VQAModelC(vocab_size=vocab_q_size, answer_vocab_size=vocab_a_size,
                         use_coverage=use_coverage,
                         use_layer_norm=use_layer_norm,
                         use_dropconnect=use_dropconnect,
                         use_dcan=use_dcan, use_pgn=use_pgn, **kw)
    elif model_type == 'D':
        return VQAModelD(vocab_size=vocab_q_size, answer_vocab_size=vocab_a_size,
                         use_coverage=use_coverage,
                         use_layer_norm=use_layer_norm,
                         use_dropconnect=use_dropconnect,
                         use_dcan=use_dcan, use_pgn=use_pgn, **kw)
    elif model_type == 'E':
        return VQAModelE(vocab_size=vocab_q_size, answer_vocab_size=vocab_a_size,
                         use_coverage=use_coverage,
                         use_layer_norm=use_layer_norm,
                         use_dropconnect=use_dropconnect,
                         use_dcan=use_dcan,
                         use_mutan=use_mutan, use_pgn=use_pgn, **kw)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from A, B, C, D, E.")


def ss_forward(model, model_type, imgs, questions, decoder_input, epsilon):
    """
    Scheduled Sampling forward pass (token-level, Bengio et al. 2015).

    At each decode step t the decoder receives the ground-truth token with
    probability `epsilon` and its own prediction from step t-1 with probability
    (1 - epsilon).

        epsilon = 1.0  =>  pure teacher forcing  (original behaviour)
        epsilon = 0.0  =>  fully autoregressive  (same as inference)

    epsilon follows an inverse-sigmoid decay schedule:
        epsilon(epoch) = k / (k + exp(epoch / k))
    so it starts close to 1 and gradually decays, forcing the model to
    recover from its own mistakes over the course of training.

    Args:
        model       : VQA model (must expose i_encoder, q_encoder, decoder, num_layers)
        model_type  : 'A'/'B' (LSTMDecoder) or 'C'/'D' (LSTMDecoderWithAttention)
        imgs        : (B, 3, 224, 224)
        questions   : (B, q_len)
        decoder_input: (B, max_len) — GT tokens [<start>, w1, w2, ...]
        epsilon     : float ∈ [0, 1]

    Returns:
        logits: (B, max_len, vocab_size)
    """
    B       = imgs.size(0)
    max_len = decoder_input.size(1)

    # ── Encode image + question ──────────────────────────────────────
    if model_type in ('C', 'D', 'E'):
        img_features = F.normalize(model.i_encoder(imgs), p=2, dim=-1)  # (B, 49, H)
        q_feat, q_hidden = model.q_encoder(questions)                    # (B, H), (B, qlen, H)
        img_mean = img_features.mean(dim=1)
        # Model E: MUTAN expects (q, v) order; C/D GatedFusion: (img_mean, q_feat)
        if model_type == 'E':
            fusion = model.fusion(q_feat, img_mean)
        else:
            fusion = model.fusion(img_mean, q_feat)
    else:
        img_feat = F.normalize(model.i_encoder(imgs), p=2, dim=1)       # (B, H)
        q_feat, _ = model.q_encoder(questions)                           # (B, H)
        fusion    = model.fusion(img_feat, q_feat)                       # (B, H)

    h = fusion.unsqueeze(0).repeat(model.num_layers, 1, 1)  # (L, B, H)
    c = torch.zeros_like(h)
    hidden = (h, c)

    # Coverage vector for models C/D/E
    coverage = None
    if model_type in ('C', 'D', 'E') and model.decoder.use_coverage:
        coverage = imgs.new_zeros(B, img_features.size(1))  # (B, 49)

    # ── Step-by-step decoding ────────────────────────────────────────
    current_token = decoder_input[:, 0]   # (B,) — first token is always <start>
    logits_list   = []

    for t in range(max_len):
        tok = current_token.unsqueeze(1)  # (B, 1)

        if model_type in ('C', 'D', 'E'):
            logit, hidden, _, coverage = model.decoder.decode_step(
                tok, hidden, img_features, q_hidden, coverage, q_token_ids=questions
            )
            # logit: (B, vocab), hidden: updated (h, c)
        else:
            emb        = model.decoder.dropout(model.decoder.embedding(tok)) # (B, 1, embed)
            if model.decoder.embed_proj is not None:
                emb = model.decoder.embed_proj(emb)
            out, hidden = model.decoder.lstm(emb, hidden)        # (B, 1, H)
            logit      = model.decoder.fc(model.decoder.out_proj(out.squeeze(1)))  # (B, vocab)

        logits_list.append(logit)

        # Choose next input: ground truth (prob epsilon) or own prediction
        if t < max_len - 1:
            if random.random() < epsilon:
                current_token = decoder_input[:, t + 1]          # GT token
            else:
                current_token = logit.detach().argmax(dim=-1)    # model's prediction

    return torch.stack(logits_list, dim=1)  # (B, max_len, vocab_size)


def css_forward(model, model_type, imgs, questions, augmentor):
    """
    Tier-6: Compute CSS contrastive loss without running the full decoder.
    Encodes both real and counterfactual samples, compares fused representations.

    Returns: scalar contrastive loss tensor
    """
    # Encode real batch (reuse encoder already warmed up)
    with torch.no_grad():
        img_features = F.normalize(model.i_encoder(imgs), p=2, dim=-1)  # (B, N, H)
        img_mean     = img_features.mean(dim=1)                          # (B, H)
        q_feature, _ = model.q_encoder(questions)                        # (B, H)

    # Generate counterfactuals (no gradient — pure augmentation)
    cf_img_feats, cf_questions = augmentor(questions, img_features.detach())

    # Compute fused real representation (WITH gradient for the fusion/encoders)
    img_features_grad = F.normalize(model.i_encoder(imgs), p=2, dim=-1)
    img_mean_grad     = img_features_grad.mean(dim=1)
    q_feature_grad, _ = model.q_encoder(questions)
    if model_type == 'E':
        f_real = model.fusion(q_feature_grad, img_mean_grad)    # MUTAN: q first
    else:
        f_real = model.fusion(img_mean_grad, q_feature_grad)    # GatedFusion: img first

    # Visual CF: zeroed image regions + original questions (no gradient through cf_img)
    cf_img_mean = cf_img_feats.mean(dim=1)  # (B, H)  — already detached
    if model_type == 'E':
        f_cf_visual = model.fusion(q_feature_grad, cf_img_mean)
    else:
        f_cf_visual = model.fusion(cf_img_mean, q_feature_grad)

    # Linguistic CF: original image + masked questions (re-encode cf_questions)
    cf_q_feat, _ = model.q_encoder(cf_questions)
    if model_type == 'E':
        f_cf_ling = model.fusion(cf_q_feat, img_mean_grad)
    else:
        f_cf_ling = model.fusion(img_mean_grad, cf_q_feat)

    return css_contrastive_loss(f_real, f_cf_visual, f_cf_ling)


# ── Training set (train2014) ─────────────────────────────────────
TRAIN_IMAGE_DIR  = "data/raw/train2014"
TRAIN_VQA_E_JSON = "data/vqa_e/VQA-E_train_set.json"

# ── Validation set (val2014) ──────────────────────────────────────
VAL_IMAGE_DIR  = "data/raw/val2014"
VAL_VQA_E_JSON = "data/vqa_e/VQA-E_val_set.json"

VOCAB_Q_PATH  = "data/processed/vocab_questions.json"
VOCAB_A_PATH  = "data/processed/vocab_answers.json"

# Set to a number (e.g. 10000) to cap samples for quick pipeline tests.
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES   = None


def train(model_type='A', epochs=10, lr=1e-3, batch_size=128, resume=None,
          scheduled_sampling=False, ss_k=5,
          finetune_cnn=False, cnn_lr_factor=0.1,
          num_workers=4, weight_decay=1e-5,
          early_stopping_patience=0, augment=False,
          use_glove=False, glove_dim=300,
          use_coverage=False, coverage_lambda=1.0,
          accum_steps=1, warmup_epochs=3,
          max_train_samples=None, max_val_samples=None,
          dropout=0.5, no_compile=False,
          grad_clip=5.0, label_smoothing=0.1,
          use_mutan=False, use_pgn=False,
          use_css=False, css_lambda=0.5, css_margin=1.0,
          use_q_highway=False, use_char_cnn=False,
          use_scst=False, scst_lambda=0.5):
    os.makedirs("checkpoints", exist_ok=True)

    vocab_q = Vocabulary(); vocab_q.load(VOCAB_Q_PATH)
    vocab_a = Vocabulary(); vocab_a.load(VOCAB_A_PATH)

    train_dataset = VQAEDataset(
        image_dir=TRAIN_IMAGE_DIR,
        vqa_e_json_path=TRAIN_VQA_E_JSON,
        vocab_q=vocab_q,
        vocab_a=vocab_a,
        split='train2014',
        max_samples=max_train_samples or MAX_TRAIN_SAMPLES,
        augment=augment
    )

    val_dataset = VQAEDataset(
        image_dir=VAL_IMAGE_DIR,
        vqa_e_json_path=VAL_VQA_E_JSON,
        vocab_q=vocab_q,
        vocab_a=vocab_a,
        split='val2014',
        max_samples=max_val_samples or MAX_VAL_SAMPLES
    )

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=vqa_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=vqa_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    # ── GloVe pretrained embeddings (optional) ────────────────────
    pretrained_q_emb = None
    pretrained_a_emb = None
    if use_glove:
        import torch as _t
        print(f"Loading GloVe {glove_dim}d embeddings ...")
        q_matrix, q_cov = build_glove_matrix(vocab_q, glove_dim=glove_dim)
        a_matrix, a_cov = build_glove_matrix(vocab_a, glove_dim=glove_dim)
        pretrained_q_emb = _t.tensor(q_matrix)
        pretrained_a_emb = _t.tensor(a_matrix)
        print(f"  Q-vocab coverage: {q_cov:.1%} | A-vocab coverage: {a_cov:.1%}")

    model     = get_model(model_type, len(vocab_q), len(vocab_a),
                           pretrained_q_emb=pretrained_q_emb,
                           pretrained_a_emb=pretrained_a_emb,
                           use_coverage=use_coverage,
                           dropout=dropout,
                           use_layer_norm=getattr(args, 'layer_norm', False),
                           use_dropconnect=getattr(args, 'dropconnect', False),
                           use_dcan=getattr(args, 'dcan', False),
                           use_mutan=getattr(args, 'use_mutan', False),
                           use_pgn=getattr(args, 'pgn', False),
                           use_q_highway=getattr(args, 'q_highway', False),
                           use_char_cnn=getattr(args, 'char_cnn', False)).to(DEVICE)

    # Tier 7C: if char_cnn requested, build char table from vocabulary
    if getattr(args, 'char_cnn', False) and hasattr(model.q_encoder, 'char_cnn'):
        model.q_encoder.char_cnn.build_char_table(vocab_q)
        print("Char-CNN         : built char table from vocab_questions")
    # PGN outputs log-probabilities (from PointerGeneratorHead.blend) instead of raw logits,
    # so we must use NLLLoss. NLLLoss does not support label_smoothing natively; we skip it.
    # Non-PGN models still use CrossEntropyLoss with label_smoothing.
    if use_pgn:
        criterion = nn.NLLLoss(ignore_index=0)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing)

    # ── Optimizer — differential LR when fine-tuning the CNN backbone ──────────
    # Models A/C use scratch CNN → finetune_cnn flag has no effect on them.
    # Models B/D use frozen ResNet → finetune_cnn selectively unfreezes
    # layer3 + layer4 with a smaller LR (cnn_lr_factor × base_lr) to avoid
    # catastrophic forgetting of pretrained ImageNet knowledge.
    if finetune_cnn and model_type in ('B', 'D', 'E'):
        model.i_encoder.unfreeze_top_layers()
        backbone_param_ids = {id(p) for p in model.i_encoder.backbone_params()}
        backbone_params = [p for p in model.parameters()
                           if id(p) in backbone_param_ids and p.requires_grad]
        other_params    = [p for p in model.parameters()
                           if id(p) not in backbone_param_ids and p.requires_grad]
        optimizer = optim.Adam([
            {'params': other_params,    'lr': lr},
            {'params': backbone_params, 'lr': lr * cnn_lr_factor},
        ], weight_decay=weight_decay, fused=_fused_adam_available())
        print(f"CNN fine-tuning  : ON | backbone LR = {lr * cnn_lr_factor:.2e}  other LR = {lr:.2e}")
        print(f"  trainable backbone params : {sum(p.numel() for p in backbone_params):,}")
    else:
        optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay,
                               fused=_fused_adam_available())

    if weight_decay > 0:
        print(f"Weight decay     : {weight_decay:.1e}")

    # ── LR Warmup + ReduceLROnPlateau ─────────────────────────────
    # Warm up linearly from lr/10 → lr over first warmup_epochs epochs,
    # then switch to ReduceLROnPlateau (halve when val loss plateaus).
    #
    # IMPORTANT: LinearLR auto-calls .step() at init (last_epoch=-1 → 0),
    # which immediately multiplies LR by start_factor. When warmup_epochs=0
    # (Phase 2/3 resume), we must use start_factor=1.0 so the LR is unchanged.
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1 if warmup_epochs > 0 else 1.0,
        end_factor=1.0,
        total_iters=max(warmup_epochs, 1)
    )
    # CosineAnnealingLR: smooth decay from peak LR to eta_min over remaining epochs
    # Much better than ReduceLROnPlateau for fixed-length training runs
    remaining_epochs = max(epochs - warmup_epochs, 1)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=remaining_epochs, eta_min=lr * 0.01
    )
    if warmup_epochs > 0:
        print(f"LR Warmup        : {warmup_epochs} epochs (lr/10 → lr)")
    else:
        print(f"LR Warmup        : DISABLED (resume training)")
    print(f"LR Schedule      : CosineAnnealing (T_max={remaining_epochs}, eta_min={lr*0.01:.1e})")

    # Mixed precision — BF16 on Ampere+ (A100, etc.), FP16 elsewhere
    use_amp  = torch.cuda.is_available()
    use_bf16 = use_amp and _supports_bf16()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    # GradScaler not needed for BF16 (wider dynamic range), still used for FP16
    scaler = GradScaler('cuda', enabled=(use_amp and not use_bf16))
    if use_bf16:
        print("AMP: BFloat16 (Ampere+ detected — no GradScaler needed)")
    elif use_amp:
        print("AMP: Float16 + GradScaler")

    history      = {'train_loss': [], 'val_loss': []}
    history_path = f"checkpoints/history_model_{model_type.lower()}.json"
    best_val_loss = float('inf')
    start_epoch   = 0

    # ── Resume from checkpoint ────────────────────────────────────
    if resume is not None:
        if not os.path.exists(resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        print(f"Resuming from: {resume}")
        ckpt = torch.load(resume, map_location=lambda storage, loc: storage)
        # Strip '_orig_mod.' prefix (backward compat with torch.compile checkpoints)
        clean_sd = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model_state_dict'].items()}

        # Use strict=False to handle phase transitions (e.g. Phase 1→2 enabling
        # coverage adds W_cov layers that don't exist in Phase 1 checkpoints).
        # Missing keys are freshly initialized by the model constructor.
        incompatible = model.load_state_dict(clean_sd, strict=False)
        if incompatible.missing_keys:
            print(f"  [INFO] Newly initialized keys (not in checkpoint): "
                  f"{incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"  [WARN] Unexpected keys in checkpoint (ignored): "
                  f"{incompatible.unexpected_keys}")

        # Optimizer/scheduler state can only be restored when the model
        # architecture AND optimizer layout are unchanged.  Skip restore when:
        #  1. Param group count changed (e.g. frozen → unfreeze adds a group)
        #  2. Model gained new parameters (e.g. Phase 1→2 adds W_cov for coverage)
        # In both cases, start with a fresh optimizer + the new LR from CLI args.
        saved_groups = len(ckpt['optimizer_state_dict']['param_groups'])
        current_groups = len(optimizer.param_groups)
        model_changed = bool(incompatible.missing_keys or incompatible.unexpected_keys)
        if saved_groups == current_groups and not model_changed:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'warmup_scheduler_state_dict' in ckpt:
                warmup_scheduler.load_state_dict(ckpt['warmup_scheduler_state_dict'])
                if 'cosine_scheduler_state_dict' in ckpt:
                    cosine_scheduler.load_state_dict(ckpt['cosine_scheduler_state_dict'])
            scaler.load_state_dict(ckpt['scaler_state_dict'])
            print("  Optimizer & scheduler state restored.")
        else:
            reason = []
            if saved_groups != current_groups:
                reason.append(f"param groups {saved_groups} → {current_groups}")
            if model_changed:
                reason.append(f"model architecture changed ({len(incompatible.missing_keys)} new keys)")
            print(f"  Optimizer state skipped ({', '.join(reason)}) "
                  f"— using fresh optimizer with current LR settings.")

        start_epoch   = ckpt['epoch']          # last completed epoch
        best_val_loss = ckpt['best_val_loss']
        history       = ckpt.get('history', history)
        print(f"  Resumed at epoch {start_epoch} | best_val_loss: {best_val_loss:.4f}")

    print(f"Model: {model_type} | Device: {DEVICE} | Dropout: {dropout}")
    if use_coverage and model_type in ('C', 'D', 'E'):
        print(f"Coverage Mechanism: ON | λ = {coverage_lambda}")

    if use_scst:
        print(f"SCST (Tier 8)     : ON | λ_scst={scst_lambda} (mixed CE + REINFORCE)")

    # Tier 6: CSS augmentor (only for C/D/E)
    css_augmentor = None
    if use_css and model_type in ('C', 'D', 'E'):
        css_augmentor = CSSAugmentor(
            mask_token_id=3,   # <unk> id in our vocab
            mask_ratio=0.3,
            region_mask_ratio=0.3,
        )
        print(f"CSS Augmentation : ON | λ={css_lambda} | margin={css_margin}")
    if accum_steps > 1:
        print(f"Gradient Accum   : {accum_steps} steps (effective batch = {batch_size * accum_steps})")
    if augment:
        print("Data augmentation: ON (RandomHorizontalFlip + ColorJitter)")
    if early_stopping_patience > 0:
        print(f"Early stopping   : patience={early_stopping_patience}")
    es_counter = 0  # early stopping counter
    if scheduled_sampling:
        print(f"Scheduled Sampling: ON | k={ss_k} (epsilon decays from "
              f"{ss_k/(ss_k+math.exp(0/ss_k)):.2f} to "
              f"{ss_k/(ss_k+math.exp((start_epoch+epochs-1)/ss_k)):.2f} over training)")

    # ── torch.compile — requires python3.12-dev (for Triton) ──────────
    # NOTE: Models C/D use a step-by-step attention loop with mutable coverage
    # state. torch.compile adds overhead from graph breaks at each step and
    # provides minimal speedup. Use --no_compile to disable.
    if torch.cuda.is_available() and not no_compile:
        try:
            model = torch.compile(model, mode='default', dynamic=True)
            print("torch.compile    : ON  (default | dynamic shapes)")
        except Exception as e:
            print(f"torch.compile    : skipped — {e}")
    elif no_compile:
        print("torch.compile    : OFF (--no_compile flag)")

    for epoch in tqdm.tqdm(range(start_epoch, start_epoch + epochs)):
        # ── Train ────────────────────────────────────────────────
        model.train()
        total_loss = 0
        optimizer.zero_grad()  # zero once before accumulation loop

        for step_idx, (imgs, questions, answer) in enumerate(train_loader):
            imgs      = imgs.to(DEVICE)
            questions = questions.to(DEVICE)
            answer    = answer.to(DEVICE)

            # Teacher forcing: input [<start>, w1, ..., wn], target [w1, ..., <end>]
            decoder_input  = answer[:, :-1]
            decoder_target = answer[:, 1:]

            with autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                if scheduled_sampling:
                    # Inverse-sigmoid epsilon decay: higher k = slower decay
                    # epoch 0 -> epsilon~1 (mostly GT), last epoch -> epsilon lowers
                    epsilon = ss_k / (ss_k + math.exp(epoch / ss_k))
                    logits  = ss_forward(model, model_type, imgs, questions,
                                         decoder_input, epsilon)
                    coverage_loss = torch.tensor(0.0, device=DEVICE)
                else:
                    result = model(imgs, questions, decoder_input)
                    # Models C/D return (logits, coverage_loss), A/B return logits
                    if isinstance(result, tuple):
                        logits, coverage_loss = result
                    else:
                        logits = result
                        coverage_loss = torch.tensor(0.0, device=DEVICE)

                vocab_size = logits.size(-1)
                ce_loss = criterion(
                    logits.view(-1, vocab_size),
                    decoder_target.contiguous().view(-1)
                )
                # Total loss = CE + λ_cov * coverage + λ_css * contrastive + λ_scst * REINFORCE
                loss = ce_loss + coverage_lambda * coverage_loss
                if css_augmentor is not None:
                    css_loss = css_forward(model, model_type, imgs, questions, css_augmentor)
                    loss = loss + css_lambda * css_loss

            # SCST runs outside AMP context (BLEU computation is not differentiable)
            if use_scst:
                # Decode target tokens → text for BLEU reward
                target_texts = []
                for row in decoder_target:
                    words = []
                    for tid in row.tolist():
                        if tid == 0 or tid == 2:  # pad or end
                            break
                        w = vocab_a.idx2word.get(tid, '<unk>') if hasattr(vocab_a, 'idx2word') \
                            else vocab_a.idx_to_word.get(tid, '<unk>')
                        words.append(w)
                    target_texts.append(' '.join(words))
                with autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                    rl_loss = scst_step(model, model_type, imgs, questions,
                                        target_texts, vocab_a, device=DEVICE)
                    loss = loss + scst_lambda * rl_loss

            # Scale loss for gradient accumulation
            loss = loss / accum_steps
            scaler.scale(loss).backward()

            # Step optimizer every accum_steps mini-batches (or at the last batch)
            if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps  # undo the /accum_steps for logging

        avg_train_loss = total_loss / len(train_loader)

        # ── Validation ───────────────────────────────────────────
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, questions, answer in val_loader:
                imgs      = imgs.to(DEVICE)
                questions = questions.to(DEVICE)
                answer    = answer.to(DEVICE)

                decoder_input  = answer[:, :-1]
                decoder_target = answer[:, 1:]

                with autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                    result     = model(imgs, questions, decoder_input)
                    if isinstance(result, tuple):
                        logits, _ = result
                    else:
                        logits = result
                    vocab_size = logits.size(-1)
                    loss = criterion(
                        logits.view(-1, vocab_size),
                        decoder_target.contiguous().view(-1)
                    )
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        current_lr   = optimizer.param_groups[0]['lr']
        if scheduled_sampling:
            eps = ss_k / (ss_k + math.exp(epoch / ss_k))
            print(f"Epoch {epoch+1}/{start_epoch + epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.2e} | SS ϵ: {eps:.3f}")
        else:
            print(f"Epoch {epoch+1}/{start_epoch + epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.2e}")

        # Step LR scheduler: warmup for first warmup_epochs, then cosine annealing
        if epoch < start_epoch + warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        # Save history after every epoch (safe against session interruption)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        # ── Checkpoint saving (storage-efficient) ─────────────────
        # comparison_epochs: epochs where we run compare.py (end of each phase)
        # Only keep per-epoch checkpoints at these milestones.
        # resume + best checkpoints are always overwritten (1 file each).
        comparison_epochs = {10, 15, 20, 25, 30}
        current_epoch = epoch + 1

        # Always save resume checkpoint (overwritten each epoch — 1 file per model)
        # Strip '_orig_mod.' prefix from torch.compile so checkpoints work everywhere
        clean_sd = {k.replace('_orig_mod.', ''): v for k, v in model.state_dict().items()}
        resume_ckpt = {
            'epoch':                current_epoch,
            'model_state_dict':     clean_sd,
            'optimizer_state_dict': optimizer.state_dict(),
            'warmup_scheduler_state_dict':  warmup_scheduler.state_dict(),
            'cosine_scheduler_state_dict':  cosine_scheduler.state_dict(),
            'scaler_state_dict':    scaler.state_dict(),
            'best_val_loss':        best_val_loss,
            'history':              history,
        }
        torch.save(resume_ckpt, f"checkpoints/model_{model_type.lower()}_resume.pth")

        # Save per-epoch checkpoint ONLY at comparison milestones
        if current_epoch in comparison_epochs:
            torch.save(clean_sd, f"checkpoints/model_{model_type.lower()}_epoch{current_epoch}.pth")
            print(f"  Saved milestone checkpoint: epoch {current_epoch}")

        # Save best checkpoint separately (overwritten — 1 file per model)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            es_counter = 0
            torch.save(clean_sd, f"checkpoints/model_{model_type.lower()}_best.pth")
            print(f"  -> New best val loss: {best_val_loss:.4f}. Saved best checkpoint.")
        else:
            # BUG FIX: Don't count early stopping during warmup epochs.
            # The LR ramp-up (lr/10 → lr) naturally destabilizes complex models
            # (C/D have 5× more LSTM params), causing val loss to increase.
            # Counting warmup epochs as "no improvement" kills C/D at epoch 4.
            if epoch >= start_epoch + warmup_epochs:
                es_counter += 1
            else:
                print(f"  Val loss did not improve (warmup — not counting for early stop)")
            if early_stopping_patience > 0 and es_counter > 0:
                print(f"  Val loss did not improve ({es_counter}/{early_stopping_patience})")
                if es_counter >= early_stopping_patience:
                    print(f"  Early stopping triggered after {epoch+1} epochs.")
                    # Save best model as the next unsaved milestone so
                    # compare.py can still find a checkpoint for this phase.
                    target_epoch = start_epoch + epochs
                    milestone_path = f"checkpoints/model_{model_type.lower()}_epoch{target_epoch}.pth"
                    best_path = f"checkpoints/model_{model_type.lower()}_best.pth"
                    if not os.path.exists(milestone_path) and os.path.exists(best_path):
                        import shutil
                        shutil.copy2(best_path, milestone_path)
                        print(f"  Copied best checkpoint → {milestone_path} (for compare.py)")
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VQA model.")
    parser.add_argument('--model',      type=str,   default='A', choices=['A', 'B', 'C', 'D', 'E'],
                        help='Model architecture (default: A)')
    parser.add_argument('--epochs',     type=int,   default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--lr',         type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--batch_size', type=int,   default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--resume',     type=str,   default=None,
                        help='Path to a resume checkpoint (model_X_resume.pth) to continue training')
    parser.add_argument('--scheduled_sampling', action='store_true',
                        help='Enable Scheduled Sampling to reduce exposure bias')
    parser.add_argument('--ss_k',       type=float, default=5.0,
                        help='Inverse-sigmoid decay speed for Scheduled Sampling (default: 5.0)')
    parser.add_argument('--finetune_cnn', action='store_true',
                        help='Unfreeze ResNet layer3+layer4 for fine-tuning (models B/D only)')
    parser.add_argument('--cnn_lr_factor', type=float, default=0.1,
                        help='LR multiplier for backbone params when fine-tuning (default: 0.1)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader worker processes (default: 4, recommend 8 for A100)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='L2 regularization weight decay (default: 1e-5)')
    parser.add_argument('--early_stopping', type=int, default=0,
                        help='Stop if val loss does not improve for N epochs (0=disabled)')
    parser.add_argument('--augment', action='store_true',
                        help='Enable image data augmentation (RandomHorizontalFlip + ColorJitter)')
    parser.add_argument('--glove', action='store_true',
                        help='Use GloVe 300d pretrained embeddings for Q-encoder and decoder')
    parser.add_argument('--glove_dim', type=int, default=300,
                        help='GloVe embedding dimension: 50, 100, 200, or 300 (default: 300)')
    parser.add_argument('--coverage', action='store_true',
                        help='Enable coverage mechanism to reduce repetition (models C/D only)')
    parser.add_argument('--coverage_lambda', type=float, default=1.0,
                        help='Weight for coverage loss term (default: 1.0)')
    # Tier 1: LSTM structural fortification
    parser.add_argument('--layer_norm', action='store_true',
                        help='Tier 1A+1C: LayerNorm inside LSTM gates + Highway connections')
    parser.add_argument('--dropconnect', action='store_true',
                        help='Tier 1B: DropConnect on hidden-to-hidden weights (AWD-LSTM)')
    # Tier 2: Dense Co-Attention
    parser.add_argument('--dcan', action='store_true',
                        help='Tier 2: Dense Co-Attention replacing BahdanauAttention (C/D/E only)')
    # Tier 4: MUTAN Tucker Fusion
    parser.add_argument('--use_mutan', action='store_true',
                        help='Tier 4: MUTAN Tucker Fusion instead of GatedFusion (Model E only)')
    # Tier 5: Pointer-Generator Network
    parser.add_argument('--pgn', action='store_true',
                        help='Tier 5: Pointer-Generator Network — copy from question tokens (C/D/E)')
    # Tier 8: SCST Reinforcement Learning
    parser.add_argument('--scst', action='store_true',
                        help='Tier 8: SCST RL — mixed CE + REINFORCE (use after Phase 3)')
    parser.add_argument('--scst_lambda', type=float, default=0.5,
                        help='Weight for SCST RL loss term (default: 0.5)')
    # Tier 7: Deep BiLSTM + Char-CNN question encoder
    parser.add_argument('--q_highway', action='store_true',
                        help='Tier 7B: Highway connections between BiLSTM layers in question encoder')
    parser.add_argument('--char_cnn', action='store_true',
                        help='Tier 7C: Char-CNN embedding prepended to word embeddings')
    # Tier 6: CSS Counterfactual Augmentation
    parser.add_argument('--css', action='store_true',
                        help='Tier 6: CSS counterfactual augmentation (visual+linguistic masking)')
    parser.add_argument('--css_lambda', type=float, default=0.5,
                        help='Weight for CSS contrastive loss (default: 0.5)')
    parser.add_argument('--css_margin', type=float, default=1.0,
                        help='Margin for CSS hinge contrastive loss (default: 1.0)')
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='Gradient accumulation steps (effective batch = batch_size × accum_steps)')
    parser.add_argument('--warmup_epochs', type=int, default=3,
                        help='LR warmup epochs (lr/10 → lr). Set 0 to disable for resume/Phase 2+ (default: 3)')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Limit number of training samples (None=use all). For quick tests use 1000-5000')
    parser.add_argument('--max_val_samples', type=int, default=None,
                        help='Limit number of validation samples (None=use all). For quick tests use 500-2000')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate for Q-encoder and decoder embeddings/LSTM (default: 0.5, recommend 0.3)')
    parser.add_argument('--no_compile', action='store_true',
                        help='Disable torch.compile (recommended for Models C/D with attention loop)')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='Maximum gradient norm for clipping (default: 5.0)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor for CrossEntropyLoss (default: 0.1)')
    args = parser.parse_args()
    train(model_type=args.model, epochs=args.epochs, lr=args.lr,
          batch_size=args.batch_size, resume=args.resume,
          scheduled_sampling=args.scheduled_sampling, ss_k=args.ss_k,
          finetune_cnn=args.finetune_cnn, cnn_lr_factor=args.cnn_lr_factor,
          num_workers=args.num_workers, weight_decay=args.weight_decay,
          early_stopping_patience=args.early_stopping, augment=args.augment,
          use_glove=args.glove, glove_dim=args.glove_dim,
          use_coverage=args.coverage, coverage_lambda=args.coverage_lambda,
          accum_steps=args.accum_steps, warmup_epochs=args.warmup_epochs,
          max_train_samples=args.max_train_samples, max_val_samples=args.max_val_samples,
          dropout=args.dropout, no_compile=args.no_compile,
          grad_clip=args.grad_clip, label_smoothing=args.label_smoothing,
          use_mutan=args.use_mutan, use_pgn=args.pgn,
          use_css=args.css, css_lambda=args.css_lambda, css_margin=args.css_margin,
          use_q_highway=args.q_highway, use_char_cnn=args.char_cnn,
          use_scst=args.scst, scst_lambda=args.scst_lambda)
        
        
        
    
