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
from dataset import VQAEDataset, VQADataset, vqa_collate_fn, build_mixed_sampler
from models.vqa_models import VQAModelA, VQAModelB, VQAModelC, VQAModelD, VQAModelE, VQAModelF
from glove_utils import build_glove_matrix
from vocab import Vocabulary
from training.css_augment import CSSAugmentor, css_contrastive_loss
from training.scst import scst_step
from training.curriculum import CurriculumSampler, compute_complexity_scores
from training.losses import SequenceFocalLoss

# ── wandb (optional — gracefully disabled if not installed / not requested) ──
try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False




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


def get_model(model_type, vocab_size,
              pretrained_q_emb=None, pretrained_a_emb=None,
              use_coverage=False, dropout=0.5,
              use_layer_norm=False, use_dropconnect=False,
              use_dcan=False, use_mutan=False, use_pgn=False,
              use_q_highway=False, use_char_cnn=False):
    """Factory function: return the model corresponding to model_type."""
    kw = dict(pretrained_q_emb=pretrained_q_emb, pretrained_a_emb=pretrained_a_emb,
              dropout=dropout, use_q_highway=use_q_highway, use_char_cnn=use_char_cnn)
    if model_type == 'A':
        return VQAModelA(vocab_size=vocab_size, answer_vocab_size=vocab_size, **kw)
    elif model_type == 'B':
        return VQAModelB(vocab_size=vocab_size, answer_vocab_size=vocab_size, **kw)
    elif model_type == 'C':
        return VQAModelC(vocab_size=vocab_size, answer_vocab_size=vocab_size,
                         use_coverage=use_coverage,
                         use_layer_norm=use_layer_norm,
                         use_dropconnect=use_dropconnect,
                         use_dcan=use_dcan, use_pgn=use_pgn, **kw)
    elif model_type == 'D':
        return VQAModelD(vocab_size=vocab_size, answer_vocab_size=vocab_size,
                         use_coverage=use_coverage,
                         use_layer_norm=use_layer_norm,
                         use_dropconnect=use_dropconnect,
                         use_dcan=use_dcan, use_pgn=use_pgn, **kw)
    elif model_type == 'E':
        return VQAModelE(vocab_size=vocab_size, answer_vocab_size=vocab_size,
                         use_coverage=use_coverage,
                         use_layer_norm=use_layer_norm,
                         use_dropconnect=use_dropconnect,
                         use_dcan=use_dcan,
                         use_mutan=use_mutan, use_pgn=use_pgn, **kw)
    elif model_type == 'F':
        return VQAModelF(vocab_size=vocab_size, answer_vocab_size=vocab_size,
                         use_coverage=use_coverage,
                         use_layer_norm=use_layer_norm,
                         use_dropconnect=use_dropconnect,
                         use_mutan=use_mutan, use_pgn=use_pgn,
                         use_q_highway=use_q_highway,
                         use_char_cnn=use_char_cnn, **kw)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from A, B, C, D, E, F.")


def ss_forward(model, model_type, imgs, questions, decoder_input, epsilon,
               img_mask=None):
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
    if model_type in ('C', 'D', 'E', 'F'):
        img_features = F.normalize(model.i_encoder(imgs), p=2, dim=-1)  # (B, S, H)
        q_feat, q_hidden = model.q_encoder(questions)                    # (B, H), (B, qlen, H)
        # Model F: masked mean to exclude padding zeros (imgs is feat tensor here)
        if model_type == 'F' and img_mask is not None:
            valid_counts = img_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
            img_mean = (img_features * img_mask.unsqueeze(-1).float()).sum(dim=1) \
                       / valid_counts
        else:
            img_mean = img_features.mean(dim=1)
        # Models E/F: MUTAN expects (q, v) order; C/D GatedFusion: (img_mean, q_feat)
        if model_type in ('E', 'F'):
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

    # Coverage vector for models C/D/E/F
    coverage = None
    if model_type in ('C', 'D', 'E', 'F') and model.decoder.use_coverage:
        coverage = imgs.new_zeros(B, img_features.size(1))  # (B, 49/k)

    # ── Step-by-step decoding ────────────────────────────────────────
    current_token = decoder_input[:, 0]   # (B,) — first token is always <start>
    logits_list   = []

    for t in range(max_len):
        tok = current_token.unsqueeze(1)  # (B, 1)

        if model_type in ('C', 'D', 'E', 'F'):
            logit, hidden, _, coverage = model.decoder.decode_step(
                tok, hidden, img_features, q_hidden, coverage,
                q_token_ids=questions, img_mask=img_mask,
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


def css_forward(model, model_type, imgs, questions, augmentor, img_mask=None):
    """
    Tier-6: Compute CSS contrastive loss without running the full decoder.
    Encodes both real and counterfactual samples, compares fused representations.

    img_mask : (B, max_k) bool or None — valid region mask for Model F.
               Used to compute masked mean instead of plain mean, preventing
               padding zeros from diluting the global image representation.

    Returns: scalar contrastive loss tensor
    """
    def _masked_mean(feats, mask):
        """Mean over valid regions only; falls back to plain mean if mask is None."""
        if mask is None:
            return feats.mean(dim=1)
        valid = mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        return (feats * mask.unsqueeze(-1).float()).sum(dim=1) / valid

    # Encode real batch (reuse encoder already warmed up)
    with torch.no_grad():
        img_features = F.normalize(model.i_encoder(imgs), p=2, dim=-1)  # (B, S, H)
        img_mean     = _masked_mean(img_features, img_mask)              # (B, H)
        q_feature, _ = model.q_encoder(questions)                        # (B, H)

    # Generate counterfactuals (no gradient — pure augmentation)
    cf_img_feats, cf_questions = augmentor(questions, img_features.detach())

    # Compute fused real representation (WITH gradient for the fusion/encoders)
    img_features_grad = F.normalize(model.i_encoder(imgs), p=2, dim=-1)
    img_mean_grad     = _masked_mean(img_features_grad, img_mask)
    q_feature_grad, _ = model.q_encoder(questions)
    if model_type in ('E', 'F'):
        f_real = model.fusion(q_feature_grad, img_mean_grad)    # MUTAN: q first
    else:
        f_real = model.fusion(img_mean_grad, q_feature_grad)    # GatedFusion: img first

    # Visual CF: zeroed image regions + original questions (no gradient through cf_img)
    # Note: CSS zeros real regions, not padding — img_mask still valid for cf_img_feats
    cf_img_mean = _masked_mean(cf_img_feats, img_mask)          # (B, H)  — already detached
    if model_type in ('E', 'F'):
        f_cf_visual = model.fusion(q_feature_grad, cf_img_mean)
    else:
        f_cf_visual = model.fusion(cf_img_mean, q_feature_grad)

    # Linguistic CF: original image + masked questions (re-encode cf_questions)
    cf_q_feat, _ = model.q_encoder(cf_questions)
    if model_type in ('E', 'F'):
        f_cf_ling = model.fusion(cf_q_feat, img_mean_grad)
    else:
        f_cf_ling = model.fusion(img_mean_grad, cf_q_feat)

    return css_contrastive_loss(f_real, f_cf_visual, f_cf_ling)


# ── Training set (train2014) ─────────────────────────────────────
TRAIN_IMAGE_DIR  = "data/images/train2014"
TRAIN_VQA_E_JSON = "data/annotations/vqa_e/VQA-E_train_set.json"

# ── Validation set (val2014) ──────────────────────────────────────
VAL_IMAGE_DIR  = "data/images/val2014"
VAL_VQA_E_JSON = "data/annotations/vqa_e/VQA-E_val_set.json"

# ── VQA v2.0 paths (Tier D2 mixed pretraining) ────────────────────
VQA_V2_TRAIN_Q_JSON   = "data/annotations/vqa_v2/v2_OpenEnded_mscoco_train2014_questions.json"
VQA_V2_TRAIN_ANN_JSON = "data/annotations/vqa_v2/v2_mscoco_train2014_annotations.json"
# VQA v2.0 images are the same train2014 directory — no extra download

VOCAB_JOINT_PATH  = "data/processed/vocab_joint.json"

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
          use_scst=False, scst_lambda=0.5,
          scst_bleu_weight=0.5, scst_meteor_weight=0.5, scst_length_bonus=0.0,
          use_wandb=False, wandb_project='vqa-e', wandb_run_name=None,
          wandb_tags=None, phase=None,
          use_curriculum=False,
          use_focal_loss=False, focal_gamma=2.0,
          mix_vqa=False, mix_vqa_fraction=0.7,
          reset_best_val_loss=False,
          butd_feat_dir=None):
    os.makedirs("checkpoints", exist_ok=True)

    vocab = Vocabulary(); vocab.load(VOCAB_JOINT_PATH)

    # ── Weights & Biases ──────────────────────────────────────────────────
    _wb = None
    if use_wandb and _WANDB_AVAILABLE:
        _run_name = wandb_run_name or f"model_{model_type.lower()}_phase{phase or '?'}"
        _config = dict(
            model=model_type, phase=phase, epochs=epochs, lr=lr,
            batch_size=batch_size, accum_steps=accum_steps,
            effective_batch=batch_size * accum_steps,
            dropout=dropout, weight_decay=weight_decay,
            grad_clip=grad_clip, label_smoothing=label_smoothing,
            warmup_epochs=warmup_epochs,
            use_glove=use_glove, glove_dim=glove_dim,
            layer_norm=getattr(args, 'layer_norm', False),
            dropconnect=getattr(args, 'dropconnect', False),
            dcan=getattr(args, 'dcan', False),
            use_mutan=use_mutan, use_pgn=use_pgn,
            use_coverage=use_coverage, coverage_lambda=coverage_lambda,
            finetune_cnn=finetune_cnn, cnn_lr_factor=cnn_lr_factor,
            scheduled_sampling=scheduled_sampling, ss_k=ss_k,
            use_css=use_css, css_lambda=css_lambda,
            use_scst=use_scst, scst_lambda=scst_lambda,
            scst_bleu_weight=scst_bleu_weight, scst_meteor_weight=scst_meteor_weight,
            scst_length_bonus=scst_length_bonus,
            use_q_highway=use_q_highway, use_char_cnn=use_char_cnn,
            augment=augment,
            use_curriculum=use_curriculum,
            use_focal_loss=use_focal_loss, focal_gamma=focal_gamma,
            mix_vqa=mix_vqa, mix_vqa_fraction=mix_vqa_fraction,
        )
        _wb = _wandb.init(
            project=wandb_project,
            name=_run_name,
            config=_config,
            tags=wandb_tags or [],
            resume='allow',
        )
        print(f"W&B run          : {_wb.url}")
    elif use_wandb and not _WANDB_AVAILABLE:
        print("W&B requested but wandb not installed — pip install wandb")

    vqae_train_dataset = VQAEDataset(
        image_dir=TRAIN_IMAGE_DIR,
        vqa_e_json_path=TRAIN_VQA_E_JSON,
        vocab=vocab,
        split='train2014',
        max_samples=max_train_samples or MAX_TRAIN_SAMPLES,
        augment=augment
    )

    val_dataset = VQAEDataset(
        image_dir=VAL_IMAGE_DIR,
        vqa_e_json_path=VAL_VQA_E_JSON,
        vocab=vocab,
        split='val2014',
        max_samples=max_val_samples or MAX_VAL_SAMPLES
    )

    # Tier 3B: Model F uses pre-extracted BUTD features instead of raw images
    if model_type == 'F':
        if butd_feat_dir is None:
            raise ValueError("--model F requires --butd_feat_dir (run extract_butd_features.py first)")
        from dataset import BUTDDataset, butd_collate_fn
        vqae_train_dataset = BUTDDataset(
            feat_dir=butd_feat_dir,
            vqa_e_json_path=TRAIN_VQA_E_JSON,
            vocab=vocab,
            split='train2014', max_samples=max_train_samples or MAX_TRAIN_SAMPLES,
        )
        val_dataset = BUTDDataset(
            feat_dir=butd_feat_dir,
            vqa_e_json_path=VAL_VQA_E_JSON,
            vocab=vocab,
            split='val2014', max_samples=max_val_samples or MAX_VAL_SAMPLES,
        )
        _collate = butd_collate_fn
        print(f"BUTD Features    : {butd_feat_dir}")
    else:
        _collate = vqa_collate_fn

    pin_memory = torch.cuda.is_available()

    # ── Tier D2: Mixed-Ratio Pretraining ────────────────────────────────────
    # Phase 1 only: mix VQA v2.0 (short answers) with VQA-E (long explanations)
    # to inject vocabulary breadth without inducing premature <end> length bias.
    # --mix_vqa is intended for Phase 1 only; later phases use pure VQA-E.
    mixed_sampler   = None
    train_dataset   = vqae_train_dataset   # default: pure VQA-E

    if mix_vqa:
        vqa_v2_files_ok = (os.path.exists(VQA_V2_TRAIN_Q_JSON) and
                           os.path.exists(VQA_V2_TRAIN_ANN_JSON))
        if not vqa_v2_files_ok:
            print(f"[WARN] --mix_vqa requested but VQA v2.0 JSON files not found:")
            print(f"  {VQA_V2_TRAIN_Q_JSON}")
            print(f"  {VQA_V2_TRAIN_ANN_JSON}")
            print("  Falling back to pure VQA-E training.")
        else:
            vqa_v2_dataset = VQADataset(
                image_dir=TRAIN_IMAGE_DIR,
                question_json_path=VQA_V2_TRAIN_Q_JSON,
                annotations_json_path=VQA_V2_TRAIN_ANN_JSON,
                vocab=vocab,
                split='train2014',
                max_samples=max_train_samples or MAX_TRAIN_SAMPLES,
                augment=augment,
            )
            train_dataset, mixed_sampler = build_mixed_sampler(
                vqa_v2_dataset, vqae_train_dataset,
                vqa_fraction=mix_vqa_fraction,
            )
            print(f"Mixed Pretraining: ON (D2) — "
                  f"VQA v2.0 {mix_vqa_fraction:.0%} / "
                  f"VQA-E {1-mix_vqa_fraction:.0%}")
            print(f"  VQA v2.0: {len(vqa_v2_dataset):,} | "
                  f"VQA-E: {len(vqae_train_dataset):,} | "
                  f"ConcatDataset: {len(train_dataset):,}")

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # ── Tier D4: Curriculum sampler — sort by question-type complexity ───────
    # Mutually exclusive with mixed_sampler (cannot use two samplers simultaneously).
    # If both are requested, mixed_sampler takes precedence (Phase 1 is the priority).
    curriculum_sampler = None
    if use_curriculum and mixed_sampler is None:
        # compute_question_type_scores works on the VQA-E annotations
        scores = compute_complexity_scores(vqae_train_dataset.annotations)
        curriculum_sampler = CurriculumSampler(scores, epoch=0, total_epochs=epochs)
        print(f"Curriculum       : ON (D4 — question-type stages, "
              f"{len(scores)} samples)")
    elif use_curriculum and mixed_sampler is not None:
        print("Curriculum       : SKIPPED (incompatible with --mix_vqa in same phase)")

    # Choose the active sampler (at most one can be active)
    active_sampler = mixed_sampler or curriculum_sampler

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(active_sampler is None),  # mutually exclusive with sampler
        sampler=active_sampler,
        collate_fn=_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
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
        matrix, cov = build_glove_matrix(vocab, glove_dim=glove_dim)
        pretrained_q_emb = _t.tensor(matrix)
        pretrained_a_emb = _t.tensor(matrix)
        print(f"  Joint Vocab coverage: {cov:.1%}")

    model     = get_model(model_type, len(vocab),
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
        model.q_encoder.char_cnn.build_char_table(vocab)
        print("Char-CNN         : built char table from vocab")
    # ── Loss function ──────────────────────────────────────────────────────────
    # PGN outputs log-probabilities via PointerGeneratorHead.blend → NLLLoss.
    # Non-PGN: SequenceFocalLoss (Tier D3, when --focal) or plain CrossEntropyLoss.
    #
    # Why NOT CrossEntropyLoss(weight=class_weights) for autoregressive models:
    #   Static class weights penalize every occurrence of a token id regardless
    #   of position.  Weight for "because" (common) would be ~0.1, destroying
    #   the structural hinge of VQA-E explanations at every decode step.
    #   SequenceFocalLoss computes per-position p_t = exp(-ce_t) and suppresses
    #   easy/common tokens naturally — no explicit weight tensor needed.
    if use_pgn:
        criterion = nn.NLLLoss(ignore_index=0)
        if use_focal_loss:
            print("Focal Loss       : skipped (incompatible with PGN NLLLoss)")
    elif use_focal_loss:
        criterion = SequenceFocalLoss(gamma=focal_gamma, pad_idx=0,
                                      label_smoothing=label_smoothing)
        print(f"Loss             : SequenceFocalLoss (Tier D3) γ={focal_gamma}")
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing)

    # ── Optimizer — differential LR when fine-tuning the CNN backbone ──────────
    # Models A/C use scratch CNN → finetune_cnn flag has no effect on them.
    # Models B/D use frozen ResNet → finetune_cnn selectively unfreezes
    # layer3 + layer4 with a smaller LR (cnn_lr_factor × base_lr) to avoid
    # catastrophic forgetting of pretrained ImageNet knowledge.
    if finetune_cnn and model_type in ('B', 'D', 'E', 'F'):
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
        #  1. No optimizer_state_dict in checkpoint (milestone/best checkpoints
        #     saved without optimizer state to save disk space)
        #  2. Param group count changed (e.g. frozen → unfreeze adds a group)
        #  3. Model gained new parameters (e.g. Phase 1→2 adds W_cov for coverage)
        # In all cases, start with a fresh optimizer + the new LR from CLI args.
        model_changed = bool(incompatible.missing_keys or incompatible.unexpected_keys)
        if 'optimizer_state_dict' not in ckpt:
            print("  Optimizer state skipped (checkpoint has no optimizer_state_dict) "
                  "— using fresh optimizer with current LR settings.")
        else:
            saved_groups = len(ckpt['optimizer_state_dict']['param_groups'])
            current_groups = len(optimizer.param_groups)
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
        if reset_best_val_loss:
            best_val_loss = float('inf')
            print("  best_val_loss reset to inf (new phase — early stopping starts fresh)")
        history       = ckpt.get('history', history)
        print(f"  Resumed at epoch {start_epoch} | best_val_loss: {best_val_loss:.4f}")

    print(f"Model: {model_type} | Device: {DEVICE} | Dropout: {dropout}")
    if use_coverage and model_type in ('C', 'D', 'E', 'F'):
        print(f"Coverage Mechanism: ON | λ = {coverage_lambda}")

    if use_scst:
        from training.scst import _METEOR_AVAILABLE
        print(f"SCST (Tier 8)     : ON | λ_scst={scst_lambda} (mixed CE + REINFORCE)")
        print(f"  Reward          : {scst_bleu_weight:.2f}×BLEU-4 + {scst_meteor_weight:.2f}×METEOR"
              f" + {scst_length_bonus:.2f}×LengthBonus")
        if scst_meteor_weight > 0 and not _METEOR_AVAILABLE:
            print("  WARNING: METEOR requested but nltk.translate.meteor_score not found.")
            print("           Falling back to unigram F1 for METEOR component.")
            print("           Install: pip install nltk && python -m nltk.downloader wordnet")

    # Tier 6: CSS augmentor (only for C/D/E)
    css_augmentor = None
    if use_css and model_type in ('C', 'D', 'E', 'F'):
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
              f"{ss_k/(ss_k+math.exp((epochs-1)/ss_k)):.2f} over {epochs} epochs)")

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

    global_step   = 0    # optimizer-step counter (cumulative across all epochs this phase)
    _LOG_EVERY_N  = 50   # log step-level W&B metrics every N optimizer steps

    for epoch in tqdm.tqdm(range(start_epoch, start_epoch + epochs)):
        # Tier D4: advance curriculum pacing each epoch
        # (curriculum_sampler is None when mix_vqa is active — they're mutually exclusive)
        if curriculum_sampler is not None:
            curriculum_sampler.set_epoch(epoch)

        # ── Train ────────────────────────────────────────────────
        model.train()
        total_loss = 0
        # Per-epoch component accumulators (for W&B epoch-level breakdown)
        _ep_ce = 0.0; _ep_cov = 0.0; _ep_css = 0.0; _ep_rl = 0.0
        _ep_grad_norm = 0.0; _ep_opt_steps = 0
        _ep_scst = {'reward_greedy': 0.0, 'reward_sample': 0.0,
                    'advantage_mean': 0.0, 'advantage_std': 0.0, 'n': 0}
        optimizer.zero_grad()  # zero once before accumulation loop

        for step_idx, batch in enumerate(train_loader):
            # Model F returns a 4-tuple (feats, questions, answers, img_mask);
            # all other models return a 3-tuple.
            if model_type == 'F':
                imgs, questions, answer, img_mask = batch
                img_mask = img_mask.to(DEVICE)
            else:
                imgs, questions, answer = batch
                img_mask = None

            imgs      = imgs.to(DEVICE)
            questions = questions.to(DEVICE)
            answer    = answer.to(DEVICE)

            # Teacher forcing: input [<start>, w1, ..., wn], target [w1, ..., <end>]
            decoder_input  = answer[:, :-1]
            decoder_target = answer[:, 1:]

            with autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                if scheduled_sampling:
                    # Inverse-sigmoid epsilon decay using RELATIVE epoch (0-indexed within phase).
                    # Must use (epoch - start_epoch), NOT absolute epoch — otherwise
                    # epsilon ≈ 0.001 when resuming at epoch 40+ (exp(8+) is huge).
                    relative_epoch = epoch - start_epoch
                    epsilon = ss_k / (ss_k + math.exp(relative_epoch / ss_k))
                    logits  = ss_forward(model, model_type, imgs, questions,
                                         decoder_input, epsilon, img_mask=img_mask)
                    coverage_loss = torch.tensor(0.0, device=DEVICE)
                else:
                    result = model(imgs, questions, decoder_input, img_mask=img_mask) \
                             if model_type == 'F' \
                             else model(imgs, questions, decoder_input)
                    # Models C/D/E/F return (logits, coverage_loss), A/B return logits
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
                css_loss = torch.tensor(0.0, device=DEVICE)
                if css_augmentor is not None:
                    css_loss = css_forward(model, model_type, imgs, questions,
                                           css_augmentor, img_mask=img_mask)
                    loss = loss + css_lambda * css_loss

            # Backward CE+coverage+CSS immediately — frees the ConvNeXt/MUTAN
            # computation graph before SCST launches its own forward passes.
            # Without this, peak VRAM = CE graph + SCST sampling graph → OOM.
            scaler.scale(loss / accum_steps).backward()
            # Capture component values (after backward, graph freed)
            _batch_ce  = ce_loss.item()
            _batch_cov = coverage_loss.item()
            _batch_css = css_loss.item()

            # SCST runs AFTER CE backward (CE graph is freed → ~40% less peak VRAM)
            rl_loss = None
            _scst_stats = None
            if use_scst:
                # Decode target tokens → text for BLEU reward
                target_texts = []
                for row in decoder_target:
                    words = []
                    for tid in row.tolist():
                        if tid == 0 or tid == 2:  # pad or end
                            break
                        w = vocab.idx2word.get(tid, '<unk>') if hasattr(vocab, 'idx2word') \
                            else vocab.idx_to_word.get(tid, '<unk>')
                        words.append(w)
                    target_texts.append(' '.join(words))
                with autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                    rl_loss, _scst_stats = scst_step(
                        model, model_type, imgs, questions,
                        target_texts, vocab, device=DEVICE,
                        max_len=decoder_input.size(1),
                        bleu_weight=scst_bleu_weight,
                        meteor_weight=scst_meteor_weight,
                        length_bonus_weight=scst_length_bonus,
                        return_stats=True)
                scaler.scale(scst_lambda * rl_loss / accum_steps).backward()

            # Step optimizer every accum_steps mini-batches (or at the last batch)
            if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                _ep_grad_norm  += grad_norm.item()
                _ep_opt_steps  += 1
                _ep_ce  += _batch_ce;  _ep_cov += _batch_cov;  _ep_css += _batch_css
                if rl_loss is not None:
                    _ep_rl += rl_loss.item()
                    if _scst_stats is not None:
                        _ep_scst['reward_greedy']   += _scst_stats['scst/reward_greedy']
                        _ep_scst['reward_sample']   += _scst_stats['scst/reward_sample']
                        _ep_scst['advantage_mean']  += _scst_stats['scst/advantage_mean']
                        _ep_scst['advantage_std']   += _scst_stats['scst/advantage_std']
                        _ep_scst['n'] += 1

                # ── Per-step W&B logging (every _LOG_EVERY_N optimizer steps) ──
                if _wb is not None and global_step % _LOG_EVERY_N == 0:
                    _lr_now = optimizer.param_groups[0]['lr']
                    _step_log = {
                        'step/ce_loss':       _batch_ce,
                        'step/coverage_loss': _batch_cov,
                        'step/css_loss':      _batch_css,
                        'step/grad_norm':     grad_norm.item(),
                        'step/lr':            _lr_now,
                    }
                    if rl_loss is not None:
                        _step_log['step/rl_loss'] = rl_loss.item()
                        if _scst_stats is not None:
                            _step_log.update({k.replace('scst/', 'step/'): v
                                              for k, v in _scst_stats.items()})
                    if finetune_cnn and len(optimizer.param_groups) > 1:
                        _step_log['step/lr_backbone'] = optimizer.param_groups[1]['lr']
                    _wb.log(_step_log, step=global_step)

            total_loss += loss.item() * accum_steps   # undo the /accum_steps for logging
            if rl_loss is not None:
                total_loss += scst_lambda * rl_loss.item() * accum_steps

        avg_train_loss = total_loss / len(train_loader)

        # ── Validation ───────────────────────────────────────────
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_batch in val_loader:
                if model_type == 'F':
                    imgs, questions, answer, img_mask = val_batch
                    img_mask = img_mask.to(DEVICE)
                else:
                    imgs, questions, answer = val_batch
                    img_mask = None

                imgs      = imgs.to(DEVICE)
                questions = questions.to(DEVICE)
                answer    = answer.to(DEVICE)

                decoder_input  = answer[:, :-1]
                decoder_target = answer[:, 1:]

                with autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                    result = model(imgs, questions, decoder_input, img_mask=img_mask) \
                             if model_type == 'F' \
                             else model(imgs, questions, decoder_input)
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
            eps = ss_k / (ss_k + math.exp((epoch - start_epoch) / ss_k))
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

        # ── W&B logging ───────────────────────────────────────────
        if _wb is not None:
            _n = max(_ep_opt_steps, 1)
            log_dict = {
                'train/loss':         avg_train_loss,
                'val/loss':           avg_val_loss,
                # Per-component epoch averages
                'train/ce_loss':      _ep_ce  / _n,
                'train/coverage_loss':_ep_cov / _n,
                'train/css_loss':     _ep_css / _n,
                # Gradient stats
                'train/grad_norm':    _ep_grad_norm / _n,
                # Learning rates
                'lr/decoder':         current_lr,
                # Early stopping state
                'train/patience':     es_counter,
                'train/best_val_loss':best_val_loss,
            }
            # Backbone LR (only when CNN is being fine-tuned)
            if finetune_cnn and len(optimizer.param_groups) > 1:
                log_dict['lr/backbone'] = optimizer.param_groups[1]['lr']
            # Scheduled Sampling epsilon
            if scheduled_sampling:
                log_dict['train/ss_epsilon'] = ss_k / (ss_k + math.exp((epoch - start_epoch) / ss_k))
            # SCST epoch averages
            if use_scst and _ep_scst['n'] > 0:
                _ns = _ep_scst['n']
                log_dict.update({
                    'train/rl_loss':          _ep_rl  / _n,
                    'scst/reward_greedy':     _ep_scst['reward_greedy']  / _ns,
                    'scst/reward_sample':     _ep_scst['reward_sample']  / _ns,
                    'scst/advantage_mean':    _ep_scst['advantage_mean'] / _ns,
                    'scst/advantage_std':     _ep_scst['advantage_std']  / _ns,
                    'scst/reward_delta':      (_ep_scst['reward_sample'] - _ep_scst['reward_greedy']) / _ns,
                })
            # Curriculum stage (1-4: binary → +color/count → +what/where → full)
            if curriculum_sampler is not None:
                _prog = curriculum_sampler.epoch / max(getattr(curriculum_sampler, 'total_epochs', epochs) - 1, 1)
                _stage = 1 if _prog < 0.25 else 2 if _prog < 0.50 else 3 if _prog < 0.75 else 4
                log_dict['data/curriculum_stage'] = _stage
            # GPU memory
            if torch.cuda.is_available():
                log_dict['gpu/memory_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
                log_dict['gpu/memory_reserved_gb']  = torch.cuda.memory_reserved()  / 1e9
            log_dict['epoch'] = epoch + 1   # keep epoch number as a field (not x-axis)
            _wb.log(log_dict, step=global_step)

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
            milestone_ckpt = {
                'epoch':            current_epoch,
                'model_state_dict': clean_sd,
                'val_loss':         avg_val_loss,
                'best_val_loss':    best_val_loss,
                'history':          history,
            }
            torch.save(milestone_ckpt, f"checkpoints/model_{model_type.lower()}_epoch{current_epoch}.pth")
            print(f"  Saved milestone checkpoint: epoch {current_epoch}")

        # Save best checkpoint separately (overwritten — 1 file per model)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            es_counter = 0
            best_ckpt = {
                'epoch':            current_epoch,
                'model_state_dict': clean_sd,
                'val_loss':         avg_val_loss,
                'best_val_loss':    best_val_loss,
                'history':          history,
            }
            torch.save(best_ckpt, f"checkpoints/model_{model_type.lower()}_best.pth")
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

    if _wb is not None:
        _wb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VQA model.")
    parser.add_argument('--model',      type=str,   default='A', choices=['A', 'B', 'C', 'D', 'E', 'F', 'G'],
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
    parser.add_argument('--scst_bleu_weight', type=float, default=0.5,
                        help='BLEU-4 coefficient in composite SCST reward (default: 0.5)')
    parser.add_argument('--scst_meteor_weight', type=float, default=0.5,
                        help='METEOR coefficient in composite SCST reward (default: 0.5)')
    parser.add_argument('--scst_length_bonus', type=float, default=0.0,
                        help='Length bonus coefficient (0=off; use ≤0.05 to avoid verbose drift)')
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
    # Weights & Biases
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases experiment tracking')
    parser.add_argument('--wandb_project', type=str, default='vqa-e',
                        help='W&B project name (default: vqa-e)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='W&B run name (default: auto-generated)')
    parser.add_argument('--wandb_tags', type=str, default=None,
                        help='Comma-separated W&B tags, e.g. "modelE,phase1"')
    parser.add_argument('--phase', type=int, default=None,
                        help='Training phase number (1/2/3/4) for logging/naming')
    # Tier D2: Mixed-ratio pretraining
    parser.add_argument('--mix_vqa', action='store_true',
                        help='Tier D2: Mix VQA v2.0 + VQA-E in Phase 1 to prevent length bias')
    parser.add_argument('--mix_vqa_fraction', type=float, default=0.7,
                        help='Fraction of each batch from VQA v2.0 when --mix_vqa (default 0.7)')
    parser.add_argument('--reset_best_val_loss', action='store_true',
                        help='Reset best_val_loss to inf on resume — use when starting a new phase '
                             'so early stopping does not inherit the previous phase threshold')
    # Tier D4: Curriculum Learning
    parser.add_argument('--curriculum', action='store_true',
                        help='Tier D4: Question-type curriculum — binary→color/count→what/where→why/how')
    # Tier D3: Autoregressive Masked Focal Loss
    parser.add_argument('--focal', action='store_true',
                        help='Tier D3: SequenceFocalLoss — dynamic focus on hard tokens (replaces CE)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma (focusing parameter, default 2.0)')
    parser.add_argument('--butd_feat_dir', type=str, default=None,
                        help='Tier 3B: directory of pre-extracted BUTD .pt feature files (--model F only)')
    # ── Model G flags (G1–G5) ─────────────────────────────────────────────────
    parser.add_argument('--geo7', action='store_true',
                        help='G1: 7-dim spatial geometry in BUTD encoder (adds w/W, h/H to 2048+5→2055)')
    parser.add_argument('--pgn3', action='store_true',
                        help='G2: Three-way Pointer-Generator Network (vocab + question copy + visual copy)')
    parser.add_argument('--infonce', action='store_true',
                        help='G3: InfoNCE image-text contrastive alignment (training only)')
    parser.add_argument('--infonce_beta', type=float, default=0.1,
                        help='G3: Weight for InfoNCE loss term (default: 0.1)')
    parser.add_argument('--infonce_tau', type=float, default=0.07,
                        help='G3: InfoNCE temperature tau (default: 0.07)')
    parser.add_argument('--infonce_z_dim', type=int, default=256,
                        help='G3: Projection head output dimension (default: 256)')
    parser.add_argument('--ohp', action='store_true',
                        help='G4: Object Hallucination Penalty in SCST reward')
    parser.add_argument('--ohp_weight', type=float, default=0.3,
                        help='G4: Weight for OHP term in SCST reward (default: 0.3)')
    parser.add_argument('--ohp_threshold', type=float, default=0.5,
                        help='G4: GloVe cosine sim threshold for hallucination (default: 0.5)')
    parser.add_argument('--len_cond', action='store_true',
                        help='G5: Length conditioning — append length bin embedding to LSTM input')
    # ── Model G data paths ────────────────────────────────────────────────────
    parser.add_argument('--merged_json', type=str,
                        default='data/processed/merged_train_filtered.json',
                        help='Path to merged dataset JSON (G training, default: data/processed/merged_train_filtered.json)')
    parser.add_argument('--vocab_q_path', type=str,
                        default='data/processed/vocab_questions.json',
                        help='Path to question vocabulary JSON for Model G')
    parser.add_argument('--vocab_a_path', type=str,
                        default='data/processed/vocab_answers.json',
                        help='Path to answer vocabulary JSON for Model G')
    args = parser.parse_args()
    _tags = [t.strip() for t in args.wandb_tags.split(',')] if args.wandb_tags else []

    # ── Model G dispatch ──────────────────────────────────────────────────────
    if args.model == 'G':
        from training.train_g import train_model_g
        train_model_g(args)
        sys.exit(0)

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
          use_scst=args.scst, scst_lambda=args.scst_lambda,
          scst_bleu_weight=args.scst_bleu_weight,
          scst_meteor_weight=args.scst_meteor_weight,
          scst_length_bonus=args.scst_length_bonus,
          use_wandb=args.wandb, wandb_project=args.wandb_project,
          wandb_run_name=args.wandb_run_name, wandb_tags=_tags,
          phase=args.phase,
          use_curriculum=args.curriculum,
          use_focal_loss=args.focal, focal_gamma=args.focal_gamma,
          mix_vqa=args.mix_vqa, mix_vqa_fraction=args.mix_vqa_fraction,
          reset_best_val_loss=args.reset_best_val_loss,
          butd_feat_dir=args.butd_feat_dir)
        
        
        
    
