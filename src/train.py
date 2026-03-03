import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import json
import argparse
import math
import random
import tqdm
sys.path.append(os.path.dirname(__file__))

# Import modules
from dataset import VQADataset, vqa_collate_fn
from models.vqa_models import VQAModelA, VQAModelB, VQAModelC, VQAModelD
from vocab import Vocabulary


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
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _supports_bf16():
    """Check if GPU supports BFloat16 (Ampere+ = compute capability >= 8.0)."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 8


def _fused_adam_available():
    """Check if fused AdamW is available (CUDA + PyTorch >= 2.0)."""
    if not torch.cuda.is_available():
        return False
    try:
        optim.AdamW([torch.zeros(1, device='cuda')], fused=True)
        return True
    except Exception:
        return False


def get_model(model_type, vocab_q_size, vocab_a_size, dropout=0.3):
    """Factory function: return the model corresponding to model_type."""
    if model_type == 'A':
        return VQAModelA(vocab_size=vocab_q_size, answer_vocab_size=vocab_a_size,
                         dropout=dropout)
    elif model_type == 'B':
        return VQAModelB(vocab_size=vocab_q_size, answer_vocab_size=vocab_a_size,
                         dropout=dropout)
    elif model_type == 'C':
        return VQAModelC(vocab_size=vocab_q_size, answer_vocab_size=vocab_a_size,
                         dropout=dropout)
    elif model_type == 'D':
        return VQAModelD(vocab_size=vocab_q_size, answer_vocab_size=vocab_a_size,
                         dropout=dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from A, B, C, D.")


def ss_forward(model, model_type, imgs, questions, decoder_input, epsilon):
    """
    Scheduled Sampling forward pass (token-level, Bengio et al. 2015).

    At each decode step t the decoder receives the ground-truth token with
    probability `epsilon` and its own prediction from step t-1 with probability
    (1 - epsilon).

        epsilon = 1.0  =>  pure teacher forcing  (original behaviour)
        epsilon = 0.0  =>  fully autoregressive  (same as inference)

    epsilon follows an inverse-sigmoid decay schedule relative to phase start:
        epsilon(relative_epoch) = k / (k + exp(relative_epoch / k))
    so it starts close to 1 and gradually decays.

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
    if model_type in ('C', 'D'):
        img_features = F.normalize(model.i_encoder(imgs), p=2, dim=-1)   # (B, 49, H)
        q_feat       = F.normalize(model.q_encoder(questions), p=2, dim=1) # (B, H)
        fusion       = model.fusion(img_features.mean(dim=1), q_feat)      # (B, H)
    else:
        img_feat = F.normalize(model.i_encoder(imgs), p=2, dim=1)          # (B, H)
        q_feat   = F.normalize(model.q_encoder(questions), p=2, dim=1)     # (B, H)
        fusion   = model.fusion(img_feat, q_feat)                           # (B, H)

    # layer 0 = fusion, upper layers = zeros
    h = torch.zeros(model.num_layers, B, fusion.size(1), device=fusion.device)
    h[0] = fusion
    c = torch.zeros_like(h)
    hidden = (h, c)

    # ── Step-by-step decoding ────────────────────────────────────────
    current_token = decoder_input[:, 0]   # (B,) — first token is always <start>
    logits_list   = []

    for t in range(max_len):
        tok = current_token.unsqueeze(1)  # (B, 1)

        if model_type in ('C', 'D'):
            logit, hidden, _ = model.decoder.decode_step(tok, hidden, img_features)
        else:
            emb        = model.decoder.dropout(model.decoder.embedding(tok))
            out, hidden = model.decoder.lstm(emb, hidden)
            logit      = model.decoder.fc(out.squeeze(1))

        logits_list.append(logit)

        if t < max_len - 1:
            if random.random() < epsilon:
                current_token = decoder_input[:, t + 1]
            else:
                current_token = logit.detach().argmax(dim=-1)

    return torch.stack(logits_list, dim=1)  # (B, max_len, vocab_size)


# ── Training set (train2014) ─────────────────────────────────────
TRAIN_IMAGE_DIR       = "data/train2014/train2014"
TRAIN_QUESTION_JSON   = "data/vqa_data_json/v2_Questions_Train_mscoco/v2_OpenEnded_mscoco_train2014_questions.json"
TRAIN_ANNOTATION_JSON = "data/vqa_data_json/v2_Annotations_Train_mscoco/v2_mscoco_train2014_annotations.json"

# ── Validation set (val2014) ──────────────────────────────────────
VAL_IMAGE_DIR         = "data/val2014"
VAL_QUESTION_JSON     = "data/vqa_data_json/v2_Questions_Val_mscoco/v2_OpenEnded_mscoco_val2014_questions.json"
VAL_ANNOTATION_JSON   = "data/vqa_data_json/v2_Annotations_Val_mscoco/v2_mscoco_val2014_annotations.json"

VOCAB_Q_PATH  = "data/processed/vocab_questions.json"
VOCAB_A_PATH  = "data/processed/vocab_answers.json"

MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES   = None


def train(model_type='A', epochs=10, lr=1e-3, batch_size=128, resume=None,
          scheduled_sampling=False, ss_k=5,
          finetune_cnn=False, cnn_lr_factor=0.1,
          num_workers=4, weight_decay=1e-5,
          early_stopping_patience=0, augment=False,
          dropout=0.3, label_smoothing=0.1,
          grad_accum_steps=1, answer_sampling=False,
          warmup_epochs=1):
    os.makedirs("checkpoints", exist_ok=True)

    vocab_q = Vocabulary(); vocab_q.load(VOCAB_Q_PATH)
    vocab_a = Vocabulary(); vocab_a.load(VOCAB_A_PATH)

    train_dataset = VQADataset(
        image_dir=TRAIN_IMAGE_DIR,
        question_json_path=TRAIN_QUESTION_JSON,
        annotations_json_path=TRAIN_ANNOTATION_JSON,
        vocab_q=vocab_q,
        vocab_a=vocab_a,
        split='train2014',
        max_samples=MAX_TRAIN_SAMPLES,
        augment=augment,
        answer_sampling=answer_sampling
    )

    val_dataset = VQADataset(
        image_dir=VAL_IMAGE_DIR,
        question_json_path=VAL_QUESTION_JSON,
        annotations_json_path=VAL_ANNOTATION_JSON,
        vocab_q=vocab_q,
        vocab_a=vocab_a,
        split='val2014',
        max_samples=MAX_VAL_SAMPLES,
        answer_sampling=False   # val always uses majority vote
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

    model     = get_model(model_type, len(vocab_q), len(vocab_a), dropout=dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing)

    # ── Optimizer — AdamW with optional differential LR ───────────────────
    fused = _fused_adam_available()
    if finetune_cnn and model_type in ('B', 'D'):
        model.i_encoder.unfreeze_top_layers()
        backbone_param_ids = {id(p) for p in model.i_encoder.backbone_params()}
        backbone_params = [p for p in model.parameters()
                           if id(p) in backbone_param_ids and p.requires_grad]
        other_params    = [p for p in model.parameters()
                           if id(p) not in backbone_param_ids and p.requires_grad]
        optimizer = optim.AdamW([
            {'params': other_params,    'lr': lr},
            {'params': backbone_params, 'lr': lr * cnn_lr_factor},
        ], weight_decay=weight_decay, fused=fused)
        print(f"CNN fine-tuning  : ON | backbone LR = {lr * cnn_lr_factor:.2e}  other LR = {lr:.2e}")
        print(f"  trainable backbone params : {sum(p.numel() for p in backbone_params):,}")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr,
                                weight_decay=weight_decay, fused=fused)

    if weight_decay > 0:
        print(f"Weight decay     : {weight_decay:.1e}")

    # ── LR Schedulers ─────────────────────────────────────────────────────
    # Warmup: linear ramp from lr*0.1 → lr over warmup_epochs epochs
    # Then: ReduceLROnPlateau halves LR when val loss plateaus for 2 epochs
    warmup_sched = None
    if warmup_epochs > 0:
        warmup_sched = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2)

    # ── Mixed Precision — updated API (torch.amp instead of torch.cuda.amp) ──
    use_amp  = torch.cuda.is_available()
    use_bf16 = use_amp and _supports_bf16()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and not use_bf16))
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
        model.load_state_dict(ckpt['model_state_dict'])

        saved_groups   = len(ckpt['optimizer_state_dict']['param_groups'])
        current_groups = len(optimizer.param_groups)
        if saved_groups == current_groups:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            main_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            scaler.load_state_dict(ckpt['scaler_state_dict'])
            print("  Optimizer & scheduler state restored.")
        else:
            print(f"  Optimizer layout changed ({saved_groups} → {current_groups} param groups) "
                  f"— using fresh optimizer with current LR settings.")

        start_epoch   = ckpt['epoch']
        best_val_loss = ckpt['best_val_loss']
        history       = ckpt.get('history', history)
        print(f"  Resumed at epoch {start_epoch} | best_val_loss: {best_val_loss:.4f}")

    print(f"Model: {model_type} | Device: {DEVICE}")
    if augment:
        print("Data augmentation: ON (RandomHorizontalFlip + ColorJitter)")
    if answer_sampling:
        print("Answer sampling  : ON (random 1 of 10 annotations per sample)")
    if warmup_epochs > 0:
        print(f"LR Warmup        : {warmup_epochs} epoch(s) (start_factor=0.1)")
    if early_stopping_patience > 0:
        print(f"Early stopping   : patience={early_stopping_patience}")
    if grad_accum_steps > 1:
        effective_bs = batch_size * grad_accum_steps
        print(f"Grad accumulation: {grad_accum_steps} steps (effective batch = {effective_bs})")
    print(f"Dropout          : {dropout}")
    print(f"Label smoothing  : {label_smoothing}")

    es_counter = 0
    if scheduled_sampling:
        eps_start = ss_k / (ss_k + math.exp(0 / ss_k))
        eps_end   = ss_k / (ss_k + math.exp((epochs - 1) / ss_k))
        print(f"Scheduled Sampling: ON | k={ss_k} (epsilon decays from "
              f"{eps_start:.3f} to {eps_end:.3f} over this phase)")

    for epoch in tqdm.tqdm(range(start_epoch, start_epoch + epochs)):
        # ── Train ────────────────────────────────────────────────
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for step, (imgs, questions, answer) in enumerate(train_loader):
            imgs      = imgs.to(DEVICE)
            questions = questions.to(DEVICE)
            answer    = answer.to(DEVICE)

            decoder_input  = answer[:, :-1]
            decoder_target = answer[:, 1:]

            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                if scheduled_sampling:
                    # epsilon relative to THIS phase's start (not absolute epoch)
                    relative_epoch = epoch - start_epoch
                    epsilon = ss_k / (ss_k + math.exp(relative_epoch / ss_k))
                    logits  = ss_forward(model, model_type, imgs, questions,
                                         decoder_input, epsilon)
                else:
                    logits  = model(imgs, questions, decoder_input)
                vocab_size = logits.size(-1)
                loss = criterion(
                    logits.view(-1, vocab_size),
                    decoder_target.contiguous().view(-1)
                )
                # scale down for accumulation
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            # optimizer step every grad_accum_steps batches (or at end of epoch)
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum_steps  # unscale for logging

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

                with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                    logits     = model(imgs, questions, decoder_input)
                    vocab_size = logits.size(-1)
                    loss = criterion(
                        logits.view(-1, vocab_size),
                        decoder_target.contiguous().view(-1)
                    )
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        current_lr   = optimizer.param_groups[0]['lr']

        if scheduled_sampling:
            relative_epoch = epoch - start_epoch
            eps = ss_k / (ss_k + math.exp(relative_epoch / ss_k))
            print(f"Epoch {epoch+1}/{start_epoch + epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.2e} | SS ϵ: {eps:.3f}")
        else:
            print(f"Epoch {epoch+1}/{start_epoch + epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.2e}")

        # ── LR Scheduling ─────────────────────────────────────────
        # Warmup only during first warmup_epochs of this phase
        relative_epoch = epoch - start_epoch
        if warmup_sched is not None and relative_epoch < warmup_epochs:
            warmup_sched.step()
        else:
            main_scheduler.step(avg_val_loss)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        # ── Checkpoint saving ─────────────────────────────────────
        comparison_epochs = {10, 15, 20}
        current_epoch = epoch + 1

        resume_ckpt = {
            'epoch':                current_epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': main_scheduler.state_dict(),
            'scaler_state_dict':    scaler.state_dict(),
            'best_val_loss':        best_val_loss,
            'history':              history,
        }
        torch.save(resume_ckpt, f"checkpoints/model_{model_type.lower()}_resume.pth")

        if current_epoch in comparison_epochs:
            torch.save(model.state_dict(), f"checkpoints/model_{model_type.lower()}_epoch{current_epoch}.pth")
            print(f"  Saved milestone checkpoint: epoch {current_epoch}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            es_counter = 0
            torch.save(model.state_dict(), f"checkpoints/model_{model_type.lower()}_best.pth")
            print(f"  -> New best val loss: {best_val_loss:.4f}. Saved best checkpoint.")
        else:
            es_counter += 1
            if early_stopping_patience > 0:
                print(f"  Val loss did not improve ({es_counter}/{early_stopping_patience})")
                if es_counter >= early_stopping_patience:
                    print(f"  Early stopping triggered after {epoch+1} epochs.")
                    target_epoch  = start_epoch + epochs
                    milestone_path = f"checkpoints/model_{model_type.lower()}_epoch{target_epoch}.pth"
                    best_path      = f"checkpoints/model_{model_type.lower()}_best.pth"
                    if not os.path.exists(milestone_path) and os.path.exists(best_path):
                        import shutil
                        shutil.copy2(best_path, milestone_path)
                        print(f"  Copied best checkpoint → {milestone_path} (for compare.py)")
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VQA model.")
    parser.add_argument('--model',        type=str,   default='A', choices=['A', 'B', 'C', 'D'])
    parser.add_argument('--epochs',       type=int,   default=10)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--batch_size',   type=int,   default=128)
    parser.add_argument('--resume',       type=str,   default=None)
    parser.add_argument('--scheduled_sampling', action='store_true')
    parser.add_argument('--ss_k',         type=float, default=5.0,
                        help='Inverse-sigmoid decay speed for Scheduled Sampling (default: 5.0). '
                             'Use ss_k=2 for faster decay over short phases.')
    parser.add_argument('--finetune_cnn', action='store_true')
    parser.add_argument('--cnn_lr_factor',type=float, default=0.1)
    parser.add_argument('--num_workers',  type=int,   default=4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--early_stopping', type=int, default=0,
                        help='Stop if val loss does not improve for N epochs (0=disabled)')
    parser.add_argument('--augment',      action='store_true',
                        help='Image augmentation: RandomHorizontalFlip + ColorJitter')
    parser.add_argument('--dropout',      type=float, default=0.3,
                        help='Dropout for decoder embedding + LSTM inter-layer (default: 0.3)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing for CrossEntropyLoss (default: 0.1)')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Gradient accumulation steps; effective_batch = batch_size × steps (default: 1)')
    parser.add_argument('--answer_sampling', action='store_true',
                        help='Randomly pick 1 of 10 human annotations per training sample')
    parser.add_argument('--warmup_epochs', type=int, default=1,
                        help='Linear LR warmup epochs at start of each phase (default: 1)')
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
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        grad_accum_steps=args.grad_accum_steps,
        answer_sampling=args.answer_sampling,
        warmup_epochs=args.warmup_epochs,
    )
