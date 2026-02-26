import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
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
from models.vqa_models import VQAModelA, VQAModelB, VQAModelC, VQAModelD, hadamard_fusion
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


def get_model(model_type, vocab_q_size, vocab_a_size):
    """Factory function: return the model corresponding to model_type."""
    if model_type == 'A':
        return VQAModelA(vocab_size=vocab_q_size, answer_vocab_size=vocab_a_size)
    elif model_type == 'B':
        return VQAModelB(vocab_size=vocab_q_size, answer_vocab_size=vocab_a_size)
    elif model_type == 'C':
        return VQAModelC(vocab_size=vocab_q_size, answer_vocab_size=vocab_a_size)
    elif model_type == 'D':
        return VQAModelD(vocab_size=vocab_q_size, answer_vocab_size=vocab_a_size)
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
    if model_type in ('C', 'D'):
        img_features = F.normalize(model.i_encoder(imgs), p=2, dim=-1)  # (B, 49, H)
        q_feat       = model.q_encoder(questions)                        # (B, H)
        fusion       = hadamard_fusion(img_features.mean(dim=1), q_feat) # (B, H)
    else:
        img_feat = F.normalize(model.i_encoder(imgs), p=2, dim=1)       # (B, H)
        q_feat   = model.q_encoder(questions)                            # (B, H)
        fusion   = hadamard_fusion(img_feat, q_feat)                     # (B, H)

    h = fusion.unsqueeze(0).repeat(model.num_layers, 1, 1)  # (L, B, H)
    c = torch.zeros_like(h)
    hidden = (h, c)

    # ── Step-by-step decoding ────────────────────────────────────────
    current_token = decoder_input[:, 0]   # (B,) — first token is always <start>
    logits_list   = []

    for t in range(max_len):
        tok = current_token.unsqueeze(1)  # (B, 1)

        if model_type in ('C', 'D'):
            logit, hidden, _ = model.decoder.decode_step(tok, hidden, img_features)
            # logit: (B, vocab), hidden: updated (h, c)
        else:
            emb        = model.decoder.dropout(model.decoder.embedding(tok)) # (B, 1, embed)
            out, hidden = model.decoder.lstm(emb, hidden)        # (B, 1, H)
            logit      = model.decoder.fc(out.squeeze(1))        # (B, vocab)

        logits_list.append(logit)

        # Choose next input: ground truth (prob epsilon) or own prediction
        if t < max_len - 1:
            if random.random() < epsilon:
                current_token = decoder_input[:, t + 1]          # GT token
            else:
                current_token = logit.detach().argmax(dim=-1)    # model's prediction

    return torch.stack(logits_list, dim=1)  # (B, max_len, vocab_size)


# ── Training set (train2014) ─────────────────────────────────────
TRAIN_IMAGE_DIR       = "data/raw/images/train2014"
TRAIN_QUESTION_JSON   = "data/raw/vqa_json/v2_OpenEnded_mscoco_train2014_questions.json"
TRAIN_ANNOTATION_JSON = "data/raw/vqa_json/v2_mscoco_train2014_annotations.json"

# ── Validation set (val2014) ──────────────────────────────────────
VAL_IMAGE_DIR         = "data/raw/images/val2014"
VAL_QUESTION_JSON     = "data/raw/vqa_json/v2_OpenEnded_mscoco_val2014_questions.json"
VAL_ANNOTATION_JSON   = "data/raw/vqa_json/v2_mscoco_val2014_annotations.json"

VOCAB_Q_PATH  = "data/processed/vocab_questions.json"
VOCAB_A_PATH  = "data/processed/vocab_answers.json"

# Set to a number (e.g. 10000) to cap samples for quick pipeline tests.
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES   = None


def train(model_type='A', epochs=10, lr=1e-3, batch_size=128, resume=None,
          scheduled_sampling=False, ss_k=5,
          finetune_cnn=False, cnn_lr_factor=0.1,
          num_workers=4, weight_decay=1e-5,
          early_stopping_patience=0, augment=False):
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
        augment=augment
    )

    val_dataset = VQADataset(
        image_dir=VAL_IMAGE_DIR,
        question_json_path=VAL_QUESTION_JSON,
        annotations_json_path=VAL_ANNOTATION_JSON,
        vocab_q=vocab_q,
        vocab_a=vocab_a,
        split='val2014',
        max_samples=MAX_VAL_SAMPLES
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

    model     = get_model(model_type, len(vocab_q), len(vocab_a)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # ── Optimizer — differential LR when fine-tuning the CNN backbone ──────────
    # Models A/C use scratch CNN → finetune_cnn flag has no effect on them.
    # Models B/D use frozen ResNet → finetune_cnn selectively unfreezes
    # layer3 + layer4 with a smaller LR (cnn_lr_factor × base_lr) to avoid
    # catastrophic forgetting of pretrained ImageNet knowledge.
    if finetune_cnn and model_type in ('B', 'D'):
        model.i_encoder.unfreeze_top_layers()
        backbone_param_ids = {id(p) for p in model.i_encoder.backbone_params()}
        backbone_params = [p for p in model.parameters()
                           if id(p) in backbone_param_ids and p.requires_grad]
        other_params    = [p for p in model.parameters()
                           if id(p) not in backbone_param_ids and p.requires_grad]
        optimizer = optim.Adam([
            {'params': other_params,    'lr': lr},
            {'params': backbone_params, 'lr': lr * cnn_lr_factor},
        ], weight_decay=weight_decay)
        print(f"CNN fine-tuning  : ON | backbone LR = {lr * cnn_lr_factor:.2e}  other LR = {lr:.2e}")
        print(f"  trainable backbone params : {sum(p.numel() for p in backbone_params):,}")
    else:
        optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    if weight_decay > 0:
        print(f"Weight decay     : {weight_decay:.1e}")

    # Halve LR when val loss stops improving for 2 consecutive epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # Mixed precision — BF16 on Ampere+ (A100, etc.), FP16 elsewhere
    use_amp  = torch.cuda.is_available()
    use_bf16 = use_amp and _supports_bf16()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    # GradScaler not needed for BF16 (wider dynamic range), still used for FP16
    scaler = GradScaler(enabled=(use_amp and not use_bf16))
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

        # Optimizer/scheduler state can only be restored when the number of
        # parameter groups matches.  Phase transitions (e.g. Phase 1 → 2)
        # change the optimizer layout (frozen → unfreeze adds a param group),
        # so we skip restoring optimizer/scheduler in that case and start
        # with a fresh optimizer + the new LR from CLI args.
        saved_groups = len(ckpt['optimizer_state_dict']['param_groups'])
        current_groups = len(optimizer.param_groups)
        if saved_groups == current_groups:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            scaler.load_state_dict(ckpt['scaler_state_dict'])
            print("  Optimizer & scheduler state restored.")
        else:
            print(f"  Optimizer layout changed ({saved_groups} → {current_groups} param groups) "
                  f"— using fresh optimizer with current LR settings.")

        start_epoch   = ckpt['epoch']          # last completed epoch
        best_val_loss = ckpt['best_val_loss']
        history       = ckpt.get('history', history)
        print(f"  Resumed at epoch {start_epoch} | best_val_loss: {best_val_loss:.4f}")

    print(f"Model: {model_type} | Device: {DEVICE}")
    if augment:
        print("Data augmentation: ON (RandomHorizontalFlip + ColorJitter)")
    if early_stopping_patience > 0:
        print(f"Early stopping   : patience={early_stopping_patience}")
    es_counter = 0  # early stopping counter
    if scheduled_sampling:
        print(f"Scheduled Sampling: ON | k={ss_k} (epsilon decays from "
              f"{ss_k/(ss_k+math.exp(0/ss_k)):.2f} to "
              f"{ss_k/(ss_k+math.exp((start_epoch+epochs-1)/ss_k)):.2f} over training)")
    for epoch in tqdm.tqdm(range(start_epoch, start_epoch + epochs)):
        # ── Train ────────────────────────────────────────────────
        model.train()
        total_loss = 0

        for imgs, questions, answer in train_loader:
            imgs      = imgs.to(DEVICE)
            questions = questions.to(DEVICE)
            answer    = answer.to(DEVICE)

            # Teacher forcing: input [<start>, w1, ..., wn], target [w1, ..., <end>]
            decoder_input  = answer[:, :-1]
            decoder_target = answer[:, 1:]

            optimizer.zero_grad()

            with autocast(enabled=use_amp, dtype=amp_dtype):
                if scheduled_sampling:
                    # Inverse-sigmoid epsilon decay: higher k = slower decay
                    # epoch 0 -> epsilon~1 (mostly GT), last epoch -> epsilon lowers
                    epsilon = ss_k / (ss_k + math.exp(epoch / ss_k))
                    logits  = ss_forward(model, model_type, imgs, questions,
                                         decoder_input, epsilon)
                else:
                    logits  = model(imgs, questions, decoder_input)
                vocab_size = logits.size(-1)
                loss = criterion(
                    logits.view(-1, vocab_size),
                    decoder_target.contiguous().view(-1)
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

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

                with autocast(enabled=use_amp, dtype=amp_dtype):
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
            eps = ss_k / (ss_k + math.exp(epoch / ss_k))
            print(f"Epoch {epoch+1}/{start_epoch + epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.2e} | SS ϵ: {eps:.3f}")
        else:
            print(f"Epoch {epoch+1}/{start_epoch + epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.2e}")

        # Step LR scheduler based on val loss
        scheduler.step(avg_val_loss)

        # Save history after every epoch (safe against session interruption)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        # Save per-epoch checkpoint (useful for resume / comparison)
        torch.save(model.state_dict(), f"checkpoints/model_{model_type.lower()}_epoch{epoch+1}.pth")

        # Save full resume checkpoint (model + optimizer + scheduler + scaler + metadata)
        resume_ckpt = {
            'epoch':                epoch + 1,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict':    scaler.state_dict(),
            'best_val_loss':        best_val_loss,
            'history':              history,
        }
        torch.save(resume_ckpt, f"checkpoints/model_{model_type.lower()}_resume.pth")

        # Save best checkpoint separately
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
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VQA model.")
    parser.add_argument('--model',      type=str,   default='A', choices=['A', 'B', 'C', 'D'],
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
    args = parser.parse_args()
    train(model_type=args.model, epochs=args.epochs, lr=args.lr,
          batch_size=args.batch_size, resume=args.resume,
          scheduled_sampling=args.scheduled_sampling, ss_k=args.ss_k,
          finetune_cnn=args.finetune_cnn, cnn_lr_factor=args.cnn_lr_factor,
          num_workers=args.num_workers, weight_decay=args.weight_decay,
          early_stopping_patience=args.early_stopping, augment=args.augment)
        
        
        
    
