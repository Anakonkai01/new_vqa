import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import os
import sys
import json
import argparse
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


def train(model_type='A', epochs=10, lr=1e-3, batch_size=128, resume=None):
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
        max_samples=MAX_TRAIN_SAMPLES
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
        num_workers=4,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=vqa_collate_fn,
        num_workers=4,
        pin_memory=pin_memory
    )

    model     = get_model(model_type, len(vocab_q), len(vocab_a)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    # Halve LR when val loss stops improving for 2 consecutive epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # Mixed precision — enabled only on CUDA
    use_amp = torch.cuda.is_available()
    scaler  = GradScaler(enabled=use_amp)

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
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch   = ckpt['epoch']          # last completed epoch
        best_val_loss = ckpt['best_val_loss']
        history       = ckpt.get('history', history)
        print(f"  Resumed at epoch {start_epoch} | best_val_loss: {best_val_loss:.4f}")

    print(f"Model: {model_type} | Device: {DEVICE}")
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

            with autocast(enabled=use_amp):
                logits     = model(imgs, questions, decoder_input)
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

                with autocast(enabled=use_amp):
                    logits     = model(imgs, questions, decoder_input)
                    vocab_size = logits.size(-1)
                    loss = criterion(
                        logits.view(-1, vocab_size),
                        decoder_target.contiguous().view(-1)
                    )
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        current_lr   = optimizer.param_groups[0]['lr']
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
            torch.save(model.state_dict(), f"checkpoints/model_{model_type.lower()}_best.pth")
            print(f"  -> New best val loss: {best_val_loss:.4f}. Saved best checkpoint.")


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
    args = parser.parse_args()
    train(model_type=args.model, epochs=args.epochs, lr=args.lr,
          batch_size=args.batch_size, resume=args.resume)
        
        
        
    
