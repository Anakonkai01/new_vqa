"""Self-Critical Sequence Training (SCST) script — Phase 4 RL fine-tuning.

Implements the REINFORCE algorithm with a greedy-decode baseline (Rennie et al.,
2017) to optimise BLEU-4 reward directly after supervised pre-training in
Phases 1–3.

Algorithm
---------
For each batch:

1. **Greedy decode** (no grad, model.eval()) to obtain the baseline sequences
   and their BLEU-4 rewards ``r_greedy``.
2. **Sampled decode** (with grad, model.train()) to obtain exploratory sequences,
   their log-probabilities, and rewards ``r_sample``.
3. **Advantage** = ``r_sample − r_greedy``  (positive if sample beats greedy).
4. **REINFORCE loss**::

       L = − mean( advantage · Σ_t log p(y_t | y_{<t}, x) )

   Multiplying by a *negative* advantage penalises sequences worse than greedy
   and rewards those that outperform it.

5. Only decoder parameters receive gradients; encoder is kept frozen to reduce
   gradient variance during RL fine-tuning.

Mixed Precision
---------------
BFloat16 is used on Ampere+ GPUs (RTX 5070 Ti); FP16 + GradScaler on older
cards; CPU runs in FP32. The greedy forward pass is wrapped in ``torch.no_grad``
so no scaler is needed there.

Usage
-----
    python src/train_rl.py --model_type E \\
        --base_checkpoint checkpoints/model_e_best.pth \\
        --epochs 3 --batch_size 32 --lr 1e-5

Notes
-----
* RL fine-tuning is designed to run *after* Phase 3 scheduled sampling.
  Pass the final SS checkpoint to ``--base_checkpoint``.
* Only spatial-attention models (C, D, E) are supported — global-vector models
  (A, B) have no ``decode_step`` compatible with the attention signature.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))

from dataset import VQAEDataset, vqa_collate_fn
from train import get_model  # reuse the shared model factory
from vocab import Vocabulary


# ── NLTK resources ────────────────────────────────────────────────────────────

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

_smoothie = SmoothingFunction().method1


# ── Device & GPU optimisations ────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
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


# ── Data paths ────────────────────────────────────────────────────────────────

TRAIN_IMAGE_DIR  = "data/raw/train2014"
TRAIN_VQA_E_JSON = "data/vqa_e/VQA-E_train_set.json"
VOCAB_Q_PATH     = "data/processed/vocab_questions.json"
VOCAB_A_PATH     = "data/processed/vocab_answers.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decode_tensor(seq: Tensor, vocab: Vocabulary) -> List[str]:
    """Convert a 1-D token-index tensor to a list of content words.

    Skips ``<pad>``, ``<start>``, and ``<end>`` tokens so that the word list
    passed to BLEU scoring contains only meaningful content tokens.

    Args:
        seq: ``LongTensor (T,)`` — single sequence of token indices.
        vocab: Answer vocabulary with ``pad_idx``, ``start_idx``, ``end_idx``
            properties and an ``idx2word`` mapping.

    Returns:
        List of decoded word strings (may be empty).
    """
    skip = {vocab.pad_idx, vocab.start_idx, vocab.end_idx}
    return [vocab.idx2word[int(i)] for i in seq if int(i) not in skip]


def _compute_rewards(
    sampled_seqs: Tensor,
    greedy_seqs: Tensor,
    target_seqs: Tensor,
    vocab: Vocabulary,
) -> Tuple[Tensor, Tensor]:
    """Compute sentence-level BLEU-4 rewards for sampled and greedy sequences.

    Args:
        sampled_seqs: ``LongTensor (B, T)`` — stochastically sampled sequences.
        greedy_seqs:  ``LongTensor (B, T)`` — greedy-decoded sequences (baseline).
        target_seqs:  ``LongTensor (B, T)`` — ground-truth answer token sequences.
        vocab: Answer vocabulary used for decoding to word strings.

    Returns:
        Tuple of:
            sampled_rewards – ``FloatTensor (B,)`` — BLEU-4 score per sample.
            greedy_rewards  – ``FloatTensor (B,)`` — BLEU-4 score per greedy decode.

    Note:
        Reward tensors are allocated on ``target_seqs.device`` so they are
        already on the correct device for the REINFORCE loss computation.
        A short-sequence penalty of 0.5× is applied to sequences with < 3 tokens
        to discourage degenerate one-word or two-word outputs.
    """
    device    = target_seqs.device  # inherit device from input — no global DEVICE
    B         = target_seqs.size(0)
    r_sampled = torch.zeros(B, device=device)
    r_greedy  = torch.zeros(B, device=device)

    for i in range(B):
        gt_words   = _decode_tensor(target_seqs[i], vocab)
        if not gt_words:
            gt_words = [vocab.idx2word.get(vocab.unk_idx, "<unk>")]

        samp_words  = _decode_tensor(sampled_seqs[i], vocab)
        greed_words = _decode_tensor(greedy_seqs[i], vocab)

        bleu4_weights = (0.25, 0.25, 0.25, 0.25)
        sr = sentence_bleu([gt_words], samp_words,
                           weights=bleu4_weights, smoothing_function=_smoothie)
        gr = sentence_bleu([gt_words], greed_words,
                           weights=bleu4_weights, smoothing_function=_smoothie)

        # Penalise degenerate short outputs (< 3 meaningful tokens).
        if len(samp_words)  < 3:
            sr *= 0.5
        if len(greed_words) < 3:
            gr *= 0.5

        r_sampled[i] = sr
        r_greedy[i]  = gr

    return r_sampled, r_greedy


def _run_encoder_and_greedy(
    model: nn.Module,
    model_type: str,
    imgs: Tensor,
    questions: Tensor,
    max_len: int,
    start_idx: int,
    end_idx: int,
    amp_dtype: torch.dtype,
) -> Tuple[Tuple[Tensor, Tensor], Tensor, Tensor, Tensor]:
    """Run the encoder under no-grad + eval mode and obtain greedy sequences.

    Separates the no-gradient portion of the SCST loop (encoder + greedy decode)
    from the differentiable sampled decode so that only the decoder parameters
    that receive gradients need to remain in train mode.

    Args:
        model: VQA model (spatial-attention type: C, D, or E).
        model_type: One of ``'C'``, ``'D'``, ``'E'``.
        imgs: ``FloatTensor (B, 3, 224, 224)``.
        questions: ``LongTensor (B, Q)``.
        max_len: Maximum decode length.
        start_idx: ``<start>`` token index.
        end_idx: ``<end>`` token index.
        amp_dtype: ``torch.bfloat16`` or ``torch.float16`` for autocast.

    Returns:
        Tuple of:
            encoder_hidden       – ``(h_0, c_0)`` LSTM init state from fusion.
            modulated_img_feats  – ``FloatTensor (B, 49, H)`` spatial features.
            q_hidden_states      – ``FloatTensor (B, Q, H)`` question states.
            greedy_seqs          – ``LongTensor  (B, max_len)`` baseline sequences.

    Raises:
        NotImplementedError: If *model_type* is not ``'C'``, ``'D'``, or ``'E'``.
    """
    if model_type not in ("C", "D", "E"):
        raise NotImplementedError(
            "RL fine-tuning is only implemented for spatial-attention models "
            f"(C, D, E). Got model_type='{model_type}'."
        )

    device = imgs.device

    with torch.no_grad(), autocast(device_type=device.type, dtype=amp_dtype,
                                   enabled=(device.type == "cuda")):
        # ── Encode image ──────────────────────────────────────────────────────
        img_features = F.normalize(model.i_encoder(imgs), p=2, dim=-1)
        # img_features: (B, 49, H)

        # ── Encode question ───────────────────────────────────────────────────
        q_feat, q_hidden_states = model.q_encoder(questions)
        # q_feat: (B, H) | q_hidden_states: (B, Q, H)

        if hasattr(model, "q_norm"):
            q_feat = model.q_norm(q_feat)  # Model E: stabilise FiLM inputs

        # ── Fusion → LSTM init state ──────────────────────────────────────────
        if model_type == "E":
            img_mean      = img_features.mean(dim=1)             # (B, H)
            fusion_global = model.fusion(img_mean, q_feat)       # (B, H) FiLM-fused
            h_0_base      = model.init_h_proj(fusion_global)     # (B, H)
            c_0_base      = model.init_c_proj(fusion_global)     # (B, H)
            h_0 = h_0_base.unsqueeze(0).repeat(model.num_layers, 1, 1)  # (L, B, H)
            c_0 = c_0_base.unsqueeze(0).repeat(model.num_layers, 1, 1)  # (L, B, H)
            # Apply FiLM to all 49 spatial patches — same as VQAModelE.forward()
            # img_features: (B, 49, H) | q_feat: (B, H) → FiLM broadcast
            modulated_img_feats = model.fusion(img_features, q_feat)  # (B, 49, H)
        else:
            # Models C / D: GatedFusion or simple concat; no FiLM modulation.
            img_mean      = img_features.mean(dim=1)             # (B, H)
            fusion_global = model.fusion(img_mean, q_feat)       # (B, H)
            h_0 = fusion_global.unsqueeze(0).repeat(model.num_layers, 1, 1)  # (L, B, H)
            c_0 = torch.zeros_like(h_0)                                       # (L, B, H)
            modulated_img_feats = img_features  # un-modulated; (B, 49, H)

        encoder_hidden = (h_0, c_0)

        # ── Greedy decode (baseline) ──────────────────────────────────────────
        greedy_seqs, _ = model.decoder.sample(
            encoder_hidden, modulated_img_feats, q_hidden_states,
            max_len, start_idx, end_idx, method="greedy",
        )
        # greedy_seqs: (B, max_len)

    return encoder_hidden, modulated_img_feats, q_hidden_states, greedy_seqs


# ── SCST training loop ────────────────────────────────────────────────────────

def train_rl_epoch(
    model: nn.Module,
    model_type: str,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    vocab_a: Vocabulary,
    start_idx: int,
    end_idx: int,
    amp_dtype: torch.dtype,
    max_len: int = 60,
) -> Tuple[float, float, float]:
    """Run one SCST epoch.

    Args:
        model: Spatial-attention VQA model (C, D, or E).
        model_type: Model variant string ``'C'``, ``'D'``, or ``'E'``.
        loader: DataLoader yielding ``(imgs, questions, answers)`` batches.
        optimizer: Decoder-only optimizer.
        scaler: GradScaler for FP16 AMP (no-op on BF16 and CPU).
        vocab_a: Answer vocabulary (with ``pad_idx`` etc.).
        start_idx: ``<start>`` token index.
        end_idx: ``<end>`` token index.
        amp_dtype: ``torch.bfloat16`` or ``torch.float16``.
        max_len: Maximum number of tokens to generate per sequence.

    Returns:
        Tuple ``(avg_loss, avg_sample_reward, avg_greedy_reward)`` for the epoch.
    """
    model.train()

    total_loss         = 0.0
    total_r_samp       = 0.0
    total_r_greed      = 0.0
    device_type        = DEVICE.type

    pbar = tqdm(loader, desc="RL SCST")
    for imgs, questions, answers in pbar:
        imgs      = imgs.to(DEVICE, non_blocking=True)
        questions = questions.to(DEVICE, non_blocking=True)
        answers   = answers.to(DEVICE, non_blocking=True)

        # ── Step 1: Encoder + greedy baseline (no grad, eval BN/Dropout) ──────
        model.eval()
        encoder_hidden, mod_img_feats, q_hidden, greedy_seqs = _run_encoder_and_greedy(
            model, model_type, imgs, questions,
            max_len, start_idx, end_idx, amp_dtype,
        )
        # Detach all encoder outputs so sampled decode gradients flow only
        # through the decoder parameters.
        encoder_hidden = (encoder_hidden[0].detach(), encoder_hidden[1].detach())
        mod_img_feats  = mod_img_feats.detach()
        q_hidden       = q_hidden.detach()

        # ── Step 2: Sampled decode (with grad, train mode) ────────────────────
        model.train()
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device_type, dtype=amp_dtype,
                      enabled=(device_type == "cuda")):
            samp_seqs, samp_log_probs = model.decoder.sample(
                encoder_hidden, mod_img_feats, q_hidden,
                max_len, start_idx, end_idx, method="sample",
            )
            # samp_seqs:      (B, max_len)  — sampled token IDs
            # samp_log_probs: (B, max_len)  — log p(y_t | y_{<t}, x)

            # ── Step 3: Rewards (non-differentiable, computed outside autocast) ─
            with torch.no_grad():
                r_samp, r_greed = _compute_rewards(
                    samp_seqs, greedy_seqs, answers, vocab_a
                )
                advantage = r_samp - r_greed  # (B,)  positive → sample better

            # ── Step 4: REINFORCE loss ────────────────────────────────────────
            # Mask pad positions so their log-probs don't contaminate the sum.
            pad_mask        = (samp_seqs != vocab_a.pad_idx).float()  # (B, max_len)
            masked_lp       = samp_log_probs * pad_mask               # (B, max_len)
            seq_log_probs   = masked_lp.sum(dim=1)                    # (B,)

            # L = − mean[ (r_sample − r_greedy) · Σ_t log p(y_t | …) ]
            loss = -(advantage * seq_log_probs).mean()

        # ── Step 5: Backward + grad clip + optimiser step ────────────────────
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss    += loss.item()
        total_r_samp  += r_samp.mean().item()
        total_r_greed += r_greed.mean().item()

        pbar.set_postfix({
            "loss":      f"{loss.item():.4f}",
            "r_samp":    f"{r_samp.mean().item():.3f}",
            "r_greed":   f"{r_greed.mean().item():.3f}",
            "advantage": f"{advantage.mean().item():.3f}",
        })

    n = len(loader)
    return total_loss / n, total_r_samp / n, total_r_greed / n


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments and run the SCST fine-tuning loop."""
    parser = argparse.ArgumentParser(
        description="Phase 4 SCST RL fine-tuning for VQA models C, D, E."
    )
    parser.add_argument("--model_type",       type=str,   default="E",
                        choices=["C", "D", "E"],
                        help="Model variant to fine-tune.")
    parser.add_argument("--base_checkpoint",  type=str,   required=True,
                        help="Path to fully CE/SS-trained checkpoint (.pth).")
    parser.add_argument("--lr",               type=float, default=1e-5,
                        help="Learning rate for decoder-only RL fine-tuning.")
    parser.add_argument("--epochs",           type=int,   default=3)
    parser.add_argument("--batch_size",       type=int,   default=32)
    parser.add_argument("--max_len",          type=int,   default=60)
    parser.add_argument("--num_workers",      type=int,   default=4)
    parser.add_argument("--use_coverage",     action="store_true",
                        help="Enable coverage mechanism (must match base checkpoint).")
    args = parser.parse_args()

    # ── Vocabularies ──────────────────────────────────────────────────────────
    print("Loading vocabularies…")
    vocab_q = Vocabulary()
    vocab_q.load(VOCAB_Q_PATH)
    vocab_a = Vocabulary()
    vocab_a.load(VOCAB_A_PATH)

    start_idx = vocab_a.start_idx
    end_idx   = vocab_a.end_idx

    # ── Dataset + DataLoader ──────────────────────────────────────────────────
    print("Loading training dataset…")
    train_dataset = VQAEDataset(
        image_dir=TRAIN_IMAGE_DIR,
        vqa_e_json_path=TRAIN_VQA_E_JSON,
        vocab_q=vocab_q,
        vocab_a=vocab_a,
        split="train2014",
        augment=False,  # no augmentation during RL fine-tuning
    )

    use_persistent = args.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=vqa_collate_fn,
        num_workers=args.num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"Initialising Model {args.model_type}…")
    model = get_model(
        model_type=args.model_type,
        vocab_q_size=len(vocab_q),
        vocab_a_size=len(vocab_a),
        use_coverage=args.use_coverage,
    ).to(DEVICE)

    # Load checkpoint — support both wrapped (model_state_dict key) and raw formats.
    print(f"Loading checkpoint: {args.base_checkpoint}")
    raw = torch.load(args.base_checkpoint, map_location=DEVICE)
    state_dict = raw.get("model_state_dict", raw)  # handle both formats
    model.load_state_dict(state_dict, strict=False)
    print(f"  Checkpoint loaded (strict=False — CLIP text-encoder keys expected missing).")

    # ── Optimiser — decoder parameters only ──────────────────────────────────
    use_fused = _fused_adam_available()
    adam_kw   = {"fused": True} if use_fused else {}
    optimizer = optim.Adam(model.decoder.parameters(), lr=args.lr, **adam_kw)
    print(f"  Optimiser: Adam(fused={use_fused}) — decoder params only.")

    # ── AMP setup ─────────────────────────────────────────────────────────────
    amp_dtype = torch.bfloat16 if _supports_bf16() else torch.float16
    # GradScaler is a no-op for BF16 (already numerically stable) but harmless.
    scaler = GradScaler(device=DEVICE.type, enabled=(DEVICE.type == "cuda" and amp_dtype == torch.float16))
    print(f"  AMP dtype: {amp_dtype}  |  GradScaler enabled: {scaler.is_enabled()}")

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\nStarting SCST fine-tuning…\n")
    for epoch in range(1, args.epochs + 1):
        print(f"── Epoch {epoch}/{args.epochs} ──────────────────────────────")
        avg_loss, r_samp, r_greed = train_rl_epoch(
            model=model,
            model_type=args.model_type,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            vocab_a=vocab_a,
            start_idx=start_idx,
            end_idx=end_idx,
            amp_dtype=amp_dtype,
            max_len=args.max_len,
        )
        print(
            f"Epoch {epoch} | loss={avg_loss:.4f} | "
            f"r_samp(BLEU-4)={r_samp:.4f} | r_greed(BLEU-4)={r_greed:.4f}"
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        os.makedirs("checkpoints", exist_ok=True)
        save_path = f"checkpoints/model_{args.model_type.lower()}_scst_epoch{epoch}.pth"
        torch.save(
            {
                "epoch":               epoch,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "reward_samp":         r_samp,
                "reward_greed":        r_greed,
                "args":                vars(args),
            },
            save_path,
        )
        print(f"  Saved → {save_path}")


if __name__ == "__main__":
    main()
