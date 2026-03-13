"""Attention heatmap visualisation for spatial-attention models (C, D, E).

For each generated token, the decoder's dual Bahdanau attention produces
``alpha`` — a ``(49,)`` weight distribution over the 7×7 spatial image grid.
This script runs a greedy decode step-by-step, collects ``alpha`` at each
step, and overlays the attention maps on the original image.

Output
------
A single PNG with ``n_tokens + 1`` columns: the original image on the left,
then one heatmap per decoded token.

Usage
-----
    python src/visualize.py --model_type E --epoch 15
    python src/visualize.py --model_type C --epoch 10 --sample_idx 5
    python src/visualize.py --model_type D --epoch 10 --output results/attn.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # headless — no display server required (Kaggle / SSH)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(__file__))

from inference import _encode_spatial, load_model_from_checkpoint
from vocab import Vocabulary


# ── Configuration ─────────────────────────────────────────────────────────────

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_DIR    = "data/raw/train2014"
VQA_E_JSON   = "data/vqa_e/VQA-E_train_set.json"
VOCAB_Q_PATH = "data/processed/vocab_questions.json"
VOCAB_A_PATH = "data/processed/vocab_answers.json"

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation and return a ``(H, W, 3)`` float32 array in [0, 1].

    Args:
        tensor: ``FloatTensor (3, H, W)`` — normalised image tensor.

    Returns:
        ``ndarray (H, W, 3)`` clipped to ``[0, 1]``.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


# ── Core decode-with-attention ────────────────────────────────────────────────

def decode_with_attention_steps(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    question_tensor: torch.Tensor,
    vocab_a: Vocabulary,
    max_len: int = 20,
    device: Union[torch.device, str] = "cpu",
) -> Tuple[List[str], List[np.ndarray]]:
    """Greedy decode with per-step attention weight collection.

    Handles both GatedFusion/concat (C, D) and FiLM (E) encoder init via
    the shared ``_encode_spatial`` helper.

    Args:
        model: ``VQAModelC``, ``VQAModelD``, or ``VQAModelE`` in eval mode.
        image_tensor: ``FloatTensor (3, 224, 224)``.
        question_tensor: ``LongTensor (Q,)``.
        vocab_a: Answer vocabulary.
        max_len: Maximum tokens to decode.
        device: Device for computation.

    Returns:
        Tuple of:
            tokens – List of decoded word strings (content tokens only).
            alphas – List of ``ndarray (49,)`` — image attention per token step.
    """
    with torch.no_grad():
        # _encode_spatial handles both Model E (FiLM) and C/D (gate/concat).
        encoder_hidden, spatial_feats, q_hidden = _encode_spatial(
            model,
            image_tensor.unsqueeze(0),
            question_tensor.unsqueeze(0),
            device,
        )
        # spatial_feats: (1, 49, H) | q_hidden: (1, Q, H)

        hidden   = encoder_hidden
        token    = torch.tensor([[vocab_a.start_idx]], dtype=torch.long, device=device)
        coverage: Optional[torch.Tensor] = None

        tokens: List[str]       = []
        alphas: List[np.ndarray] = []

        for _ in range(max_len):
            logit, hidden, alpha, coverage = model.decoder.decode_step(
                token, hidden, spatial_feats, q_hidden, coverage
            )
            # logit: (1, vocab_size) | alpha: (1, 49)
            pred = logit.argmax(dim=-1).item()

            if pred == vocab_a.end_idx:
                break

            tokens.append(vocab_a.idx2word.get(pred, "<unk>"))
            alphas.append(alpha.squeeze(0).cpu().numpy())   # (49,)
            token = torch.tensor([[pred]], dtype=torch.long, device=device)

    return tokens, alphas


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualize_attention(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    question_text: str,
    question_tensor: torch.Tensor,
    vocab_a: Vocabulary,
    output_path: str,
    device: Union[torch.device, str] = "cpu",
    max_len: int = 20,
) -> None:
    """Generate and save an attention heatmap figure for one sample.

    Each attention map is:
    1. Reshaped from ``(49,)`` → ``(7, 7)``.
    2. Min-max normalised to ``[0, 1]``.
    3. Upsampled to ``(224, 224)`` via bilinear interpolation.
    4. Blended with the original image using ``alpha=0.5`` and the ``jet`` colormap.

    Args:
        model: Spatial-attention VQA model.
        image_tensor: ``FloatTensor (3, 224, 224)`` — normalised image.
        question_text: Original question string (used for the figure title).
        question_tensor: ``LongTensor (Q,)`` — tokenised question.
        vocab_a: Answer vocabulary.
        output_path: Path to save the output PNG.
        device: Device for computation.
        max_len: Maximum tokens to decode.
    """
    tokens, alphas = decode_with_attention_steps(
        model, image_tensor, question_tensor, vocab_a,
        max_len=max_len, device=device,
    )

    if not alphas:
        print("No tokens were decoded — cannot produce attention visualisation.")
        return

    img_np  = _denormalize(image_tensor)   # (224, 224, 3)
    n_cols  = len(tokens) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3.5))

    axes[0].imshow(img_np)
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis("off")

    for i, (word, alpha) in enumerate(zip(tokens, alphas)):
        # alpha: (49,) → (7, 7) → normalise → upsample to (224, 224)
        attn_map = alpha.reshape(7, 7)
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        attn_up  = np.array(
            Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
                (224, 224), Image.BILINEAR
            )
        ) / 255.0  # (224, 224)

        axes[i + 1].imshow(img_np)
        axes[i + 1].imshow(attn_up, alpha=0.5, cmap="jet")
        axes[i + 1].set_title(f'"{word}"', fontsize=9)
        axes[i + 1].axis("off")

    answer = " ".join(tokens)
    fig.suptitle(f"Q: {question_text}\nA: {answer}", fontsize=10, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise per-token attention heatmaps for spatial-attention models."
    )
    parser.add_argument("--model_type", type=str, default="E",
                        choices=["C", "D", "E"],
                        help="Model type (C, D, or E — all have spatial attention).")
    parser.add_argument("--epoch",      type=int, default=10,
                        help="Epoch checkpoint to load.")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Index of the VQA-E annotation to visualise.")
    parser.add_argument("--output",     type=str, default=None,
                        help="Output PNG path. Default: checkpoints/attn_model_X.png")
    args = parser.parse_args()

    checkpoint  = f"checkpoints/model_{args.model_type.lower()}_epoch{args.epoch}.pth"
    output_path = args.output or f"checkpoints/attn_model_{args.model_type.lower()}.png"

    if not os.path.exists(checkpoint):
        # Fallback to best checkpoint
        best_ckpt = f"checkpoints/model_{args.model_type.lower()}_best.pth"
        if os.path.exists(best_ckpt):
            print(f"[INFO] {checkpoint} not found — using {best_ckpt}")
            checkpoint = best_ckpt
        else:
            print(f"Checkpoint not found: {checkpoint}")
            sys.exit(1)

    vocab_q = Vocabulary()
    vocab_q.load(VOCAB_Q_PATH)
    vocab_a = Vocabulary()
    vocab_a.load(VOCAB_A_PATH)

    model = load_model_from_checkpoint(
        args.model_type, checkpoint, len(vocab_q), len(vocab_a), device=DEVICE
    )

    with open(VQA_E_JSON) as f:
        annotations = json.load(f)

    sample   = annotations[args.sample_idx]
    q_text   = sample["question"]
    img_id   = sample["img_id"]
    img_path = os.path.join(IMAGE_DIR, f"COCO_train2014_{img_id:012d}.jpg")

    img_tensor = _TRANSFORM(Image.open(img_path).convert("RGB"))
    q_tensor   = torch.tensor(vocab_q.numericalize(q_text), dtype=torch.long)

    visualize_attention(
        model, img_tensor, q_text, q_tensor, vocab_a,
        output_path=output_path, device=DEVICE,
    )
    print(f"Question : {q_text}")
    print(f"Output   : {output_path}")
