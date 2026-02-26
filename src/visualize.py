"""
visualize.py — Visualize attention heatmaps for Model C and D.

For each generated token, the decoder attention computes alpha (batch, 49),
a weight distribution over 49 image regions (7x7 grid). This script:
  1. Runs greedy decode step by step
  2. Collects alpha at each step
  3. Overlays attention heatmaps on the original image for each token

Usage:
    python src/visualize.py --model_type C
    python src/visualize.py --model_type D --epoch 5
    python src/visualize.py --model_type C --output results/attn.png
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(__file__))
from vocab import Vocabulary
from inference import get_model, greedy_decode
from models.vqa_models import hadamard_fusion

# ── Config ───────────────────────────────────────────────────────
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_DIR       = "data/raw/images/train2014"
QUESTION_JSON   = "data/raw/vqa_json/v2_OpenEnded_mscoco_train2014_questions.json"
VOCAB_Q_PATH    = "data/processed/vocab_questions.json"
VOCAB_A_PATH    = "data/processed/vocab_answers.json"


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def denormalize(tensor):
    """Convert normalized image tensor back to [0,1] range for display."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def decode_with_attention_steps(model, image_tensor, question_tensor,
                                vocab_a, max_len=20, device='cpu'):
    """
    Run greedy decode step by step, collecting alpha (attention weights) at each step.
    Returns:
        tokens : list of predicted token strings
        alphas : list of (49,) numpy arrays — attention map per step
    """
    with torch.no_grad():
        img      = image_tensor.unsqueeze(0).to(device)
        question = question_tensor.unsqueeze(0).to(device)

        img_features  = model.i_encoder(img)                   # (1, 49, 1024)
        img_features  = F.normalize(img_features, p=2, dim=-1)
        question_feat = model.q_encoder(question)              # (1, 1024)

        img_mean = img_features.mean(dim=1)                    # (1, 1024)
        fusion   = hadamard_fusion(img_mean, question_feat)    # (1, 1024)

        h_0 = fusion.unsqueeze(0).repeat(model.num_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)
        hidden = (h_0, c_0)

        start_idx = vocab_a.word2idx['<start>']
        end_idx   = vocab_a.word2idx['<end>']
        token     = torch.tensor([[start_idx]], dtype=torch.long).to(device)

        tokens = []
        alphas = []

        for _ in range(max_len):
            logit, hidden, alpha = model.decoder.decode_step(token, hidden, img_features)
            pred = logit.argmax(dim=-1).item()

            if pred == end_idx:
                break

            word = vocab_a.idx2word.get(pred, '<unk>')
            tokens.append(word)
            alphas.append(alpha.squeeze(0).cpu().numpy())   # (49,)
            token = torch.tensor([[pred]], dtype=torch.long).to(device)

    return tokens, alphas


def visualize_attention(model, image_tensor, original_image, question_text,
                        vocab_a, output_path, device='cpu'):
    """
    Draw the original image + per-token attention heatmaps.
    """
    tokens, alphas = decode_with_attention_steps(
        model, image_tensor, None, vocab_a, device=device
    )

    # fallback: if model has no attention (A/B), skip visualization
    if not alphas:
        print("This model does not have attention or no tokens were generated.")
        return

    n_tokens  = len(tokens)
    img_np    = denormalize(image_tensor)   # (224, 224, 3)

    # 1 column for original image + n_tokens columns for heatmaps
    n_cols = n_tokens + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3.5))

    # original image
    axes[0].imshow(img_np)
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis('off')

    # per-token heatmaps
    for i, (word, alpha) in enumerate(zip(tokens, alphas)):
        # alpha: (49,) -> reshape (7, 7) -> upsample to (224, 224)
        attn_map = alpha.reshape(7, 7)
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        attn_up  = np.array(Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
            (224, 224), Image.BILINEAR)) / 255.0   # (224, 224)

        axes[i + 1].imshow(img_np)
        axes[i + 1].imshow(attn_up, alpha=0.5, cmap='jet')
        axes[i + 1].set_title(f'"{word}"', fontsize=9)
        axes[i + 1].axis('off')

    answer = ' '.join(tokens)
    fig.suptitle(f'Q: {question_text}\nA: {answer}', fontsize=10, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


# ── wrapper to pass question_tensor into decode_with_attention_steps ─────────
def _decode_wrapper(model, image_tensor, question_tensor, vocab_a, device='cpu'):
    """Wrapper matching the signature expected by visualize_attention."""
    with torch.no_grad():
        img      = image_tensor.unsqueeze(0).to(device)
        question = question_tensor.unsqueeze(0).to(device)

        img_features  = model.i_encoder(img)
        img_features  = F.normalize(img_features, p=2, dim=-1)
        question_feat = model.q_encoder(question)

        img_mean = img_features.mean(dim=1)
        fusion   = hadamard_fusion(img_mean, question_feat)

        h_0 = fusion.unsqueeze(0).repeat(model.num_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)
        hidden = (h_0, c_0)

        start_idx = vocab_a.word2idx['<start>']
        end_idx   = vocab_a.word2idx['<end>']
        token     = torch.tensor([[start_idx]], dtype=torch.long).to(device)

        tokens = []
        alphas = []

        for _ in range(20):
            logit, hidden, alpha = model.decoder.decode_step(token, hidden, img_features)
            pred = logit.argmax(dim=-1).item()
            if pred == end_idx:
                break
            word = vocab_a.idx2word.get(pred, '<unk>')
            tokens.append(word)
            alphas.append(alpha.squeeze(0).cpu().numpy())
            token = torch.tensor([[pred]], dtype=torch.long).to(device)

    return tokens, alphas


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='C', choices=['C', 'D'],
                        help='Only C and D have attention')
    parser.add_argument('--epoch',      type=int, default=10)
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Index of the sample in the question JSON')
    parser.add_argument('--output',     type=str, default=None,
                        help='Output path (default: checkpoints/attn_model_X.png)')
    args = parser.parse_args()

    output = args.output or f"checkpoints/attn_model_{args.model_type.lower()}.png"
    checkpoint = f"checkpoints/model_{args.model_type.lower()}_epoch{args.epoch}.pth"

    if not os.path.exists(checkpoint):
        print(f"Checkpoint not found: {checkpoint}")
        sys.exit(1)

    # load vocab
    vocab_q = Vocabulary(); vocab_q.load(VOCAB_Q_PATH)
    vocab_a = Vocabulary(); vocab_a.load(VOCAB_A_PATH)

    # load model
    model = get_model(args.model_type, len(vocab_q), len(vocab_a))
    model.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage))
    model.to(DEVICE)
    model.eval()

    # load sample
    with open(QUESTION_JSON, 'r') as f:
        questions = json.load(f)['questions']

    sample    = questions[args.sample_idx]
    q_text    = sample['question']
    img_id    = sample['image_id']
    img_path  = os.path.join(IMAGE_DIR, f"COCO_train2014_{img_id:012d}.jpg")

    transform     = get_transform()
    original_img  = Image.open(img_path).convert("RGB")
    img_tensor    = transform(original_img)
    q_tensor      = torch.tensor(vocab_q.numericalize(q_text), dtype=torch.long)

    tokens, alphas = _decode_wrapper(model, img_tensor, q_tensor, vocab_a, device=DEVICE)

    if not alphas:
        print("No tokens were decoded.")
        sys.exit(1)

    img_np    = denormalize(img_tensor)
    n_tokens  = len(tokens)
    n_cols    = n_tokens + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3.5))

    axes[0].imshow(img_np)
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis('off')

    for i, (word, alpha) in enumerate(zip(tokens, alphas)):
        attn_map = alpha.reshape(7, 7)
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        attn_up  = np.array(Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
            (224, 224), Image.BILINEAR)) / 255.0

        axes[i + 1].imshow(img_np)
        axes[i + 1].imshow(attn_up, alpha=0.5, cmap='jet')
        axes[i + 1].set_title(f'"{word}"', fontsize=9)
        axes[i + 1].axis('off')

    answer = ' '.join(tokens)
    fig.suptitle(f'Model {args.model_type} | Q: {q_text}\nA: {answer}',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else '.', exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Question : {q_text}")
    print(f"Answer   : {answer}")
    print(f"Saved    : {output}")
    plt.close()
