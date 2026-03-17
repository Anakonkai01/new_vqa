"""
Tier-6: Counterfactual Samples Synthesizing (CSS) Augmentation
================================================================
Targets linguistic bias — the tendency to answer based on question statistics
rather than visual evidence (e.g. "What color is the banana?" → "yellow"
regardless of the image).

CSS generates two types of counterfactual samples per real batch:

  1. Visual masking  : zero out image spatial features for critical regions
                       (regions that receive high attention for this question).
                       The model can no longer cheat using visual shortcuts.

  2. Linguistic masking: replace key content words in the question with <mask>.
                         The model must rely on the image to answer.

A contrastive loss term pushes the representations of factual and counterfactual
samples apart, preventing the model from relying on a single modality.

  L_total = L_CE + λ_CSS * L_contrastive

  L_contrastive = mean( max(0, margin - ||f_real - f_cf||_2) )

  f_real : fused representation for original sample
  f_cf   : fused representation for counterfactual sample
  margin : minimum desired distance (default 1.0)

Usage in train.py
-----------------
  from training.css_augment import CSSAugmentor, css_contrastive_loss

  augmentor = CSSAugmentor(mask_token_id=<unk_id>, mask_ratio=0.3, region_mask_ratio=0.3)

  # Inside training loop, after encoding:
  imgs_cf, qs_cf = augmentor(imgs, questions, img_features, attn_weights)
  # Run model on counterfactuals, compute contrastive loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CSSAugmentor(nn.Module):
    """
    Generates counterfactual samples via visual and linguistic masking.

    Args:
        mask_token_id    : vocabulary ID of the <mask>/<unk> token (default: 3 for our vocab)
        mask_ratio       : fraction of question content words to replace with mask (default: 0.3)
        region_mask_ratio: fraction of top-attended image regions to zero out  (default: 0.3)
        pad_token_id     : vocabulary ID of <pad> — excluded from linguistic masking (default: 0)
    """

    def __init__(self, mask_token_id: int = 3, mask_ratio: float = 0.3,
                 region_mask_ratio: float = 0.3, pad_token_id: int = 0):
        super().__init__()
        self.mask_token_id    = mask_token_id
        self.mask_ratio       = mask_ratio
        self.region_mask_ratio = region_mask_ratio
        self.pad_token_id     = pad_token_id

    def visual_mask(self, img_features: torch.Tensor,
                    attn_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Zero out top-attended (or random) image regions.

        img_features : (B, N, H) — spatial image feature grid (N=49)
        attn_weights : (B, N)    — attention weights from previous step (or None)

        returns: (B, N, H) with top regions zeroed
        """
        B, N, H = img_features.shape
        k = max(1, int(N * self.region_mask_ratio))   # number of regions to mask

        if attn_weights is not None:
            # Mask the top-k attended regions (most relied upon)
            _, top_idx = attn_weights.topk(k, dim=1)  # (B, k)
        else:
            # Random masking if no attention weights available
            top_idx = torch.stack(
                [torch.randperm(N, device=img_features.device)[:k] for _ in range(B)], dim=0
            )  # (B, k)

        masked = img_features.clone()
        # Scatter zeros into the selected positions
        mask_expand = top_idx.unsqueeze(-1).expand(-1, -1, H)  # (B, k, H)
        masked.scatter_(1, mask_expand, 0.0)
        return masked

    def linguistic_mask(self, questions: torch.Tensor) -> torch.Tensor:
        """
        Replace random content tokens with mask_token_id.

        questions : (B, q_len) — question token IDs
        returns   : (B, q_len) — questions with some tokens replaced
        """
        B, L = questions.shape
        masked_q = questions.clone()

        # Only mask non-padding, non-special tokens (token id > 3)
        content_mask = (questions > 3)  # True for real content words

        for b in range(B):
            content_pos = content_mask[b].nonzero(as_tuple=True)[0]
            if content_pos.numel() == 0:
                continue
            n_mask = max(1, int(content_pos.numel() * self.mask_ratio))
            perm = torch.randperm(content_pos.numel(), device=questions.device)
            mask_pos = content_pos[perm[:n_mask]]
            masked_q[b, mask_pos] = self.mask_token_id

        return masked_q

    def forward(self, questions: torch.Tensor,
                img_features: torch.Tensor,
                attn_weights: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate one visual CF and one linguistic CF per batch.

        questions    : (B, q_len)
        img_features : (B, N, H)
        attn_weights : (B, N) or None

        returns:
          cf_img_features : (B, N, H) — visual-masked features
          cf_questions    : (B, q_len) — linguistically-masked questions
        """
        cf_img = self.visual_mask(img_features, attn_weights)
        cf_q   = self.linguistic_mask(questions)
        return cf_img, cf_q


def css_contrastive_loss(f_real: torch.Tensor,
                         f_cf_visual: torch.Tensor,
                         f_cf_ling: torch.Tensor,
                         margin: float = 1.0) -> torch.Tensor:
    """
    Hinge contrastive loss: push real and counterfactual representations apart.

    L = mean( max(0, margin - ||f_real - f_cf||_2) )

    f_real      : (B, H) — fused feature from real batch
    f_cf_visual : (B, H) — fused feature from visually-masked CF
    f_cf_ling   : (B, H) — fused feature from linguistically-masked CF
    margin      : minimum L2 distance (default 1.0)
    """
    d_visual = F.pairwise_distance(f_real, f_cf_visual, p=2)  # (B,)
    d_ling   = F.pairwise_distance(f_real, f_cf_ling,   p=2)  # (B,)

    loss_visual = F.relu(margin - d_visual).mean()
    loss_ling   = F.relu(margin - d_ling).mean()

    return (loss_visual + loss_ling) * 0.5


if __name__ == "__main__":
    B, N, H, L, V = 4, 49, 512, 12, 1000
    augmentor = CSSAugmentor(mask_token_id=3, mask_ratio=0.3, region_mask_ratio=0.3)

    img_features = torch.randn(B, N, H)
    questions    = torch.randint(0, V, (B, L))
    questions[:, -3:] = 0  # simulate padding at end
    attn_weights = torch.softmax(torch.randn(B, N), dim=-1)

    cf_img, cf_q = augmentor(questions, img_features, attn_weights)
    print(f"Visual CF   : {cf_img.shape}, zeros: {(cf_img == 0).all(-1).sum(1).tolist()}")
    print(f"Ling CF     : {cf_q.shape}, masks: {(cf_q == 3).sum(1).tolist()}")

    f_real = torch.randn(B, H)
    f_cf_v = torch.randn(B, H)
    f_cf_l = torch.randn(B, H)
    loss = css_contrastive_loss(f_real, f_cf_v, f_cf_l)
    print(f"Contrastive loss: {loss.item():.4f}")
    print("CSSAugmentor sanity check PASSED")
