"""
InfoNCEProjectionHeads — G3 contrastive visual-language alignment.

Projects (v_bar, q_feat) into shared 256-dim L2-normalized space,
then computes symmetric InfoNCE loss (Eq 35-36).

Training only — this module is NOT used at inference.
Parameter count: 2 × Linear(1024, 256) = 2 × 262,144 = 524,288 params.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCEProjectionHeads(nn.Module):
    """
    Dual linear projection heads for visual-language contrastive alignment.

    Architecture (per spec Eq 35):
        z_img  = L2_norm(W_proj_i · v_bar)   ∈ R^{z_dim}
        z_text = L2_norm(W_proj_t · q_feat)  ∈ R^{z_dim}

    where:
        v_bar  = masked mean of BUTD region features (B, H)
        q_feat = attention-pooled BiLSTM question representation (B, H)

    These are stored in VQAOutput.infonce_z and consumed by infonce_loss().

    Args:
        img_dim  : BUTD projected feature dimension (default 1024)
        text_dim : BiLSTM q_feat dimension (default 1024)
        z_dim    : shared contrastive embedding dimension (default 256)
        tau      : InfoNCE temperature (default 0.07)
    """

    def __init__(
        self,
        img_dim: int = 1024,
        text_dim: int = 1024,
        z_dim: int = 256,
        tau: float = 0.07,
    ):
        super().__init__()
        self.proj_img = nn.Linear(img_dim, z_dim, bias=False)
        self.proj_txt = nn.Linear(text_dim, z_dim, bias=False)
        self.tau      = tau

    def forward(
        self,
        v_bar: torch.Tensor,
        q_feat: torch.Tensor,
    ):
        """
        Args:
            v_bar  : (B, img_dim) — masked mean BUTD features
            q_feat : (B, text_dim) — attention-pooled question features

        Returns:
            z_img  : (B, z_dim) L2-normalized image embeddings
            z_text : (B, z_dim) L2-normalized text embeddings
        """
        z_img  = F.normalize(self.proj_img(v_bar),   dim=-1)  # (B, 256)
        z_text = F.normalize(self.proj_txt(q_feat),  dim=-1)  # (B, 256)
        return z_img, z_text


def infonce_loss(
    z_img: torch.Tensor,
    z_text: torch.Tensor,
    tau: float = 0.07,
) -> torch.Tensor:
    """
    Symmetric InfoNCE (NT-Xent) loss (Eq 36).

        L = (L_{i→t} + L_{t→i}) / 2

    where L_{i→t} = CrossEntropy(sim(z_img, z_text^T) / tau).

    In-batch negatives: sample i is positive for itself, negative for all j≠i.

    Args:
        z_img  : (B, z_dim) L2-normalized image embeddings
        z_text : (B, z_dim) L2-normalized text embeddings
        tau    : temperature

    Returns:
        scalar loss tensor
    """
    B       = z_img.size(0)
    sim     = torch.matmul(z_img, z_text.T) / tau   # (B, B)
    labels  = torch.arange(B, device=z_img.device)
    L_i2t   = F.cross_entropy(sim,   labels)
    L_t2i   = F.cross_entropy(sim.T, labels)
    return (L_i2t + L_t2i) * 0.5
