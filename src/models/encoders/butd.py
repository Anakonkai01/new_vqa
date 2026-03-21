"""
BUTDFeatureEncoder  — Model F baseline (geo_dim=5, 1-layer MLP).
BUTDFeatureEncoderG1 — Model G (geo_dim=7, 2-layer MLP per spec Eq 3).

Re-exported from legacy encoder_cnn.py (Step C).
G1 adds BUTDFeatureEncoderG1 here (Step D).

G1 spatial vector (7-dim, Eq 2):
  [x1/W, y1/H, x2/W, y2/H, (x2-x1)/W, (y2-y1)/H, area/(W*H)]
  Added: explicit aspect ratio dims w/W and h/H.

G1 projection (2-layer MLP, Eq 3):
  v_i = LayerNorm(ReLU(W2 · ReLU(W1 · v_raw + b1) + b2))
  W1 ∈ R^{1024×2055}, W2 ∈ R^{1024×1024}
"""

import os
import sys

import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from encoder_cnn import BUTDFeatureEncoder


class BUTDFeatureEncoderG1(nn.Module):
    """
    Model G visual feature projector.

    Differences from BUTDFeatureEncoder (Model F):
      - feat_dim=2055  (roi 2048 + geo 7)  vs. 2053 (roi 2048 + geo 5)
      - 2-layer MLP    (W1 + W2)            vs. 1-layer (W1 only)

    The extra geo dims (w/W, h/H) provide explicit aspect ratio information
    without requiring the projection to learn implicit subtraction (x2-x1).

    Input:  (B, k, 2055) — pre-extracted G1 BUTD features
    Output: (B, k, output_size)
    """

    def __init__(self, feat_dim: int = 2055, output_size: int = 1024):
        super().__init__()
        # 2-layer MLP: Linear → ReLU → Linear → ReLU → LayerNorm
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, output_size),   # W1: (feat_dim → H)
            nn.ReLU(inplace=True),
            nn.Linear(output_size, output_size), # W2: (H → H)
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_size),
        )

    def forward(self, x):
        # x: (B, k, feat_dim)
        return self.proj(x)  # (B, k, output_size)

    def unfreeze_top_layers(self):
        pass   # no pretrained backbone

    def backbone_params(self):
        return []


__all__ = ["BUTDFeatureEncoder", "BUTDFeatureEncoderG1"]
