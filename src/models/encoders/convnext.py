"""
ConvNeXtSpatialEncoder — ConvNeXt-Base spatial encoder for Model E.

Re-exported from legacy encoder_cnn.py (Step C).
Pure CNN (no transformers), rivals ViT accuracy.
Output: (B, 49, output_size) — same shape as ResNetSpatialEncoder.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from encoder_cnn import ConvNeXtSpatialEncoder

__all__ = ["ConvNeXtSpatialEncoder"]
