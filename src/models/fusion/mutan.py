"""
MUTANFusion — Multimodal Tucker Fusion for Models E/F/G.

Re-exported from legacy vqa_models.py (Step C).
Captures multiplicative cross-modal interactions via Tucker decomposition.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vqa_models import MUTANFusion

__all__ = ["MUTANFusion"]
