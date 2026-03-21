"""
GatedFusion — learnable gated multimodal fusion (Models A-D).

Re-exported from legacy vqa_models.py (Step C).
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vqa_models import GatedFusion

__all__ = ["GatedFusion"]
