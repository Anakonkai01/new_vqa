"""
ResNetEncoder and ResNetSpatialEncoder — pretrained ResNet101 image encoders.

Re-exported from legacy encoder_cnn.py (Step C).
ResNetSpatialEncoder is used by Model D; not used by Model G (which uses BUTD).
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from encoder_cnn import ResNetEncoder, ResNetSpatialEncoder

__all__ = ["ResNetEncoder", "ResNetSpatialEncoder"]
