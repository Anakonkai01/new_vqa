"""
SimpleCNN and SimpleCNNSpatial — scratch CNN image encoders.

Re-exported from legacy encoder_cnn.py (Step C).
Step D will not modify these classes (they are not used by Model G which uses BUTD).
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from encoder_cnn import SimpleCNN, SimpleCNNSpatial

__all__ = ["SimpleCNN", "SimpleCNNSpatial"]
