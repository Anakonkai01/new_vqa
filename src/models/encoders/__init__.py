"""
src/models/encoders — visual and question encoder modules.

Re-exports from legacy files (Step C): old files untouched, new structure established here.
Step D will add G-specific variants (e.g., BUTDFeatureEncoderG1 with geo_dim=7) in these files.
"""

from .cnn import SimpleCNN, SimpleCNNSpatial
from .resnet import ResNetEncoder, ResNetSpatialEncoder
from .convnext import ConvNeXtSpatialEncoder
from .butd import BUTDFeatureEncoder
from .question import QuestionEncoder

__all__ = [
    "SimpleCNN",
    "SimpleCNNSpatial",
    "ResNetEncoder",
    "ResNetSpatialEncoder",
    "ConvNeXtSpatialEncoder",
    "BUTDFeatureEncoder",
    "QuestionEncoder",
]
