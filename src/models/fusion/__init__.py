"""
src/models/fusion — multimodal fusion modules.

Re-exports from legacy files (Step C).
"""

from .gated import GatedFusion
from .mutan import MUTANFusion

__all__ = [
    "GatedFusion",
    "MUTANFusion",
]
