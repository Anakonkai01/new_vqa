"""
src/models/decoders — LSTM decoder modules.

Re-exports from legacy files (Step C).
Step D will wire G2 (3-way PGN) and G5 (length embedding) into LSTMDecoderWithAttention.
"""

from .lstm import LSTMDecoder
from .attention import LSTMDecoderWithAttention

__all__ = [
    "LSTMDecoder",
    "LSTMDecoderWithAttention",
]
