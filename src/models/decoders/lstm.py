"""
LSTMDecoder — attention-free LSTM decoder for Models A/B.

Re-exported from legacy decoder_lstm.py (Step C).
Also re-exports Tier-1 helpers used by LSTMDecoderWithAttention.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from decoder_lstm import LSTMDecoder, LayerNormLSTMStack, WeightDrop, HighwayLayer

__all__ = ["LSTMDecoder", "LayerNormLSTMStack", "WeightDrop", "HighwayLayer"]
