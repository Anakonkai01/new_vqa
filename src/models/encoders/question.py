"""
QuestionEncoder — BiLSTM question encoder with optional GloVe init.

Re-exported from legacy encoder_question.py (Step C).
Shared across all models A-G.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from encoder_question import QuestionEncoder

__all__ = ["QuestionEncoder"]
