"""Helpers for the Model H answer/explanation output contract."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple


_BECAUSE_RE = re.compile(r"\bbecause\b", flags=re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Collapse whitespace and trim."""
    return " ".join(str(text or "").strip().split())


def split_answer_explanation(text: str) -> Tuple[str, str]:
    """
    Split a generated or target string into:
      - answer prefix
      - explanation suffix after the first standalone 'because'

    Contract:
      "motorcycle because the bike is on a stand"
          -> ("motorcycle", "the bike is on a stand")
      "because there are many birds"
          -> ("", "there are many birds")
      "motorcycle"
          -> ("motorcycle", "")
    """
    text = normalize_text(text)
    if not text:
        return "", ""

    match = _BECAUSE_RE.search(text)
    if match is None:
        return text, ""

    answer = normalize_text(text[:match.start()])
    explanation = normalize_text(text[match.end():])
    return answer, explanation


def split_answer_explanations(texts: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Vectorized wrapper over split_answer_explanation()."""
    answers: List[str] = []
    explanations: List[str] = []
    for text in texts:
        answer, explanation = split_answer_explanation(text)
        answers.append(answer)
        explanations.append(explanation)
    return answers, explanations


def strip_empty_explanations(
    explanations: Iterable[str],
    *,
    fallback_token: str = "<empty>",
) -> List[str]:
    """
    Replace empty explanations with a sentinel token so downstream metrics
    keep a stable candidate/reference interface.
    """
    out: List[str] = []
    for text in explanations:
        norm = normalize_text(text)
        out.append(norm if norm else fallback_token)
    return out
