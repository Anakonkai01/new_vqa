"""
CIDEr-D for SCST rewards (training/scst.py).

pycocoevalcap.cider.Cider API (no `df=` kwarg): constructor takes optional
test/refs corpora; compute_score(gts, res) computes per-hypothesis scores.
"""
from typing import Dict, List, Tuple

from pycocoevalcap.cider.cider import Cider


def compute_cider(
    gts: Dict[str, List[str]],
    res: Dict[str, List[str]],
) -> Tuple[float, List[float]]:
    """
    gts, res: {str(id): [sentence, ...]} as expected by pycocoevalcap CIDEr.
    Returns (mean_score, list of per-item scores).
    """
    cider = Cider()
    return cider.compute_score(gts, res)
