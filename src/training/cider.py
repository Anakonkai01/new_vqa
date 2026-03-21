import os
import json
import collections
import gc
from pycocoevalcap.cider.cider import Cider

class CIDErScorer:
    """
    Wrapper for CIDEr metric from pycocoevalcap.
    Calculates Consensus-based Image Description Evaluation.
    """
    def __init__(self, df='corpus'):
        """
        df: Document frequency parameter. Default is 'corpus'.
        """
        self.cider_scorer = Cider(df=df)

    def compute_score(self, gts, res):
        """
        gts: dict mapping image id to list of reference sentences
            {id: ["ref1", "ref2", ...], ...}
        res: dict mapping image id to list of hypothesis sentences (usually 1)
            {id: ["hyp1"], ...}
        
        Returns:
            average_score, scores
        """
        # compute cider
        score, scores = self.cider_scorer.compute_score(gts, res)
        return score, scores

def compute_cider(gts, res):
    scorer = CIDErScorer()
    return scorer.compute_score(gts, res)

if __name__ == '__main__':
    # Test
    gts = {
        'img1': ['a man playing a guitar', 'a man is playing a guitar', 'a guitar player'],
        'img2': ['a dog playing with a ball', 'a dog and a ball', 'a dog catching a ball']
    }
    res = {
        'img1': ['a man playing a guitar'],
        'img2': ['a dog playing with a frisbee']
    }
    score, scores = compute_cider(gts, res)
    print(f"CIDEr Average Score: {score}")
    print(f"CIDEr Individual Scores: {scores}")
