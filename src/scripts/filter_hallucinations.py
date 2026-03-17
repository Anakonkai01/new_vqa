"""
filter_hallucinations.py — Tier D5
===================================
Filters VQA-E annotation JSON to remove explanations that are likely
hallucinated (not grounded in the visual scene).

DETECTION HEURISTICS (no image required — text-only):
  1. Named Entity hallucinations: Named entities (PERSON, ORG, GPE, EVENT)
     in explanations are almost always hallucinated — VQA-E explanations
     should describe visual content, not named people/places/organizations.
  2. Length filter: explanations < 4 words after "because" are too short
     to be meaningful explanations.
  3. Copy-paste filter: explanation that is ≥ 80% token overlap with the
     question (the model copied the question rather than explaining).
  4. Repetition filter: explanation has ≥ 50% repeated n-grams (degenerate).

Output: filtered JSON + a report of how many samples were removed per heuristic.

Usage:
  pip install spacy
  python -m spacy download en_core_web_sm
  python src/scripts/filter_hallucinations.py \\
      --input  data/vqa_e/VQA-E_train_set.json \\
      --output data/vqa_e/VQA-E_train_filtered.json \\
      --report
"""

import json
import argparse
import re
from collections import Counter

try:
    import spacy
    _NLP = None   # lazy-loaded
    def _get_nlp():
        global _NLP
        if _NLP is None:
            _NLP = spacy.load('en_core_web_sm')
        return _NLP
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


# Named entity types that signal hallucinations in visual explanation context
_HALLUCINATION_ENT_TYPES = {'PERSON', 'ORG', 'GPE', 'EVENT', 'WORK_OF_ART', 'LAW', 'NORP'}

# Minimum explanation word count after "because"
_MIN_EXPLANATION_WORDS = 4


def _tokenize(text):
    """Simple whitespace tokenizer — used for overlap and repetition checks."""
    return re.findall(r"[a-z']+", text.lower())


def has_named_entity_hallucination(explanation_text, nlp):
    """Return True if the explanation contains a hallucinated named entity."""
    doc = nlp(explanation_text)
    for ent in doc.ents:
        if ent.label_ in _HALLUCINATION_ENT_TYPES:
            return True
    return False


def is_too_short(explanation_text):
    """Return True if the explanation (after 'because') is too short."""
    parts = explanation_text.lower().split('because', 1)
    after_because = parts[1].strip() if len(parts) > 1 else explanation_text
    return len(_tokenize(after_because)) < _MIN_EXPLANATION_WORDS


def is_copy_of_question(explanation_text, question_text, threshold=0.8):
    """Return True if explanation heavily overlaps with the question."""
    exp_tokens = set(_tokenize(explanation_text))
    q_tokens   = set(_tokenize(question_text))
    if not exp_tokens or not q_tokens:
        return False
    overlap = len(exp_tokens & q_tokens) / len(exp_tokens)
    return overlap >= threshold


def has_degenerate_repetition(explanation_text, threshold=0.5):
    """Return True if >50% of bigrams are repeated (degenerate output)."""
    tokens = _tokenize(explanation_text)
    if len(tokens) < 4:
        return False
    bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
    counts  = Counter(bigrams)
    repeated = sum(cnt for cnt in counts.values() if cnt > 1)
    return repeated / len(bigrams) >= threshold


def score_annotation(ann, nlp=None):
    """
    Score a single annotation. Returns (keep: bool, reasons: list[str]).
    reasons is non-empty only when keep=False.
    """
    question    = ann.get('question', '')
    exp_list    = ann.get('explanation', [])
    valid_exps  = [e for e in exp_list if isinstance(e, str) and e.strip()]

    # No explanation → keep (VQA-E may have empty explanations)
    if not valid_exps:
        return True, []

    # Check every explanation — if ALL are bad, remove the annotation
    reasons_per_exp = []
    for exp in valid_exps:
        reasons = []
        if is_too_short(exp):
            reasons.append('too_short')
        if is_copy_of_question(exp, question):
            reasons.append('copy_of_question')
        if has_degenerate_repetition(exp):
            reasons.append('degenerate_repetition')
        if SPACY_AVAILABLE and nlp is not None:
            if has_named_entity_hallucination(exp, nlp):
                reasons.append('named_entity')
        reasons_per_exp.append(reasons)

    # Keep if at least one explanation passes all checks
    all_bad = all(bool(r) for r in reasons_per_exp)
    if all_bad:
        # Collect all unique reasons across all explanations
        all_reasons = list({r for rs in reasons_per_exp for r in rs})
        return False, all_reasons
    return True, []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help='Input VQA-E JSON path')
    parser.add_argument('--output', required=True, help='Output filtered JSON path')
    parser.add_argument('--report', action='store_true', help='Print per-heuristic counts')
    parser.add_argument('--no_ner', action='store_true',
                        help='Skip spaCy NER check (faster, no spaCy needed)')
    args = parser.parse_args()

    if not SPACY_AVAILABLE and not args.no_ner:
        print("spaCy not installed. Run: pip install spacy && python -m spacy download en_core_web_sm")
        print("Or use --no_ner to skip NER-based filtering.")
        raise SystemExit(1)

    nlp = None
    if SPACY_AVAILABLE and not args.no_ner:
        print("Loading spaCy en_core_web_sm ...")
        nlp = _get_nlp()

    print(f"Loading {args.input} ...")
    with open(args.input) as f:
        annotations = json.load(f)

    kept    = []
    removed = 0
    reason_counts = Counter()

    for ann in annotations:
        keep, reasons = score_annotation(ann, nlp)
        if keep:
            kept.append(ann)
        else:
            removed += 1
            reason_counts.update(reasons)

    print(f"Original  : {len(annotations):,}")
    print(f"Kept      : {len(kept):,}  ({len(kept)/len(annotations):.1%})")
    print(f"Removed   : {removed:,}  ({removed/len(annotations):.1%})")
    if args.report:
        print("\nRemoval reasons:")
        for reason, cnt in reason_counts.most_common():
            print(f"  {reason:<25} : {cnt:,}")

    import os
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(kept, f, indent=2)
    print(f"\nSaved filtered annotations → {args.output}")


if __name__ == '__main__':
    main()
