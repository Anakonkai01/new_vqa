#!/usr/bin/env python3
"""
RE-CREATE VQA-E_train_filtered.json from original
Using text-only filtering logic (no Faster R-CNN needed)

Filters match the secondary filters from filter_hallucinations.py:
- too_short: explanation after "because" is < 4 words
- copy_of_question: >= 80% token overlap between explanation and question
"""

import json
import re
from collections import Counter

print("="*80)
print("RECREATING VQA-E_train_filtered.json")
print("="*80)

ORIG_FILE = "data/annotations/vqa_e/VQA-E_train_set.json"
FILT_FILE = "data/annotations/vqa_e/VQA-E_train_filtered.json"
OUTPUT_FILE = "data/processed/vqae_recreated_filtered.json"

# ============================================================================
# Text-only filtering logic (from filter_hallucinations.py)
# ============================================================================

def tokenize(text):
    """Whitespace tokenizer for overlap and length checks"""
    return re.findall(r"[a-z']+", text.lower())

def is_too_short(explanation_text):
    """Return True if explanation (after 'because') is under 4 words"""
    parts = explanation_text.lower().split('because', 1)
    after_part = parts[1].strip() if len(parts) > 1 else explanation_text
    return len(tokenize(after_part)) < 4

def is_copy_of_question(explanation_text, question_text, threshold=0.8):
    """Return True if >= 80% of explanation tokens appear in question"""
    exp_tokens = set(tokenize(explanation_text))
    q_tokens = set(tokenize(question_text))
    if not exp_tokens or not q_tokens:
        return False
    overlap = len(exp_tokens & q_tokens) / len(exp_tokens)
    return overlap >= threshold

# ============================================================================
# Score annotation (from filter_hallucinations.py)
# ============================================================================

def score_annotation(ann):
    """
    Keep annotation if at least ONE explanation passes all checks.
    Remove only if ALL explanations fail.
    
    Returns: (keep: bool, reasons: list[str])
    """
    question = ann.get('question', '')
    exp_list = ann.get('explanation', [])
    valid_exps = [e for e in exp_list if isinstance(e, str) and e.strip()]
    
    # No explanation → keep unconditionally
    if not valid_exps:
        return True, []
    
    reasons_per_exp = []
    
    for exp in valid_exps:
        reasons = []
        
        # Text-only checks (no RCNN available for visual grounding)
        if is_too_short(exp):
            reasons.append('too_short')
        
        if is_copy_of_question(exp, question):
            reasons.append('copy_of_question')
        
        reasons_per_exp.append(reasons)
    
    # Keep if at least one explanation has no issues
    all_bad = all(bool(r) for r in reasons_per_exp)
    if all_bad:
        all_reasons = list({r for rs in reasons_per_exp for r in rs})
        return False, all_reasons
    return True, []

# ============================================================================
# Load and filter
# ============================================================================

print(f"\n[1] Loading original: {ORIG_FILE}")
with open(ORIG_FILE) as f:
    original = json.load(f)
print(f"    Samples: {len(original)}")

print(f"\n[2] Loading existing filtered: {FILT_FILE}")
with open(FILT_FILE) as f:
    existing_filtered = json.load(f)
print(f"    Samples: {len(existing_filtered)}")

print(f"\n[3] Applying text-only filters...")

kept = []
removed = 0
reason_counts = Counter()

for ann in original:
    keep, reasons = score_annotation(ann)
    if keep:
        kept.append(ann)
    else:
        removed += 1
        reason_counts.update(reasons)

total = len(original)
print(f"    Kept: {len(kept):,} ({len(kept)/total:.1%})")
print(f"    Removed: {removed:,} ({removed/total:.1%})")

print(f"\n[4] Removal breakdown:")
for reason, cnt in reason_counts.most_common():
    pct = cnt / total * 100
    print(f"    {reason:<25} : {cnt:>6,}  ({pct:.1f}%)")

# ============================================================================
# Compare with existing filtered
# ============================================================================

print(f"\n[5] COMPARISON WITH EXISTING FILTERED:")

kept_set = {(a.get('img_id'), tuple(a.get('question'))) for a in kept}
existing_set = {(a.get('img_id'), tuple(a.get('question'))) for a in existing_filtered}

only_in_kept = kept_set - existing_set
only_in_existing = existing_set - kept_set

print(f"    Recreated: {len(kept):,}")
print(f"    Existing:  {len(existing_filtered):,}")
print(f"    Match: {len(kept_set & existing_set):,}")
print(f"    Only in recreated: {len(only_in_kept):,}")
print(f"    Only in existing: {len(only_in_existing):,}")

if len(only_in_kept) == 0 and len(only_in_existing) == 0:
    print(f"\n    ✅ PERFECT MATCH! Recreated filter matches existing filtered.json")
else:
    match_pct = len(kept_set & existing_set) / len(kept_set) * 100 if kept else 0
    print(f"    ⚠️  {match_pct:.1f}% overlap - might have different ordering or logic")

# ============================================================================
# Save recreated version
# ============================================================================

print(f"\n[6] Saving recreated version: {OUTPUT_FILE}")
import os
os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_FILE)), exist_ok=True)
with open(OUTPUT_FILE, 'w') as f:
    json.dump(kept, f, indent=2)
print(f"    ✅ Saved")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"""
Text-only filtering produces: {len(kept):,} samples
Original filtered.json has:   {len(existing_filtered):,} samples

If MATCH (✅):
  → Use filtered.json confidently (logic verified)
  
If NO MATCH (⚠️):
  → Existing might use additional logic (e.g. RCNN-based)
  → Can either:
     a) Use recreated version (transparent, auditable)
     b) Investigate discrepancies further
""")

print("="*80)
