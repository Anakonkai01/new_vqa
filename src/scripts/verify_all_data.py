#!/usr/bin/env python3
"""
VERIFICATION PHASE: Inspect all 3 data sources before merging
- Check formats, structures, data types
- Sample inspection (spot-check)
- Count statistics
- Identify potential issues
"""

import json
import os
from collections import defaultdict, Counter

print("\n" + "="*80)
print("DATA VERIFICATION PHASE - Before Merging")
print("="*80)

# ============================================================================
# PART 1: VQA-X Inspection
# ============================================================================
print("\n" + "-"*80)
print("1. VQA-X INSPECTION")
print("-"*80)

VQA_X_FILE = "data/annotations/vqa_x/train_x.json"

if not os.path.exists(VQA_X_FILE):
    print(f"❌ File not found: {VQA_X_FILE}")
else:
    print(f"✅ File exists: {VQA_X_FILE}")
    
    with open(VQA_X_FILE) as f:
        vqa_x = json.load(f)
    
    print(f"\n  Type: {type(vqa_x).__name__}")
    print(f"  Total samples: {len(vqa_x)}")
    
    if isinstance(vqa_x, list) and len(vqa_x) > 0:
        print(f"\n  Sample structure (first 2 fields):")
        sample = vqa_x[0]
        for key in list(sample.keys())[:2]:
            val = sample[key]
            if isinstance(val, str):
                val = val[:50] + "..." if len(val) > 50 else val
            print(f"    - {key}: {val}")
        
        # Stats
        questions = [s.get('question', '') for s in vqa_x]
        explanations = [s.get('explanation', []) for s in vqa_x]
        
        q_lengths = [len(q.split()) for q in questions if isinstance(q, str)]
        
        # Explanations might be list or string
        e_lengths = []
        for e in explanations:
            if isinstance(e, list):
                e_lengths.append(sum(len(item.split()) for item in e if isinstance(item, str)))
            elif isinstance(e, str):
                e_lengths.append(len(e.split()))
        
        print(f"\n  Questions:")
        print(f"    - Count: {len(questions)}")
        print(f"    - Avg length: {sum(q_lengths)/len(q_lengths):.1f} words")
        print(f"    - Min/Max: {min(q_lengths)}-{max(q_lengths)} words")
        
        print(f"\n  Explanations:")
        print(f"    - Count: {len(explanations)}")
        print(f"    - Avg length: {sum(e_lengths)/len(e_lengths):.1f} words")
        print(f"    - Min/Max: {min(e_lengths)}-{max(e_lengths)} words")

# ============================================================================
# PART 2: A-OKVQA Inspection
# ============================================================================
print("\n" + "-"*80)
print("2. A-OKVQA INSPECTION")
print("-"*80)

AOKVQA_FILE = "data/annotations/aokvqa/aokvqa_v1p0_train.json"

if not os.path.exists(AOKVQA_FILE):
    print(f"❌ File not found: {AOKVQA_FILE}")
else:
    print(f"✅ File exists: {AOKVQA_FILE}")
    
    with open(AOKVQA_FILE) as f:
        aokvqa = json.load(f)
    
    print(f"\n  Type: {type(aokvqa).__name__}")
    print(f"  Total samples: {len(aokvqa)}")
    
    if isinstance(aokvqa, list) and len(aokvqa) > 0:
        print(f"\n  Sample structure (first 2 fields):")
        sample = aokvqa[0]
        for key in list(sample.keys())[:2]:
            val = sample[key]
            if isinstance(val, (list, dict)):
                val = str(val)[:50]
            elif isinstance(val, str):
                val = val[:50]
            print(f"    - {key}: {val}")
        
        # Structure check
        print(f"\n  Keys in sample: {list(sample.keys())}")
        
        # Check "rationales" field
        had_rationales = sum(1 for s in aokvqa if 'rationales' in s)
        print(f"    - Samples with 'rationales': {had_rationales}/{len(aokvqa)}")
        
        if had_rationales > 0 and 'rationales' in aokvqa[0]:
            rationale_counts = [len(s.get('rationales', [])) for s in aokvqa]
            print(f"    - Avg rationales per sample: {sum(rationale_counts)/len(rationale_counts):.1f}")

# ============================================================================
# PART 3: VQA-E Inspection
# ============================================================================
print("\n" + "-"*80)
print("3. VQA-E INSPECTION")
print("-"*80)

VQA_E_FILE = "data/annotations/vqa_e/VQA-E_train_filtered.json"

if not os.path.exists(VQA_E_FILE):
    print(f"❌ File not found: {VQA_E_FILE}")
else:
    print(f"✅ File exists: {VQA_E_FILE}")
    
    with open(VQA_E_FILE) as f:
        vqa_e = json.load(f)
    
    print(f"\n  Type: {type(vqa_e).__name__}")
    print(f"  Total samples: {len(vqa_e)}")
    
    if isinstance(vqa_e, list) and len(vqa_e) > 0:
        print(f"\n  Sample structure (first 3 fields):")
        sample = vqa_e[0]
        for key in list(sample.keys())[:3]:
            val = sample[key]
            if isinstance(val, str):
                val = val[:50] + "..." if len(val) > 50 else val
            print(f"    - {key}: {val}")
        
        # Stats
        questions = [s.get('question', '') for s in vqa_e]
        explanations = [s.get('explanation', []) for s in vqa_e]
        
        q_lengths = [len(q.split()) for q in questions if isinstance(q, str)]
        
        # Explanations might be list or string
        e_lengths = []
        for e in explanations:
            if isinstance(e, list):
                # Filter out non-string elements (like confidence scores)
                text_items = [item for item in e if isinstance(item, str)]
                e_lengths.append(sum(len(item.split()) for item in text_items))
            elif isinstance(e, str):
                e_lengths.append(len(e.split()))
        
        print(f"\n  Questions:")
        print(f"    - Count: {len([q for q in questions if q])}")
        print(f"    - Avg length: {sum(q_lengths)/len(q_lengths):.1f} words" if q_lengths else "    - N/A")
        
        print(f"\n  Explanations:")
        print(f"    - Count: {len([e for e in explanations if e])}")
        print(f"    - Avg length: {sum(e_lengths)/len(e_lengths):.1f} words" if e_lengths else "    - N/A")

# ============================================================================
# PART 4: Summary & Recommendations
# ============================================================================
print("\n" + "="*80)
print("SUMMARY & RECOMMENDATIONS")
print("="*80)

sources_ok = []
sources_issues = []

if os.path.exists(VQA_X_FILE):
    sources_ok.append("VQA-X ✅")
else:
    sources_issues.append("VQA-X ❌")

if os.path.exists(AOKVQA_FILE):
    sources_ok.append("A-OKVQA ✅")
else:
    sources_issues.append("A-OKVQA ❌")

if os.path.exists(VQA_E_FILE):
    sources_ok.append("VQA-E ✅")
else:
    sources_issues.append("VQA-E ❌")

print(f"\nSources ready: {', '.join(sources_ok)}")
if sources_issues:
    print(f"Issues: {', '.join(sources_issues)}")

print("\n" + "-"*80)
print("NEXT STEPS:")
print("-"*80)
print("""
1. Review data structure above
2. If all formats OK → Ready for Step 1 (merge)
3. If any issues → Fix before merge

When ready:
  python3 src/scripts/prepare_data.py
""")

print("\n" + "="*80)
