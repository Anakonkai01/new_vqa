#!/usr/bin/env python3
"""
STEP 1: Merge VQA-X, A-OKVQA, VQA-E into unified format
Handles different field names and structures from each source
"""

import json
import os
from pathlib import Path
from collections import defaultdict

print("\n" + "="*80)
print("STEP 1: MERGE DATA SOURCES")
print("="*80)

OUTPUT_FILE = "data/processed/unified_vqa_data.json"
os.makedirs("data/processed", exist_ok=True)

# ============================================================================
# Function to extract numeric image ID from COCO format
# ============================================================================
def extract_image_id(img_id):
    """
    Extract numeric ID from various formats:
    - COCO_train2014_000000262146 → 262146
    - 262146 → 262146
    """
    if isinstance(img_id, int):
        return img_id
    if isinstance(img_id, str):
        parts = img_id.split('_')
        if len(parts) >= 3:
            try:
                return int(parts[-1])
            except ValueError:
                pass
        try:
            return int(img_id)
        except ValueError:
            pass
    return None

# ============================================================================
# VQA-X Processing
# ============================================================================
print("\n[1/3] Processing VQA-X...")

vqax_samples = []
with open('data/annotations/vqa_x/train_x.json') as f:
    vqa_x = json.load(f)

for item in vqa_x:
    try:
        # Extract explanation (it's a list with 1 item)
        explanation = item.get('explanation', [])
        if isinstance(explanation, list) and len(explanation) > 0:
            explanation = explanation[0]
        else:
            explanation = ""
        
        unified = {
            'source': 'vqa_x',
            'image_id': extract_image_id(item['img_id']),
            'question': item.get('sent', ''),
            'explanation': explanation,
            'answer': list(item.get('label', {}).keys())[0] if item.get('label') else '',
            'question_id': item.get('question_id'),
        }
        vqax_samples.append(unified)
    except Exception as e:
        print(f"  Error processing VQA-X item: {e}")
        continue

print(f"  ✓ Processed: {len(vqax_samples)} samples")

# ============================================================================
# A-OKVQA Processing
# ============================================================================
print("\n[2/3] Processing A-OKVQA...")

aokvqa_samples = []
with open('data/annotations/aokvqa/aokvqa_v1p0_train.json') as f:
    aokvqa = json.load(f)

for item in aokvqa:
    try:
        # A-OKVQA has multiple rationales, we'll create one sample per rationale
        rationales = item.get('rationales', [])
        question = item.get('question', '')
        image_id = extract_image_id(item.get('image_id'))
        
        # Create sample for each rationale
        for rationale in rationales:
            unified = {
                'source': 'aokvqa',
                'image_id': image_id,
                'question': question,
                'explanation': rationale,
                'answer': '',  # A-OKVQA answers are multiple choice, skip for now
                'question_id': item.get('question_id'),
            }
            aokvqa_samples.append(unified)
    except Exception as e:
        print(f"  Error processing A-OKVQA item: {e}")
        continue

print(f"  ✓ Processed: {len(aokvqa_samples)} samples ({len(aokvqa_samples)/len(aokvqa):.1f}x due to multiple rationales)")

# ============================================================================
# VQA-E Processing
# ============================================================================
print("\n[3/3] Processing VQA-E...")

vqae_samples = []
with open('data/annotations/vqa_e/VQA-E_train_filtered.json') as f:
    vqa_e = json.load(f)

for item in vqa_e:
    try:
        # Extract explanation (it's a list with [text, confidence])
        explanation = item.get('explanation', [])
        if isinstance(explanation, list) and len(explanation) > 0:
            explanation = explanation[0]  # Get text, ignore confidence score
        else:
            explanation = ""
        
        unified = {
            'source': 'vqa_e',
            'image_id': extract_image_id(item.get('img_id')),
            'question': item.get('question', ''),
            'explanation': explanation,
            'answer': item.get('answer', ''),
            'question_id': item.get('question_id'),
        }
        vqae_samples.append(unified)
    except Exception as e:
        print(f"  Error processing VQA-E item: {e}")
        continue

print(f"  ✓ Processed: {len(vqae_samples)} samples")

# ============================================================================
# Merge All
# ============================================================================
print("\n" + "-"*80)
print("MERGING...")
print("-"*80)

all_samples = vqax_samples + aokvqa_samples + vqae_samples

print(f"\nTotal samples: {len(all_samples)}")
print(f"  - VQA-X: {len(vqax_samples)} ({len(vqax_samples)/len(all_samples)*100:.1f}%)")
print(f"  - A-OKVQA: {len(aokvqa_samples)} ({len(aokvqa_samples)/len(all_samples)*100:.1f}%)")
print(f"  - VQA-E: {len(vqae_samples)} ({len(vqae_samples)/len(all_samples)*100:.1f}%)")

# Statistics
print(f"\nDATA QUALITY CHECKS:")
missing_question = sum(1 for s in all_samples if not s.get('question', '').strip())
missing_explanation = sum(1 for s in all_samples if not s.get('explanation', '').strip())
missing_image = sum(1 for s in all_samples if s.get('image_id') is None)

print(f"  Missing question: {missing_question} ({missing_question/len(all_samples)*100:.1f}%)")
print(f"  Missing explanation: {missing_explanation} ({missing_explanation/len(all_samples)*100:.1f}%)")
print(f"  Missing image_id: {missing_image} ({missing_image/len(all_samples)*100:.1f}%)")

# Save to file
print(f"\nSAVING to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w') as f:
    json.dump(all_samples, f, indent=2)

print(f"✅ Saved {len(all_samples)} samples")

# Summary
print("\n" + "="*80)
print("STEP 1 COMPLETE")
print("="*80)
print(f"""
Output: {OUTPUT_FILE}
Samples: {len(all_samples)}

Next step:
  python3 src/scripts/filter_data.py
""")

print("="*80 + "\n")
