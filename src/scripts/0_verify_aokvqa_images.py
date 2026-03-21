#!/usr/bin/env python3
"""
STEP 0: Verify A-OKVQA image IDs exist in COCO 2014
Check how many A-OKVQA samples can be matched to existing COCO images
"""

import json
import os
from pathlib import Path
from collections import defaultdict

# Paths
AOKVQA_FILE = "data/annotations/aokvqa/aokvqa_v1p0_train.json"
COCO_TRAIN_DIR = "data/images/train2014"
COCO_VAL_DIR = "data/images/val2014"

print("=" * 80)
print("STEP 0: Verify A-OKVQA Images → COCO 2014 Mapping")
print("=" * 80)

# Step 1: Load A-OKVQA and check image_id format
print("\n[1/4] Loading A-OKVQA data...")
with open(AOKVQA_FILE) as f:
    aokvqa = json.load(f)

print(f"  A-OKVQA samples: {len(aokvqa)}")

# Sample first 3
for i in range(min(3, len(aokvqa))):
    print(f"    Sample {i+1}: image_id={aokvqa[i]['image_id']} (type: {type(aokvqa[i]['image_id']).__name__})")

# Step 2: Build set of existing COCO image IDs from filenames
print("\n[2/4] Scanning COCO 2014 images directories...")

coco_image_ids = set()

def extract_image_id(filename):
    """Extract numeric ID from COCO filename format"""
    # Filename format: COCO_train2014_000000000001.jpg or similar
    # Extract the last 12-digit number
    base = Path(filename).stem  # Remove .jpg extension
    parts = base.split('_')
    if len(parts) >= 3:
        try:
            return int(parts[-1])
        except ValueError:
            pass
    return None

# Scan train2014
if os.path.exists(COCO_TRAIN_DIR):
    train_files = os.listdir(COCO_TRAIN_DIR)
    for fname in train_files:
        img_id = extract_image_id(fname)
        if img_id:
            coco_image_ids.add(img_id)
    print(f"  Train2014: {len([f for f in train_files if f.endswith('.jpg')])} images, unique IDs: {len(coco_image_ids)}")

else:
    print(f"  ❌ Train2014 directory not found: {COCO_TRAIN_DIR}")

# Scan val2014
if os.path.exists(COCO_VAL_DIR):
    val_files = os.listdir(COCO_VAL_DIR)
    val_count = len([f for f in val_files if f.endswith('.jpg')])
    for fname in val_files:
        img_id = extract_image_id(fname)
        if img_id:
            coco_image_ids.add(img_id)
    print(f"  Val2014: {val_count} images, total unique IDs: {len(coco_image_ids)}")
else:
    print(f"  ❌ Val2014 directory not found: {COCO_VAL_DIR}")

print(f"\n  Total COCO 2014 image IDs found: {len(coco_image_ids)}")

# Step 3: Check A-OKVQA image_id mapping
print("\n[3/4] Checking A-OKVQA → COCO 2014 mapping...")

matched = 0
missing = []
stats = defaultdict(int)

for item in aokvqa:
    img_id = item['image_id']
    
    # A-OKVQA uses COCO 2017 format, but let's check if numeric ID exists in COCO 2014
    if img_id in coco_image_ids:
        matched += 1
    else:
        missing.append(img_id)
    stats['checked'] += 1

match_pct = (matched / stats['checked'] * 100) if stats['checked'] > 0 else 0

print(f"  Matched: {matched} / {stats['checked']} ({match_pct:.1f}%)")
print(f"  Missing: {len(missing)} ({100 - match_pct:.1f}%)")

# Step 4: Summary & Recommendation
print("\n[4/4] Summary & Recommendation")
print("=" * 80)

if match_pct >= 90:
    status = "✅ PASS - Can use A-OKVQA with COCO 2014"
    action = "Filter out missing images and proceed with Step 1"
elif match_pct >= 70:
    status = "⚠️  PARTIAL - Many A-OKVQA images found, some missing"
    action = "Can use, but will lose ~{}% of data".format(int(100 - match_pct))
else:
    status = "❌ FAIL - Most A-OKVQA image IDs not in COCO 2014"
    action = "Need to download COCO 2017 images (~1GB) or skip A-OKVQA"

print(f"\nMatch Rate: {match_pct:.1f}%")
print(f"Status: {status}")
print(f"Recommended Action: {action}")

if match_pct < 100:
    print(f"\nSample missing image_ids (first 5):")
    for img_id in missing[:5]:
        print(f"  - {img_id}")

print("\n" + "=" * 80)
print("Decision:")
print("=" * 80)

if match_pct >= 85:
    print("""
OPTION A (Recommended): Use A-OKVQA with COCO 2014
- Action: Modify prepare_data.py to filter out missing images
- Loss: ~{}% of A-OKVQA data
- Time: 5 minutes to fix prepare_data.py
- Result: ~{} samples instead of 51K

OPTION B: Download COCO 2017 val images
- Action: Download missing 2017 val images
- Time: 30-45 minutes
- Result: Full 51K A-OKVQA samples
""".format(int(100 - match_pct), int(51000 * match_pct / 100)))
else:
    print("""
Options are limited - must choose one:

OPTION A: Skip A-OKVQA
- Use only VQA-X (29.5K) + VQA-E (172.8K) = ~200K samples
- Time: Update prepare_data.py immediately, proceed with Step 1
- Advantage: No delays, data ready

OPTION B: Download COCO 2017 val images
- Time: 30-45 minutes download + ~5 min GPU setup
- Result: Full A-OKVQA (51K) + VQA-X + VQA-E = ~254K samples
- Advantage: Maximum data for training
""")

print("\n✋ NEXT STEP:")
print("   User decision: Choose OPTION A or B")
print("   Then execute: python3 src/scripts/setup_step1.py")
