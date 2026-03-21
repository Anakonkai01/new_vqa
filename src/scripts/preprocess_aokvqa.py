"""
Bước 3: Preprocess A-OKVQA → Unified Format

Input:  data/annotations/aokvqa/aokvqa_v1p0_train.json
        data/annotations/aokvqa/aokvqa_v1p0_val.json
Output: data/annotations/aokvqa/aokvqa_train_unified.json
        data/annotations/aokvqa/aokvqa_val_unified.json

Format thực tế của A-OKVQA:
  - image_id: int (COCO 2017, ~95% trùng với COCO 2014)
  - question: field "question"
  - answer: choices[correct_choice_idx]
  - rationales: list of 3 human-written strings
"""

import json
import os


TRAIN_IMG_DIR = "data/images/train2014"
VAL_IMG_DIR   = "data/images/val2014"
TRAIN_IN      = "data/annotations/aokvqa/aokvqa_v1p0_train.json"
VAL_IN        = "data/annotations/aokvqa/aokvqa_v1p0_val.json"
TRAIN_OUT     = "data/annotations/aokvqa/aokvqa_train_unified.json"
VAL_OUT       = "data/annotations/aokvqa/aokvqa_val_unified.json"


def image_exists(img_id: int) -> bool:
    fname_train = f"COCO_train2014_{img_id:012d}.jpg"
    fname_val   = f"COCO_val2014_{img_id:012d}.jpg"
    if os.path.exists(os.path.join(TRAIN_IMG_DIR, fname_train)):
        return True
    if os.path.exists(os.path.join(VAL_IMG_DIR, fname_val)):
        return True
    return False


def process_file(in_path: str, split_tag: str) -> tuple:
    with open(in_path) as f:
        raw = json.load(f)

    unified = []
    skipped_missing  = 0
    skipped_no_rat   = 0

    for item in raw:
        img_id   = item["image_id"]
        question = item["question"]
        choices  = item["choices"]
        idx      = item["correct_choice_idx"]
        answer   = choices[idx]
        rats     = item.get("rationales", [])

        # Keep only non-empty string rationales
        rats = [r for r in rats if isinstance(r, str) and r.strip()]

        if not rats:
            skipped_no_rat += 1
            continue

        if not image_exists(img_id):
            skipped_missing += 1
            continue

        unified.append({
            "img_id":                 img_id,
            "question":               question,
            "multiple_choice_answer": answer,
            "explanation":            rats,      # list of up to 3 rationales
            "source":                 "aokvqa",
            "split":                  split_tag,
        })

    return unified, skipped_missing, skipped_no_rat, len(raw)


def main():
    os.makedirs(os.path.dirname(TRAIN_OUT), exist_ok=True)

    print("Processing A-OKVQA train...")
    train_data, train_miss, train_nort, train_total = process_file(TRAIN_IN, "train")
    with open(TRAIN_OUT, "w") as f:
        json.dump(train_data, f)

    print("Processing A-OKVQA val...")
    val_data, val_miss, val_nort, val_total = process_file(VAL_IN, "val")
    with open(VAL_OUT, "w") as f:
        json.dump(val_data, f)

    print()
    print("=" * 55)
    print("A-OKVQA Preprocessing Report")
    print("=" * 55)
    print(f"{'':30s} {'Train':>8}  {'Val':>8}")
    print(f"  {'Input':30s} {train_total:>8}  {val_total:>8}")
    print(f"  {'Skipped (image not found)':30s} {train_miss:>8}  {val_miss:>8}")
    print(f"  {'Skipped (no rationale)':30s} {train_nort:>8}  {val_nort:>8}")
    print(f"  {'Saved':30s} {len(train_data):>8}  {len(val_data):>8}")
    print(f"  {'Pass rate':30s} {len(train_data)/train_total*100:>7.1f}%  {len(val_data)/val_total*100:>7.1f}%")
    print()

    # Rationale count distribution
    from collections import Counter
    rat_counts = Counter(len(e["explanation"]) for e in train_data)
    print("Rationale count distribution (train):")
    for k in sorted(rat_counts):
        print(f"  {k} rationale(s): {rat_counts[k]:,} samples")
    print()

    print(f"Saved: {TRAIN_OUT}")
    print(f"Saved: {VAL_OUT}")

    # Spot-check
    print()
    print("--- Spot-check (first 2 entries) ---")
    for i, entry in enumerate(train_data[:2]):
        print(f"[{i}] img_id={entry['img_id']}  answer='{entry['multiple_choice_answer']}'")
        print(f"    q: {entry['question']}")
        for j, r in enumerate(entry['explanation']):
            print(f"    r{j}: {r[:90]}")


if __name__ == "__main__":
    main()
