"""
Bước 2: Preprocess VQA-X → Unified Format

Input:  data/annotations/vqa_x/train_x.json
        data/annotations/vqa_x/val_x.json
Output: data/annotations/vqa_x/vqa_x_train_unified.json
        data/annotations/vqa_x/vqa_x_val_unified.json

Format thực tế của VQA-X:
  - img_id: string "COCO_train2014_000000262146" → parse thành int 262146
  - question: field "sent" (không phải "question")
  - answer: dict "label" {word: weight} → key có weight cao nhất
  - explanation: list of 1 string
"""

import json
import os
import re


TRAIN_IMG_DIR  = "data/images/train2014"
VAL_IMG_DIR    = "data/images/val2014"
TRAIN_IN       = "data/annotations/vqa_x/train_x.json"
VAL_IN         = "data/annotations/vqa_x/val_x.json"
TRAIN_OUT      = "data/annotations/vqa_x/vqa_x_train_unified.json"
VAL_OUT        = "data/annotations/vqa_x/vqa_x_val_unified.json"


def parse_img_id(raw_id: str) -> int:
    """
    "COCO_train2014_000000262146" → 262146
    "COCO_val2014_000000393223"   → 393223
    Fallback: try int() directly.
    """
    if isinstance(raw_id, int):
        return raw_id
    m = re.search(r"_(\d+)$", str(raw_id))
    if m:
        return int(m.group(1))
    return int(raw_id)


def get_split(raw_id: str) -> str:
    """Infer train/val from the img_id string."""
    s = str(raw_id)
    if "val2014" in s:
        return "val"
    return "train"


def image_exists(img_id: int, hint_split: str) -> bool:
    fname_train = f"COCO_train2014_{img_id:012d}.jpg"
    fname_val   = f"COCO_val2014_{img_id:012d}.jpg"
    if os.path.exists(os.path.join(TRAIN_IMG_DIR, fname_train)):
        return True
    if os.path.exists(os.path.join(VAL_IMG_DIR, fname_val)):
        return True
    return False


def get_answer(label: dict) -> str:
    """
    label = {"skiing": 1} or {"no": 0.6, "yes": 1}
    Return key with highest weight.
    """
    if not label:
        return ""
    return max(label, key=lambda k: label[k])


def process_file(in_path: str, split_tag: str) -> tuple:
    with open(in_path) as f:
        raw = json.load(f)

    unified = []
    skipped_missing = 0
    skipped_no_expl = 0

    for item in raw:
        raw_id   = item["img_id"]
        img_id   = parse_img_id(raw_id)
        hint     = get_split(raw_id)
        question = item["sent"]
        label    = item.get("label", {})
        answer   = get_answer(label)
        expl_raw = item.get("explanation", [])

        # Normalise explanation to list of strings
        if isinstance(expl_raw, str):
            expl_list = [expl_raw]
        elif isinstance(expl_raw, list):
            expl_list = [e for e in expl_raw if isinstance(e, str) and e.strip()]
        else:
            expl_list = []

        if not expl_list:
            skipped_no_expl += 1
            continue

        if not image_exists(img_id, hint):
            skipped_missing += 1
            continue

        unified.append({
            "img_id":                  img_id,
            "question":                question,
            "multiple_choice_answer":  answer,
            "explanation":             expl_list,
            "source":                  "vqa_x",
            "split":                   split_tag,
        })

    return unified, skipped_missing, skipped_no_expl, len(raw)


def main():
    os.makedirs(os.path.dirname(TRAIN_OUT), exist_ok=True)

    print("Processing VQA-X train...")
    train_data, train_miss, train_noexpl, train_total = process_file(TRAIN_IN, "train")
    with open(TRAIN_OUT, "w") as f:
        json.dump(train_data, f)

    print("Processing VQA-X val...")
    val_data, val_miss, val_noexpl, val_total = process_file(VAL_IN, "val")
    with open(VAL_OUT, "w") as f:
        json.dump(val_data, f)

    print()
    print("=" * 55)
    print("VQA-X Preprocessing Report")
    print("=" * 55)
    print(f"{'':30s} {'Train':>8}  {'Val':>8}")
    print(f"  {'Input':30s} {train_total:>8}  {val_total:>8}")
    print(f"  {'Skipped (image not found)':30s} {train_miss:>8}  {val_miss:>8}")
    print(f"  {'Skipped (no explanation)':30s} {train_noexpl:>8}  {val_noexpl:>8}")
    print(f"  {'Saved':30s} {len(train_data):>8}  {len(val_data):>8}")
    print(f"  {'Pass rate':30s} {len(train_data)/train_total*100:>7.1f}%  {len(val_data)/val_total*100:>7.1f}%")
    print()
    print(f"Saved: {TRAIN_OUT}")
    print(f"Saved: {VAL_OUT}")

    # Spot-check
    print()
    print("--- Spot-check (first 2 entries) ---")
    for i, entry in enumerate(train_data[:2]):
        print(f"[{i}] img_id={entry['img_id']}  answer='{entry['multiple_choice_answer']}'")
        print(f"    q: {entry['question']}")
        print(f"    e: {entry['explanation'][0][:80]}")


if __name__ == "__main__":
    main()
