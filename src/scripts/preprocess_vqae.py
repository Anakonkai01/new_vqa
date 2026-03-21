"""
Bước 4: Preprocess VQA-E → Unified Format

Input:  data/annotations/vqa_e/VQA-E_train_set.json   (181,298 samples)
        data/annotations/vqa_e/VQA-E_val_set.json
Output: data/annotations/vqa_e/vqa_e_train_unified.json
        data/annotations/vqa_e/vqa_e_val_unified.json

Format thực tế của VQA-E:
  - explanation: [string, float]  → lấy [0] (text), bỏ [1] (confidence score)
  - img_id: int  ✅
  - multiple_choice_answer: string  ✅
  - question: string  ✅
"""

import json
import os


TRAIN_IN  = "data/annotations/vqa_e/VQA-E_train_set.json"
VAL_IN    = "data/annotations/vqa_e/VQA-E_val_set.json"
TRAIN_OUT = "data/annotations/vqa_e/vqa_e_train_unified.json"
VAL_OUT   = "data/annotations/vqa_e/vqa_e_val_unified.json"

REQUIRED_FIELDS = {"img_id", "question", "multiple_choice_answer", "explanation"}


def process_file(in_path: str, split_tag: str) -> tuple:
    with open(in_path) as f:
        raw = json.load(f)

    unified        = []
    skipped_schema = 0
    skipped_expl   = 0

    for item in raw:
        # Schema check
        if not REQUIRED_FIELDS.issubset(item.keys()):
            skipped_schema += 1
            continue

        expl_raw = item["explanation"]

        # Extract text from [string, float] pair
        if isinstance(expl_raw, list) and len(expl_raw) >= 1 and isinstance(expl_raw[0], str):
            expl_text = expl_raw[0].strip()
        elif isinstance(expl_raw, str):
            expl_text = expl_raw.strip()
        else:
            skipped_expl += 1
            continue

        if not expl_text:
            skipped_expl += 1
            continue

        unified.append({
            "img_id":                 int(item["img_id"]),
            "question":               item["question"],
            "multiple_choice_answer": item["multiple_choice_answer"],
            "explanation":            [expl_text],
            "source":                 "vqa_e",
            "split":                  split_tag,
        })

    return unified, skipped_schema, skipped_expl, len(raw)


def main():
    os.makedirs(os.path.dirname(TRAIN_OUT), exist_ok=True)

    print("Processing VQA-E train...")
    train_data, tr_schema, tr_expl, tr_total = process_file(TRAIN_IN, "train")
    with open(TRAIN_OUT, "w") as f:
        json.dump(train_data, f)

    print("Processing VQA-E val...")
    val_data, va_schema, va_expl, va_total = process_file(VAL_IN, "val")
    with open(VAL_OUT, "w") as f:
        json.dump(val_data, f)

    print()
    print("=" * 55)
    print("VQA-E Preprocessing Report")
    print("=" * 55)
    print(f"{'':30s} {'Train':>8}  {'Val':>8}")
    print(f"  {'Input':30s} {tr_total:>8}  {va_total:>8}")
    print(f"  {'Skipped (schema error)':30s} {tr_schema:>8}  {va_schema:>8}")
    print(f"  {'Skipped (no explanation)':30s} {tr_expl:>8}  {va_expl:>8}")
    print(f"  {'Saved':30s} {len(train_data):>8}  {len(val_data):>8}")
    print(f"  {'Pass rate':30s} {len(train_data)/tr_total*100:>7.1f}%  {len(val_data)/va_total*100:>7.1f}%")
    print()
    print(f"Saved: {TRAIN_OUT}")
    print(f"Saved: {VAL_OUT}")

    # Spot-check
    print()
    print("--- Spot-check (first 2 entries) ---")
    for i, entry in enumerate(train_data[:2]):
        print(f"[{i}] img_id={entry['img_id']}  answer='{entry['multiple_choice_answer']}'")
        print(f"    q: {entry['question']}")
        print(f"    e: {entry['explanation'][0][:90]}")


if __name__ == "__main__":
    main()
