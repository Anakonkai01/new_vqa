"""
Bước 5: Merge tất cả sources → merged_train_raw.json

Input:  data/annotations/vqa_e/vqa_e_train_unified.json
        data/annotations/vqa_x/vqa_x_train_unified.json
        data/annotations/aokvqa/aokvqa_train_unified.json
Output: data/processed/merged_train_raw.json
"""

import json
import os
from collections import Counter


SOURCES = [
    ("vqa_e",   "data/annotations/vqa_e/vqa_e_train_unified.json"),
    ("vqa_x",   "data/annotations/vqa_x/vqa_x_train_unified.json"),
    ("aokvqa",  "data/annotations/aokvqa/aokvqa_train_unified.json"),
]
OUT = "data/processed/merged_train_raw.json"


def length_bin(text: str) -> str:
    n = len(text.split())
    if n <= 5:
        return "short"
    if n <= 14:
        return "medium"
    return "long"


def jaccard(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def main():
    os.makedirs("data/processed", exist_ok=True)

    merged = []
    source_counts = {}

    for src_name, path in SOURCES:
        with open(path) as f:
            data = json.load(f)
        source_counts[src_name] = len(data)
        merged.extend(data)
        print(f"  Loaded {len(data):>7,}  ← {src_name}")

    # Assign length_bin
    for item in merged:
        expl = item["explanation"][0] if item["explanation"] else ""
        item["length_bin"] = length_bin(expl)

    # Flag near-duplicate questions within same image (Jaccard ≥ 0.9)
    # Group by img_id, compare questions pairwise
    from collections import defaultdict
    img_groups = defaultdict(list)
    for i, item in enumerate(merged):
        img_groups[item["img_id"]].append(i)

    dup_flags = 0
    for img_id, idxs in img_groups.items():
        if len(idxs) < 2:
            continue
        questions = [merged[i]["question"].lower() for i in idxs]
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                if jaccard(questions[a], questions[b]) >= 0.9:
                    merged[idxs[b]]["_dup_flag"] = True
                    dup_flags += 1

    with open(OUT, "w") as f:
        json.dump(merged, f)

    # ── Stats ──────────────────────────────────────────────
    total = len(merged)
    bin_counts  = Counter(d["length_bin"] for d in merged)
    ans_counts  = Counter(d["multiple_choice_answer"] for d in merged)

    # Avg explanation length per source
    src_lengths = defaultdict(list)
    for d in merged:
        src_lengths[d["source"]].append(len(d["explanation"][0].split()))

    print()
    print("=" * 60)
    print("Merge Report")
    print("=" * 60)
    print(f"  {'Source':<12} {'Count':>8}  {'%':>6}  {'Avg expl len':>12}")
    print(f"  {'-'*12} {'-'*8}  {'-'*6}  {'-'*12}")
    for src_name, _ in SOURCES:
        cnt = source_counts[src_name]
        pct = cnt / total * 100
        avg = sum(src_lengths[src_name]) / len(src_lengths[src_name])
        print(f"  {src_name:<12} {cnt:>8,}  {pct:>5.1f}%  {avg:>11.1f}w")
    print(f"  {'TOTAL':<12} {total:>8,}  {'100.0':>5}%")
    print()

    print("  Length bin distribution:")
    for bin_name in ("short", "medium", "long"):
        cnt = bin_counts[bin_name]
        print(f"    {bin_name:<8} {cnt:>8,}  ({cnt/total*100:.1f}%)")
    print()

    print("  Top 20 answers:")
    for ans, cnt in ans_counts.most_common(20):
        print(f"    {ans:<25} {cnt:>7,}")
    print()

    print(f"  Potential duplicate Q pairs flagged: {dup_flags:,}")
    print()
    print(f"Saved: {OUT}  ({total:,} samples)")


if __name__ == "__main__":
    main()
