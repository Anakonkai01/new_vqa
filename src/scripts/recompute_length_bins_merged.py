#!/usr/bin/env python3
"""
Recompute string length_bin labels on merged JSON from full target length
(answer + because + explanation), for tools that read JSON offline.

Training uses _full_seq_length_bin inside VQAGenerativeDataset.__getitem__;
this script syncs stored JSON fields for reporting / external pipelines.

Usage:
  PYTHONPATH=src python src/scripts/recompute_length_bins_merged.py \\
      --json data/processed/merged_train_filtered.json --in-place
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_SRC = os.path.join(os.path.dirname(__file__), "..")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from data.dataset import _full_seq_length_bin  # noqa: E402

_BIN_NAMES = ("short", "medium", "long")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=str, required=True)
    p.add_argument("--in-place", action="store_true", help="Overwrite input file (backup recommended)")
    p.add_argument("--out", type=str, default=None, help="Output path if not in-place")
    args = p.parse_args()

    with open(args.json, "r") as f:
        data = json.load(f)

    for ann in data:
        ans = ann.get("multiple_choice_answer", "") or ""
        exp_list = ann.get("explanation", [])
        if isinstance(exp_list, list):
            exp_text = next((e for e in exp_list if isinstance(e, str) and e.strip()), "")
        else:
            exp_text = str(exp_list or "")
        bi = _full_seq_length_bin(ans, exp_text)
        ann["length_bin"] = _BIN_NAMES[bi]

    out_path = args.json if args.in_place else (args.out or args.json + ".recomputed.json")
    if not args.in_place and args.out is None:
        print(f"Writing to {out_path} (use --in-place to overwrite source)")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Updated {len(data)} rows → {out_path}")


if __name__ == "__main__":
    main()
