#!/usr/bin/env python3
"""
Verify a directory of pre-extracted .pt files (Model H vs Model F formats).

Checks:
  - Homogeneous last dimension D of region rows (detect mix of 1029 vs 1031, etc.)
  - Optional: consistent grid_feat presence and channel count

Usage (from repo root):
  export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
  python src/scripts/verify_extracted_visual_features.py --dir data/vg_features

Former name: verify_vg_features.py

Exit code 1 if inconsistent region_dim or unreadable sample (when --strict).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Repo src on path when PYTHONPATH=.../src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.dataset import audit_butd_feature_dir  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Audit .pt feature directory for Model H")
    p.add_argument("--dir", type=str, default="data/vg_features", help="Folder of {img_id}.pt")
    p.add_argument("--max-files", type=int, default=2048, help="Max files to sample (shuffled)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strict", action="store_true", help="Exit 1 on mismatch or parse failures")
    p.add_argument("--json", type=str, default=None, help="Write report JSON path")
    args = p.parse_args()

    if not os.path.isdir(args.dir):
        print(f"[ERROR] Not a directory: {args.dir!r}")
        sys.exit(1)

    try:
        rep = audit_butd_feature_dir(
            args.dir,
            max_files=args.max_files,
            seed=args.seed,
            raise_on_mismatch=args.strict,
        )
    except ValueError as e:
        print(f"[FAIL] {e}")
        sys.exit(1)

    print(json.dumps({k: v for k, v in rep.items() if k != "issues"}, indent=2))
    for msg in rep.get("issues") or []:
        print(f"[issue] {msg}")

    if args.strict and rep.get("region_dim") is None:
        sys.exit(1)

    if args.json:
        jp = os.path.abspath(args.json)
        jd = os.path.dirname(jp)
        if jd:
            os.makedirs(jd, exist_ok=True)
        with open(jp, "w") as f:
            json.dump(rep, f, indent=2)
        print(f"Wrote {jp}")


if __name__ == "__main__":
    main()
