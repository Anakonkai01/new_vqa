"""
Bước 6: 5-Stage Quality Filter

Input:  data/processed/merged_train_raw.json
        data/annotations/vqa_v2/instances_train2014.json
        data/annotations/vqa_v2/instances_val2014.json
Output: data/processed/merged_train_filtered.json   (score >= 3)
        data/processed/merged_train_scored.json      (all + quality_score)
        data/processed/filter_report.json

Scoring:
  Stage 1 (length gate)       fail → -1
  Stage 2 (copy-of-question)  fail → -1
  Stage 3 (visual grounding)  fail → -2  (hallucination = worst offense)
  Stage 4 (answer consistency) fail → -1
  Stage 5 (deduplication)     fail → -1
  Max score = 5  (all pass)
  Kept if score >= 3
"""

import json
import os
import re
from collections import defaultdict

import spacy

# ── Paths ──────────────────────────────────────────────────────
MERGED_IN        = "data/processed/merged_train_raw.json"
INSTANCES_TRAIN  = "data/annotations/vqa_v2/instances_train2014.json"
INSTANCES_VAL    = "data/annotations/vqa_v2/instances_val2014.json"
FILTERED_OUT     = "data/processed/merged_train_filtered.json"
SCORED_OUT       = "data/processed/merged_train_scored.json"
REPORT_OUT       = "data/processed/filter_report.json"

# ── Config ─────────────────────────────────────────────────────
MIN_EXPL_WORDS       = 5
MAX_EXPL_WORDS       = 35
MAX_EXPL_WORDS_AOKVQA = 40   # A-OKVQA rationales are longer and higher quality
JACCARD_Q_THRESH     = 0.6   # Stage 2: reject if >= this
GROUND_THRESH        = 0.3   # Stage 3: reject if grounding_ratio < this (when >=2 nouns)
DUP_THRESH           = 0.85  # Stage 5: reject if explanation Jaccard >= this (same img)
MIN_SCORE            = 3     # kept if score >= this

GENERIC_NOUNS = {
    "thing", "stuff", "way", "time", "picture", "image", "photo",
    "scene", "view", "area", "part", "side", "type", "kind", "form",
    "lot", "number", "group", "set", "pair", "piece", "bit", "look",
}


# ── Helpers ────────────────────────────────────────────────────

def tokenize(text: str) -> set:
    return set(re.findall(r"[a-z]+", text.lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def load_coco_objects(paths: list) -> dict:
    """Build {image_id: set(category_names)} from COCO instances JSONs."""
    img_to_objs = defaultdict(set)
    for path in paths:
        print(f"  Loading COCO instances: {path}")
        with open(path) as f:
            d = json.load(f)
        cat_map = {c["id"]: c["name"] for c in d["categories"]}
        for ann in d["annotations"]:
            img_to_objs[ann["image_id"]].add(cat_map[ann["category_id"]])
    return img_to_objs


def extract_content_nouns(doc) -> list:
    """Extract non-generic NOUN/PROPN lemmas using spaCy doc."""
    nouns = []
    for tok in doc:
        if tok.pos_ in ("NOUN", "PROPN") and tok.lemma_.lower() not in GENERIC_NOUNS:
            nouns.append(tok.lemma_.lower())
    return nouns


# ── Stage functions ────────────────────────────────────────────

def stage1_length(expl: str, source: str) -> bool:
    """True = pass.  A-OKVQA rationales allow up to 40 words."""
    n = len(expl.split())
    max_w = MAX_EXPL_WORDS_AOKVQA if source == "aokvqa" else MAX_EXPL_WORDS
    return MIN_EXPL_WORDS <= n <= max_w


def stage2_copy_question(question: str, expl: str) -> bool:
    """True = pass (not a copy)."""
    return jaccard(question, expl) < JACCARD_Q_THRESH


def stage3_grounding(expl_doc, img_id: int, img_to_objs: dict) -> bool:
    """True = pass (sufficiently grounded)."""
    nouns = extract_content_nouns(expl_doc)
    if len(nouns) < 2:
        return True   # too few nouns to judge → give benefit of doubt
    objects = img_to_objs.get(img_id, set())
    if not objects:
        return True   # no COCO annotation for this image → skip
    matched = sum(1 for n in nouns if n in objects)
    ratio = matched / len(nouns)
    return ratio >= GROUND_THRESH


# Number word ↔ digit normalization for Stage 4
_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12",
}
_NUM_DIGITS = {v: k for k, v in _NUM_WORDS.items()}


def _normalize_numbers(text: str) -> str:
    """Replace digit strings with their word equivalents and vice-versa."""
    words = text.lower().split()
    out = []
    for w in words:
        out.append(w)
        if w in _NUM_WORDS:
            out.append(_NUM_WORDS[w])   # add digit alias
        if w in _NUM_DIGITS:
            out.append(_NUM_DIGITS[w])  # add word alias
    return " ".join(out)


def stage4_answer_consistency(answer: str, expl: str, source: str) -> bool:
    """True = pass.

    VQA-X and A-OKVQA are human-curated; their explanations deliberately use
    different phrasing from the answer (e.g. 'skiing' → 'wearing skis',
    'macintosh' → 'mac').  Stage 4 is noise-reduction for VQA-E only.

    For VQA-E:
      - Yes/no answers: skip (explanations are visual descriptions).
      - Numbers: normalise digit ↔ word ('2' ↔ 'two').
      - Others: check answer stem (first 5 chars) OR exact match anywhere in
        explanation.  Genuine contradictions (zero token overlap) still fail.
    """
    if source in ("vqa_x", "aokvqa"):
        return True

    ans_lower = answer.lower().strip()
    if ans_lower in ("yes", "no"):
        return True

    expl_lower  = expl.lower()
    expl_norm   = _normalize_numbers(expl_lower)
    ans_norm    = _normalize_numbers(ans_lower)

    # Exact match after number normalisation
    if ans_norm in expl_norm:
        return True

    # Stem match: first 5 chars of answer word appears in any explanation word
    expl_words = set(re.findall(r"[a-z]+", expl_lower))
    for ans_tok in re.findall(r"[a-z]+", ans_lower):
        if len(ans_tok) >= 4:
            if any(w[:len(ans_tok)] == ans_tok or ans_tok[:5] == w[:5]
                   for w in expl_words if len(w) >= 4):
                return True

    return False


def stage5_dedup(expl: str, seen_expls: list) -> bool:
    """True = pass (not a near-duplicate of prior explanations for same image)."""
    for prev in seen_expls:
        if jaccard(expl, prev) >= DUP_THRESH:
            return False
    return True


# ── Main ───────────────────────────────────────────────────────

def main():
    os.makedirs("data/processed", exist_ok=True)

    print("Loading data...")
    with open(MERGED_IN) as f:
        data = json.load(f)
    print(f"  Total input: {len(data):,}")

    print("Loading COCO object annotations...")
    img_to_objs = load_coco_objects([INSTANCES_TRAIN, INSTANCES_VAL])
    print(f"  Images with annotations: {len(img_to_objs):,}")

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    # Per-source stage counters: {source: {stage: fail_count}}
    sources = ["vqa_e", "vqa_x", "aokvqa"]
    stage_fails = {src: {s: 0 for s in range(1, 6)} for src in sources}
    source_totals = {src: 0 for src in sources}

    # For Stage 5: track per-image seen explanations
    img_seen_expls = defaultdict(list)

    # Score all samples
    scored = []
    print("Scoring samples (this may take 10-15 min)...")

    # Batch spaCy for speed
    explanations = [d["explanation"][0] for d in data]
    batch_size = 2000
    docs = []
    for i in range(0, len(explanations), batch_size):
        batch = explanations[i:i + batch_size]
        docs.extend(nlp.pipe(batch, batch_size=batch_size))
        if (i // batch_size) % 10 == 0:
            print(f"  spaCy: {i:>7,} / {len(data):,}", flush=True)

    print("  spaCy done. Applying filter stages...")
    for item, doc in zip(data, docs):
        src   = item["source"]
        expl  = item["explanation"][0]
        q     = item["question"]
        ans   = item["multiple_choice_answer"]
        img   = item["img_id"]

        source_totals[src] = source_totals.get(src, 0) + 1

        score = 5

        # Stage 1
        if not stage1_length(expl, src):
            score -= 1
            stage_fails[src][1] += 1

        # Stage 2
        if not stage2_copy_question(q, expl):
            score -= 1
            stage_fails[src][2] += 1

        # Stage 3
        if not stage3_grounding(doc, img, img_to_objs):
            score -= 2
            stage_fails[src][3] += 1

        # Stage 4
        if not stage4_answer_consistency(ans, expl, src):
            score -= 1
            stage_fails[src][4] += 1

        # Stage 5
        if not stage5_dedup(expl, img_seen_expls[img]):
            score -= 1
            stage_fails[src][5] += 1
        else:
            img_seen_expls[img].append(expl)

        item["quality_score"] = score
        scored.append(item)

    # Split filtered vs all-scored
    filtered = [d for d in scored if d["quality_score"] >= MIN_SCORE]

    # Save outputs
    with open(SCORED_OUT, "w") as f:
        json.dump(scored, f)
    with open(FILTERED_OUT, "w") as f:
        json.dump(filtered, f)

    # Build report
    from collections import Counter
    score_dist_all = Counter(d["quality_score"] for d in scored)
    score_dist_src = {src: Counter(d["quality_score"] for d in scored if d["source"] == src)
                      for src in sources}

    report = {
        "total_input":    len(data),
        "total_kept":     len(filtered),
        "pass_rate":      len(filtered) / len(data),
        "source_totals":  source_totals,
        "stage_fails":    stage_fails,
        "score_dist":     {str(k): v for k, v in score_dist_all.items()},
        "score_dist_src": {src: {str(k): v for k, v in score_dist_src[src].items()}
                           for src in sources},
    }
    with open(REPORT_OUT, "w") as f:
        json.dump(report, f, indent=2)

    # ── Print report ──────────────────────────────────────────
    total = len(data)
    print()
    print("=" * 72)
    print("Filter Report")
    print("=" * 72)
    print(f"  {'':30s} {'vqa_e':>9}  {'vqa_x':>9}  {'aokvqa':>9}  {'Total':>9}")
    print(f"  {'-'*30} {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")
    print(f"  {'Input':30s} {source_totals['vqa_e']:>9,}  {source_totals['vqa_x']:>9,}  {source_totals['aokvqa']:>9,}  {total:>9,}")
    for s in range(1, 6):
        label = f"Stage {s} fail"
        row = [stage_fails[src][s] for src in sources]
        print(f"  {label:30s} {row[0]:>9,}  {row[1]:>9,}  {row[2]:>9,}  {sum(row):>9,}")
    print(f"  {'-'*30} {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")

    score_labels = [("GOLD  (score 5)", 5), ("GOOD  (score 4)", 4),
                    ("OK    (score 3)", 3), ("REJECT (score<=2)", None)]
    for label, sc in score_labels:
        row = []
        for src in sources:
            if sc is None:
                cnt = sum(v for k, v in score_dist_src[src].items() if int(k) <= 2)
            else:
                cnt = score_dist_src[src].get(sc, 0)
            row.append(cnt)
        tot = sum(row)
        print(f"  {label:30s} {row[0]:>9,}  {row[1]:>9,}  {row[2]:>9,}  {tot:>9,}")
    print(f"  {'-'*30} {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")

    kept_src = [sum(score_dist_src[src].get(sc, 0) for sc in [3, 4, 5]) for src in sources]
    print(f"  {'Kept (score>=3)':30s} {kept_src[0]:>9,}  {kept_src[1]:>9,}  {kept_src[2]:>9,}  {len(filtered):>9,}")
    rates = [k / source_totals[s] * 100 for k, s in zip(kept_src, sources)]
    print(f"  {'Pass rate':30s} {rates[0]:>8.1f}%  {rates[1]:>8.1f}%  {rates[2]:>8.1f}%  {len(filtered)/total*100:>8.1f}%")
    print()
    print(f"Saved filtered : {FILTERED_OUT}  ({len(filtered):,} samples)")
    print(f"Saved scored   : {SCORED_OUT}  ({len(scored):,} samples)")
    print(f"Saved report   : {REPORT_OUT}")


if __name__ == "__main__":
    main()
