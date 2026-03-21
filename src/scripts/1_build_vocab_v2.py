"""
Bước 7: Rebuild Vocabulary từ merged filtered data

Input:  data/processed/merged_train_filtered.json
        data/annotations/vqa_v2/v2_OpenEnded_mscoco_train2014_questions.json
        data/annotations/vqa_v2/v2_mscoco_train2014_annotations.json
Output: data/processed/vocab_questions.json   (overwrite, backup trước)
        data/processed/vocab_answers.json     (overwrite, backup trước)

Threshold = 3 (token phải xuất hiện >= 3 lần để vào vocab)
"""

import json
import os
import re
import shutil
from collections import Counter

FILTERED_DATA  = "data/processed/merged_train_filtered.json"
VQA2_QUESTIONS = "data/annotations/vqa_v2/v2_OpenEnded_mscoco_train2014_questions.json"
VQA2_ANSWERS   = "data/annotations/vqa_v2/v2_mscoco_train2014_annotations.json"

VOCAB_Q_OUT    = "data/processed/vocab_questions.json"
VOCAB_A_OUT    = "data/processed/vocab_answers.json"
BACKUP_DIR     = "data/processed/backup"

THRESHOLD      = 3

# Special tokens (must stay at fixed indices)
PAD_TOKEN   = "<pad>"    # 0
START_TOKEN = "<start>"  # 1
END_TOKEN   = "<end>"    # 2
UNK_TOKEN   = "<unk>"    # 3
SPECIALS    = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]


def tokenize(text: str) -> list:
    return re.findall(r"[a-z']+", text.lower())


def build_vocab(counter: Counter, threshold: int) -> dict:
    """Return {token: idx} with specials at 0-3."""
    vocab = {tok: i for i, tok in enumerate(SPECIALS)}
    for word, freq in sorted(counter.items(), key=lambda x: -x[1]):
        if freq >= threshold and word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def load_existing_vocab(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def main():
    os.makedirs(BACKUP_DIR, exist_ok=True)

    # ── Backup existing vocabs ────────────────────────────────
    for src, tag in [(VOCAB_Q_OUT, "questions"), (VOCAB_A_OUT, "answers")]:
        if os.path.exists(src):
            old = load_existing_vocab(src)
            dst = os.path.join(BACKUP_DIR, f"vocab_{tag}_{len(old)}.json")
            shutil.copy2(src, dst)
            print(f"Backed up {src} ({len(old):,} tokens) → {dst}")

    # ── Load data ─────────────────────────────────────────────
    print("\nLoading filtered data...")
    with open(FILTERED_DATA) as f:
        filtered = json.load(f)
    print(f"  Filtered samples: {len(filtered):,}")

    print("Loading VQA v2.0 data...")
    with open(VQA2_QUESTIONS) as f:
        vqa2_q = json.load(f)["questions"]
    with open(VQA2_ANSWERS) as f:
        vqa2_a = json.load(f)["annotations"]
    print(f"  VQA v2.0 questions: {len(vqa2_q):,}")

    # ── Count tokens ─────────────────────────────────────────
    q_counter = Counter()
    a_counter = Counter()

    # From filtered multi-source data
    for d in filtered:
        q_counter.update(tokenize(d["question"]))
        a_counter.update(tokenize(d["explanation"][0]))
        a_counter.update(tokenize(d["multiple_choice_answer"]))

    # From VQA v2.0 (for Phase 1 warm-up compatibility)
    for item in vqa2_q:
        q_counter.update(tokenize(item["question"]))
    for item in vqa2_a:
        a_counter.update(tokenize(item["multiple_choice_answer"]))

    # ── Build vocabs ──────────────────────────────────────────
    vocab_q = build_vocab(q_counter, THRESHOLD)
    vocab_a = build_vocab(a_counter, THRESHOLD)

    # ── Save in Vocabulary.load() format ─────────────────────
    def to_vocab_format(w2i: dict) -> dict:
        i2w = {str(v): k for k, v in w2i.items()}
        return {"word2idx": w2i, "idx2word": i2w, "idx": len(w2i)}

    with open(VOCAB_Q_OUT, "w") as f:
        json.dump(to_vocab_format(vocab_q), f)
    with open(VOCAB_A_OUT, "w") as f:
        json.dump(to_vocab_format(vocab_a), f)

    # ── Report ────────────────────────────────────────────────
    def _latest_backup(tag):
        candidates = [os.path.join(BACKUP_DIR, x) for x in os.listdir(BACKUP_DIR)
                      if tag in x and x.endswith(".json")]
        return max(candidates, key=os.path.getmtime) if candidates else ""

    def _vocab_len(path):
        if not path or not os.path.exists(path):
            return 0
        d = load_existing_vocab(path)
        if "word2idx" in d:
            return len(d["word2idx"])
        return len(d)

    old_q_n = _vocab_len(_latest_backup("questions"))
    old_a_n = _vocab_len(_latest_backup("answers"))

    total_q_occ = sum(q_counter.values())
    total_a_occ = sum(a_counter.values())
    q_coverage  = sum(c for w, c in q_counter.items() if w in vocab_q) / total_q_occ * 100
    a_coverage  = sum(c for w, c in a_counter.items() if w in vocab_a) / total_a_occ * 100

    print()
    print("=" * 55)
    print("Vocabulary Rebuild Report")
    print("=" * 55)
    print(f"  Threshold: {THRESHOLD} (token must appear >= {THRESHOLD}x)")
    print()
    print(f"  {'':20s}  {'Old':>8}  {'New':>8}  {'Delta':>8}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'Question vocab':20s}  {old_q_n:>8,}  {len(vocab_q):>8,}  {len(vocab_q)-old_q_n:>+8,}")
    print(f"  {'Answer vocab':20s}  {old_a_n:>8,}  {len(vocab_a):>8,}  {len(vocab_a)-old_a_n:>+8,}")
    print()
    print(f"  Question token coverage: {q_coverage:.2f}%")
    print(f"  Answer token coverage:   {a_coverage:.2f}%")
    print()

    # New tokens by source
    sources = ["vqa_x", "aokvqa", "vqa_e"]
    src_new_tokens = {}
    for src in sources:
        src_items = [d for d in filtered if d["source"] == src]
        src_counter = Counter()
        for d in src_items:
            src_counter.update(tokenize(d["explanation"][0]))
            src_counter.update(tokenize(d["multiple_choice_answer"]))
        # tokens in new vocab but not in old vocab (compare against backed-up vocab)
        old_backup = _latest_backup("answers_8813")
        old_set = set()
        if old_backup and os.path.exists(old_backup):
            raw = load_existing_vocab(old_backup)
            old_set = set(raw.get("word2idx", raw).keys())
        new_toks = [w for w in src_counter if w in vocab_a and w not in old_set]
        src_new_tokens[src] = len(new_toks)

    print("  New answer tokens contributed by source:")
    for src in sources:
        print(f"    {src:<12} +{src_new_tokens[src]:,}")
    print()
    print(f"Saved: {VOCAB_Q_OUT}  ({len(vocab_q):,} tokens)")
    print(f"Saved: {VOCAB_A_OUT}  ({len(vocab_a):,} tokens)")

    # Spot-check: verify specials at correct indices
    print()
    print("Special token indices:")
    for tok in SPECIALS:
        print(f"  {tok:<10} → {vocab_a[tok]}")


if __name__ == "__main__":
    main()
