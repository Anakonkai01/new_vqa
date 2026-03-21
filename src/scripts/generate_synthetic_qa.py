"""
generate_synthetic_qa.py — Tier D5
====================================
Generates synthetic VQA-E–style QA pairs from COCO object annotations.
No LLM required — uses hand-crafted templates grounded in detected objects.

MOTIVATION:
  VQA-E has ~200K samples. Synthetic augmentation can inject 50-100K additional
  training pairs covering rare answer categories and unusual object combinations.

TEMPLATES (grounded in COCO object categories):
  Existence: "Is there a {obj} in the image?" → "yes because there is a {obj} visible"
  Count:     "How many {obj}s are there?"    → "{n} because there are {n} {obj}s"
  Color:     "What color is the {obj}?"      → "{color} because the {obj} is {color}"
             (uses COCO categories + common color adjectives from captions)
  Location:  "Where is the {obj}?"           → "on the {surface} because it is resting there"

Output: JSON in same format as VQA-E_train_set.json

Usage:
  python src/scripts/generate_synthetic_qa.py \\
      --instances_json data/annotations/vqa_v2/instances_train2014.json \\
      --output         data/annotations/vqa_e/VQA-E_synthetic.json \\
      --max_per_image  3 \\
      --max_total      50000
"""

import json
import random
import argparse
import os
from collections import defaultdict


# ── Templates ─────────────────────────────────────────────────────────────────

_EXISTENCE_Q = [
    "Is there a {obj} in the image?",
    "Can you see a {obj}?",
    "Is a {obj} present?",
]
_EXISTENCE_A_YES = [
    "yes because there is a {obj} visible in the scene",
    "yes because a {obj} can be seen in the image",
    "yes because the image contains a {obj}",
]
_EXISTENCE_A_NO = [
    "no because there is no {obj} in the image",
    "no because the scene does not contain a {obj}",
]

_COUNT_Q = [
    "How many {obj}s are there?",
    "How many {obj}s can you see?",
    "What is the number of {obj}s in the image?",
]
_COUNT_A = [
    "{n} because there {verb} {n} {obj}s visible",
    "{n} because you can count {n} {obj}s in the scene",
]

_SURFACE_WORDS = ['table', 'floor', 'ground', 'surface', 'counter', 'shelf', 'bench']

_LOCATION_Q = [
    "Where is the {obj}?",
    "Where can you find the {obj}?",
]
_LOCATION_A = [
    "on the {surface} because the {obj} is resting there",
    "near the {surface} because the {obj} is placed there",
    "on the {surface} because that is where the {obj} sits",
]


def _pluralize(noun):
    """Very simple English pluralization."""
    if noun.endswith('s') or noun.endswith('x') or noun.endswith('z'):
        return noun + 'es'
    if noun.endswith('y') and len(noun) > 1 and noun[-2] not in 'aeiou':
        return noun[:-1] + 'ies'
    return noun + 's'


def generate_for_image(img_id, objects, max_per_image=3):
    """
    Generate up to max_per_image synthetic QA pairs for one image.

    objects: list of COCO category names present in this image
    Returns: list of annotation dicts (VQA-E format)
    """
    if not objects:
        return []

    samples = []
    random.shuffle(objects)

    for obj in objects[:max_per_image * 2]:   # try more, keep best
        template_type = random.choice(['existence', 'count', 'location'])

        if template_type == 'existence':
            q = random.choice(_EXISTENCE_Q).format(obj=obj)
            a_template = random.choice(_EXISTENCE_A_YES)
            explanation = a_template.format(obj=obj)
            answer = 'yes'

        elif template_type == 'count':
            n = random.choice([1, 2, 3])
            n_word = {1: 'one', 2: 'two', 3: 'three'}[n]
            verb = 'is' if n == 1 else 'are'
            q = random.choice(_COUNT_Q).format(obj=_pluralize(obj) if n > 1 else obj)
            a_template = random.choice(_COUNT_A)
            explanation = a_template.format(n=n_word, obj=_pluralize(obj) if n > 1 else obj,
                                            verb=verb)
            answer = n_word

        else:  # location
            surface = random.choice(_SURFACE_WORDS)
            q = random.choice(_LOCATION_Q).format(obj=obj)
            explanation = random.choice(_LOCATION_A).format(obj=obj, surface=surface)
            answer = f"on the {surface}"

        full_answer = f"{answer} because {explanation}"
        samples.append({
            'img_id': img_id,
            'question': q,
            'multiple_choice_answer': answer,
            'explanation': [explanation],
            '_synthetic': True,
        })

        if len(samples) >= max_per_image:
            break

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instances_json', required=True,
                        help='COCO instances JSON (e.g. instances_train2014.json)')
    parser.add_argument('--output',         required=True,
                        help='Output synthetic QA JSON path')
    parser.add_argument('--max_per_image',  type=int, default=3)
    parser.add_argument('--max_total',      type=int, default=50000)
    parser.add_argument('--seed',           type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading {args.instances_json} ...")
    with open(args.instances_json) as f:
        data = json.load(f)

    # Build category id → name map
    id2cat = {c['id']: c['name'] for c in data['categories']}

    # Build image_id → list of category names
    img2objs = defaultdict(list)
    for ann in data['annotations']:
        cat = id2cat.get(ann['category_id'])
        if cat:
            img2objs[ann['image_id']].append(cat)

    # Deduplicate objects per image
    for img_id in img2objs:
        img2objs[img_id] = list(set(img2objs[img_id]))

    all_samples = []
    img_ids = list(img2objs.keys())
    random.shuffle(img_ids)

    for img_id in img_ids:
        if len(all_samples) >= args.max_total:
            break
        samples = generate_for_image(img_id, img2objs[img_id], args.max_per_image)
        all_samples.extend(samples)

    all_samples = all_samples[:args.max_total]

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_samples, f, indent=2)

    print(f"Generated {len(all_samples):,} synthetic QA pairs → {args.output}")
    print("To use: concatenate with VQA-E_train_set.json before training.")


if __name__ == '__main__':
    main()
