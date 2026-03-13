import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vocab import Vocabulary

# VQA-E single annotation file (contains question + answer + explanation)
TRAIN_VQA_E_PATHS = [
    "data/vqa_e/VQA-E_train_set.json",
    "data/raw/vqa_e_json/VQA-E_train_set.json",
]
TRAIN_CAPTIONS_PATHS = [
    "data/raw/annotations/captions_train2014.json",
    "data/vqa_data_json/captions_train2014.json"
]

OUTPUT_DIR = "data/processed"


def get_first_existing_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    vqa_path = get_first_existing_path(TRAIN_VQA_E_PATHS)
    if not vqa_path:
        print(f"ERROR: VQA-E train set not found in any of {TRAIN_VQA_E_PATHS}")
        print("Please download VQA-E and place it in data/vqa_e/")
        return
        
    print(f"\nReading VQA-E annotation file: {vqa_path}")
    with open(vqa_path, 'r') as f:
        annotations = json.load(f)  # root is a list


    print(f"Loaded {len(annotations)} VQA-E annotations.")
    
    cap_path = get_first_existing_path(TRAIN_CAPTIONS_PATHS)
    captions = []
    if cap_path:
        print(f"Reading COCO Captions file: {cap_path}")
        with open(cap_path, 'r') as f:
            captions = json.load(f)['annotations']
        print(f"Loaded {len(captions)} COCO Captions.")
    else:
        print("COCO Captions not found, vocab will be built for VQA-E only.")

    # Build question vocabulary from 'question' field
    print("\n1. Building question vocabulary...")
    questions_list = [ann['question'] for ann in annotations if 'question' in ann]
    q_vocab = Vocabulary()
    q_vocab.build(questions_list, threshold=3)
    q_out_path = os.path.join(OUTPUT_DIR, 'vocab_questions.json')
    q_vocab.save(q_out_path)
    print(f"   Vocab size: {len(q_vocab)} | Saved to: {q_out_path}")

    # Build answer vocabulary from 'answer + because + explanation'
    # VQA-E format: 'answer' field (some versions use 'multiple_choice_answer')
    # and 'explanation' field (a list, take first element)
    print("\n2. Building answer vocabulary (answer + explanation)...")
    answers_list = []
    for ann in annotations:
        answer = ann.get('multiple_choice_answer', '')
        explanation_list = ann.get('explanation', [])
        # explanation = [text_string, confidence_score] — take index 0
        explanation = explanation_list[0] if explanation_list and isinstance(explanation_list[0], str) else ''
        if explanation:
            a_text = f"{answer} because {explanation}"
        else:
            a_text = answer
        answers_list.append(a_text)
        
    for cap in captions:
        answers_list.append(cap['caption'])

    # threshold=3: VQA-E is ~6x smaller than VQA 2.0, need to keep more words
    a_vocab = Vocabulary()
    a_vocab.build(answers_list, threshold=3)
    a_out_path = os.path.join(OUTPUT_DIR, 'vocab_answers.json')
    a_vocab.save(a_out_path)
    print(f"   Vocab size: {len(a_vocab)} | Saved to: {a_out_path}")
    print(f"\nSample answer texts:")
    for ann in annotations[:3]:
        answer = ann.get('multiple_choice_answer', '')
        exp = ann.get('explanation', [''])[0]
        print(f"  Q: {ann['question']}")
        print(f"  A: {answer} because {exp}\n")

if __name__ == '__main__':
    main()
