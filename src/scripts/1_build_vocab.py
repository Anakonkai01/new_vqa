import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vocab import Vocabulary

# VQA-E single annotation file (contains question + answer + explanation)
TRAIN_VQA_E_JSON = "data/raw/vqa_e_json/VQA-E_train_set.json"

OUTPUT_DIR = "data/processed"


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    print(f"\nReading VQA-E annotation file: {TRAIN_VQA_E_JSON}")
    try:
        with open(TRAIN_VQA_E_JSON, 'r') as f:
            annotations = json.load(f)  # root is a list
    except FileNotFoundError:
        print(f"ERROR: File not found: {TRAIN_VQA_E_JSON}")
        print("Please download VQA-E and place it in data/vqa_e/")
        return

    print(f"Loaded {len(annotations)} annotations.")

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
