import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vocab import Vocabulary

# VQA-E single annotation file (contains question + answer + explanation)
TRAIN_VQA_E_JSON = "data/vqa_e/VQA-E_train_set.json"

# VQA v2.0 files
TRAIN_VQA_V2_Q = "data/raw/vqa_v2/v2_OpenEnded_mscoco_train2014_questions.json"
TRAIN_VQA_V2_A = "data/raw/vqa_v2/v2_mscoco_train2014_annotations.json"

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

    # Load VQA v2.0 Data
    questions_v2 = []
    answers_v2 = []
    try:
        print(f"\nReading VQA v2.0 questions: {TRAIN_VQA_V2_Q}")
        with open(TRAIN_VQA_V2_Q, 'r') as f:
            q_data = json.load(f)
            questions_v2 = [q['question'] for q in q_data.get('questions', [])]
        print(f"Loaded {len(questions_v2)} VQA v2.0 questions.")
        
        print(f"Reading VQA v2.0 annotations: {TRAIN_VQA_V2_A}")
        with open(TRAIN_VQA_V2_A, 'r') as f:
            a_data = json.load(f)
            answers_v2 = [a.get('multiple_choice_answer', '') for a in a_data.get('annotations', [])]
        print(f"Loaded {len(answers_v2)} VQA v2.0 answers.")
    except FileNotFoundError as e:
        print(f"WARNING: VQA v2.0 files not found, proceeding with VQA-E only. Details: {e}")

    # Build question vocabulary from 'question' field
    print("\n1. Building question vocabulary...")
    questions_list = [ann['question'] for ann in annotations if 'question' in ann]
    questions_list.extend(questions_v2)
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
    
    answers_list.extend(answers_v2)

    # threshold=3: VQA-E is ~6x smaller than VQA 2.0, need to keep more words
    a_vocab = Vocabulary()
    a_vocab.build(answers_list, threshold=3)

    # Force-Add Tier 6 CSS Tokens
    ABSTENTION_VISUAL = "i cannot answer because the object is hidden"
    ABSTENTION_LING = "i cannot answer because the question is unclear"
    print("\n3. Force-adding Tier 6 CSS Tokens...")
    for sentence in [ABSTENTION_VISUAL, ABSTENTION_LING]:
        for word in a_vocab.tokenize(sentence):
            if word not in a_vocab.word2idx:
                a_vocab.add_word(word)

    a_out_path = os.path.join(OUTPUT_DIR, 'vocab_answers.json')
    a_vocab.save(a_out_path)
    print(f"   Vocab size: {len(a_vocab)} | Saved to: {a_out_path}")
    print(f"\nSample answer texts:")
    for ann in annotations[:3]:
        answer = ann.get('multiple_choice_answer', '')
        exp = ann.get('explanation', [''])[0] if isinstance(ann.get('explanation'), list) else ''
        print(f"  Q: {ann.get('question', '')}")
        print(f"  A: {answer} because {exp}\n")

if __name__ == '__main__':
    main()
