import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vocab import Vocabulary

# VQA-E single annotation file (contains question + answer + explanation)
TRAIN_VQA_E_JSON = "data/annotations/vqa_e/VQA-E_train_set.json"

# VQA v2.0 files
TRAIN_VQA_V2_Q = "data/annotations/vqa_v2/v2_OpenEnded_mscoco_train2014_questions.json"
TRAIN_VQA_V2_A = "data/annotations/vqa_v2/v2_mscoco_train2014_annotations.json"

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
        print("Please download VQA-E and place it in data/annotations/vqa_e/")
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

    # Build joint vocabulary
    print("\n1. Building joint vocabulary...")
    joint_texts = []
    
    # 1. VQA-E Questions
    joint_texts.extend([ann['question'] for ann in annotations if 'question' in ann])
    # 2. VQA-E Answers + Explanations
    for ann in annotations:
        answer = ann.get('multiple_choice_answer', '')
        explanation_list = ann.get('explanation', [])
        explanation = explanation_list[0] if explanation_list and isinstance(explanation_list[0], str) else ''
        if explanation:
            joint_texts.append(f"{answer} because {explanation}")
        else:
            joint_texts.append(answer)

    # 3. VQA v2.0 Questions
    joint_texts.extend(questions_v2)
    # 4. VQA v2.0 Answers
    joint_texts.extend(answers_v2)

    joint_vocab = Vocabulary()
    joint_vocab.build(joint_texts, threshold=3)

    # 5. Force-Add Tier 6 CSS Tokens
    ABSTENTION_VISUAL = "i cannot answer because the object is hidden"
    ABSTENTION_LING = "i cannot answer because the question is unclear"
    print("\n2. Force-adding Tier 6 CSS Tokens...")
    for sentence in [ABSTENTION_VISUAL, ABSTENTION_LING]:
        for word in joint_vocab.tokenize(sentence):
            if word not in joint_vocab.word2idx:
                joint_vocab.add_word(word)

    out_path = os.path.join(OUTPUT_DIR, 'vocab_joint.json')
    joint_vocab.save(out_path)
    print(f"   Joint Vocab size: {len(joint_vocab)} | Saved to: {out_path}")
    
    print(f"\nSample texts:")
    for ann in annotations[:3]:
        answer = ann.get('multiple_choice_answer', '')
        exp = ann.get('explanation', [''])[0] if isinstance(ann.get('explanation'), list) else ''
        print(f"  Q: {ann.get('question', '')}")
        print(f"  A: {answer} because {exp}\n")

if __name__ == '__main__':
    main()
