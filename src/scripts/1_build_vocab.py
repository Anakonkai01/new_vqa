import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vocab import Vocabulary

QUESTION_JSON_PATH = "data/raw/vqa_json/v2_OpenEnded_mscoco_train2014_questions.json"

ANNOTATION_JSON_PATH = "data/raw/vqa_json/v2_mscoco_train2014_annotations.json"

OUTPUT_DIR = "data/processed"


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # handdle question
    print(f"\n1. Reading question file: {QUESTION_JSON_PATH}")
    try:
        with open(QUESTION_JSON_PATH, 'r') as f:
            questions_data = json.load(f)['questions']
    except FileNotFoundError:
        print(f"ERROR: File not found: {QUESTION_JSON_PATH}")
        print("Please check the file path in the configuration section above.")
        return

    print("Building question vocabulary...")
    # Extract all question texts
    questions_list = [q['question'] for q in questions_data]
    
    # Build vocab (filter words appearing >= 3 times)
    q_vocab = Vocabulary()
    q_vocab.build(questions_list, threshold=3)
    
    # Save file
    q_out_path = os.path.join(OUTPUT_DIR, 'vocab_questions.json')
    q_vocab.save(q_out_path)
    print(f"Done! Saved to: {q_out_path}")

    # haddle answer 
    print(f"\n2. Reading annotation file: {ANNOTATION_JSON_PATH}")
    try:
        with open(ANNOTATION_JSON_PATH, 'r') as f:
            annotations_data = json.load(f)['annotations']
    except FileNotFoundError:
        print(f"ERROR: File not found: {ANNOTATION_JSON_PATH}")
        return

    print("Building answer vocabulary...")
    # Get the most common answer (multiple_choice_answer)
    answers_list = [ann['multiple_choice_answer'] for ann in annotations_data]
    
    # Build vocab (stricter filter, threshold=5)
    a_vocab = Vocabulary()
    a_vocab.build(answers_list, threshold=5)
    
    # Save file
    a_out_path = os.path.join(OUTPUT_DIR, 'vocab_answers.json')
    a_vocab.save(a_out_path)
    print(f"Done! Saved to: {a_out_path}")

if __name__ == '__main__':
    main()