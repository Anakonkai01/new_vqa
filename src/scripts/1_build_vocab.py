import sys
import os
import json

# --- 1. SETUP ĐƯỜNG DẪN IMPORT (Giữ nguyên) ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vocab import Vocabulary

# ==============================================================================
# ⚙️ CẤU HÌNH (BẠN CHỈ CẦN SỬA Ở ĐÂY)
# ==============================================================================

# Đường dẫn đến file câu hỏi (Train)
QUESTION_JSON_PATH = "data/raw/vqa_json/v2_OpenEnded_mscoco_train2014_questions.json"

# Đường dẫn đến file câu trả lời (Train)
ANNOTATION_JSON_PATH = "data/raw/vqa_json/v2_mscoco_train2014_annotations.json"

# Nơi bạn muốn lưu file từ điển sau khi tạo xong
OUTPUT_DIR = "data/processed"

# ==============================================================================

def main():
    # Kiểm tra xem folder output có chưa, chưa có thì tạo
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 Đã tạo thư mục: {OUTPUT_DIR}")

    # --- 2. XỬ LÝ CÂU HỎI (QUESTION) ---
    print(f"\n🔹 1. Đang đọc file câu hỏi: {QUESTION_JSON_PATH}")
    try:
        with open(QUESTION_JSON_PATH, 'r') as f:
            questions_data = json.load(f)['questions']
    except FileNotFoundError:
        print(f"❌ LỖI: Không tìm thấy file {QUESTION_JSON_PATH}")
        print("👉 Bạn hãy kiểm tra lại đường dẫn ở phần 'CẤU HÌNH' bên trên nhé.")
        return

    print("🔨 Đang xây dựng từ điển câu hỏi...")
    # Lấy toàn bộ nội dung câu hỏi
    questions_list = [q['question'] for q in questions_data]
    
    # Tạo vocab (Lọc từ xuất hiện >= 3 lần)
    q_vocab = Vocabulary()
    q_vocab.build(questions_list, threshold=3)
    
    # Lưu file
    q_out_path = os.path.join(OUTPUT_DIR, 'vocab_questions.json')
    q_vocab.save(q_out_path)
    print(f"✅ Xong! Đã lưu tại: {q_out_path}")

    # --- 3. XỬ LÝ CÂU TRẢ LỜI (ANSWER) ---
    print(f"\n🔹 2. Đang đọc file annotations: {ANNOTATION_JSON_PATH}")
    try:
        with open(ANNOTATION_JSON_PATH, 'r') as f:
            annotations_data = json.load(f)['annotations']
    except FileNotFoundError:
        print(f"❌ LỖI: Không tìm thấy file {ANNOTATION_JSON_PATH}")
        return

    print("🔨 Đang xây dựng từ điển câu trả lời...")
    # Lấy câu trả lời phổ biến nhất (multiple_choice_answer)
    answers_list = [ann['multiple_choice_answer'] for ann in annotations_data]
    
    # Tạo vocab (Lọc kỹ hơn, threshold=5)
    a_vocab = Vocabulary()
    a_vocab.build(answers_list, threshold=5)
    
    # Lưu file
    a_out_path = os.path.join(OUTPUT_DIR, 'vocab_answers.json')
    a_vocab.save(a_out_path)
    print(f"✅ Xong! Đã lưu tại: {a_out_path}")

if __name__ == '__main__':
    main()