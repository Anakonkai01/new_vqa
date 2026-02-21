import os
import json
import h5py
import numpy as np
from PIL import Image

def create_dummy_data():
    print("⏳ Đang tạo dữ liệu giả (Dummy Data)...")
    
    # Tạo các thư mục nếu chưa có
    os.makedirs("data/raw/vqa_json", exist_ok=True)
    os.makedirs("data/raw/images/train2014", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    NUM_SAMPLES = 100

    # 0. Tạo ảnh giả (random noise) cho Model A (CNN scratch load raw images)
    for i in range(NUM_SAMPLES):
        img_name = f"COCO_train2014_{i:012d}.jpg"
        img_path = os.path.join("data/raw/images/train2014", img_name)
        # Tạo ảnh RGB ngẫu nhiên 224x224
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(dummy_img).save(img_path)
    print(f"✅ Đã tạo {NUM_SAMPLES} ảnh giả: data/raw/images/train2014/")

    # 1. Tạo file H5 giả — shape (N, 2048) sau khi đã mean pool 14x14
    dummy_features = np.random.randn(NUM_SAMPLES, 2048).astype(np.float32)
    dummy_ids = [f"COCO_train2014_{i:012d}.jpg".encode('utf-8') for i in range(NUM_SAMPLES)]

    with h5py.File("data/processed/train_features.h5", "w") as f:
        f.create_dataset("features", data=dummy_features)
        f.create_dataset("ids", data=dummy_ids)
    print("✅ Đã tạo file H5 giả: data/processed/train_features.h5")

    # 2. Tạo file JSON câu hỏi giả
    questions = []
    for i in range(NUM_SAMPLES):
        questions.append({
            "question_id": i,
            "image_id": i,
            "question": "what color is the cat" if i % 2 == 0 else "where is the dog"
        })
    with open("data/raw/vqa_json/v2_OpenEnded_mscoco_train2014_questions.json", "w") as f:
        json.dump({"questions": questions}, f)
    print("✅ Đã tạo file Questions JSON giả")

    # 3. Tạo file JSON câu trả lời giả
    annotations = []
    for i in range(NUM_SAMPLES):
        annotations.append({
            "question_id": i,
            "multiple_choice_answer": "red" if i % 2 == 0 else "outside"
        })
    with open("data/raw/vqa_json/v2_mscoco_train2014_annotations.json", "w") as f:
        json.dump({"annotations": annotations}, f)
    print("✅ Đã tạo file Annotations JSON giả")

if __name__ == "__main__":
    create_dummy_data()