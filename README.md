# **Bài 2 (7đ): VQA Project**

> **[IMPORTANT NOTICE]**
> This repository has been heavily upgraded to feature **Model E** combining CLIP ViT, BiLSTM, FiLM Fusion, and Reinforcement Learning on VQA-E dataset.
> For the new, final and complete codebase architecture and training instructions, please read `CONTEXT_HANDOFF.md`.
> To run the training pipeline, see `train_model_e.ipynb`.

---

Sử dụng model LSTM và model CNN xây dựng kiến trúc cho bài toán Visual Question Answering.

**Input**: ảnh và câu hỏi

**Output**: câu trả lời. Lưu ý câu trả lời phải được sinh ra bởi LSTM-decoder

Xây dựng các loại kiến trúc khác nhau dựa trên các đặc điểm:

1) Không có và có dùng cơ chế Attention

2) Train từ đầu và có dùng Pretrained model

Đánh giá các mô hình này dựa vào độ đo nào? Hãy so sánh các mô hình xây dựng ở trên thông qua các độ đo này. 

