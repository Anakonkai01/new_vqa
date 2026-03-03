SIÊU KẾ HOẠCH TÁI CẤU TRÚC DỰ ÁN VQA-E (MASTER BLUEPRINT) - BẢN CHI TIẾT

Mục tiêu: Nâng cấp dự án VQA cơ bản thành hệ thống Generative VQA-E kết hợp Học đa nhiệm (Multi-task), FiLM Fusion, Reinforcement Learning (SCST) và LLM-as-a-Judge.

GIAI ĐOẠN 1: DATA OVERHAUL & MULTI-TASK PIPELINE (Thay máu dữ liệu)

1.1. Cập nhật src/scripts/1_build_vocab.py

Thuật toán xây dựng từ vựng phải chuyển từ việc coi cả câu trả lời là 1 ID sang việc tách từ (word-level tokenization).

Mã nguồn lõi cần thay đổi:

import nltk
from collections import Counter

# Tải bộ công cụ tách từ
nltk.download('punkt')

def build_multitask_vocab(vqa_e_json_path, coco_captions_json_path, freq_threshold=3):
    counter = Counter()
    
    # 1. Quét dữ liệu VQA-E
    with open(vqa_e_json_path, 'r') as f:
        vqa_data = json.load(f)
        for ann in vqa_data['annotations']:
            # Xử lý Question
            q_tokens = nltk.word_tokenize(ann['question'].lower())
            counter.update(q_tokens)
            
            # Xử lý Answer + Explanation
            # Ghép chuỗi: "yes because the man is holding an umbrella"
            full_ans = f"{ann['answer']} because {ann['explanation']}"
            a_tokens = nltk.word_tokenize(full_ans.lower())
            counter.update(a_tokens)
            
    # 2. Quét dữ liệu COCO Captions (Nếu áp dụng Multi-task)
    with open(coco_captions_json_path, 'r') as f:
        caption_data = json.load(f)
        for ann in caption_data['annotations']:
            c_tokens = nltk.word_tokenize(ann['caption'].lower())
            counter.update(c_tokens)

    # 3. Lọc từ theo tần suất và tạo word2idx, idx2word
    # (Giữ nguyên logic thêm <pad>, <start>, <end>, <unk> của bạn)
    ...


1.2. Nâng cấp src/dataset.py thành MultiTaskDataset

Lớp Dataset cần linh hoạt trả về dữ liệu tùy thuộc vào Task đang được train (Image Captioning hay VQA-E).

Mã nguồn lõi:

class MultiTaskVQADataset(Dataset):
    def __init__(self, mode='vqa_e', ...):
        self.mode = mode # 'vqa_e' hoặc 'caption'
        # ... logic load JSON tương ứng ...

    def __getitem__(self, index):
        img_tensor = self._load_image(...)
        
        if self.mode == 'caption':
            caption_text = self.captions[index]
            tokens = [self.vocab.word2idx['<start>']] + \
                     self.vocab.numericalize(caption_text) + \
                     [self.vocab.word2idx['<end>']]
            target_tensor = torch.tensor(tokens, dtype=torch.long)
            
            # Trả về tensor rỗng cho Question để giữ chung format collate_fn
            empty_q = torch.tensor([self.vocab.word2idx['<pad>']], dtype=torch.long)
            return img_tensor, empty_q, target_tensor
            
        elif self.mode == 'vqa_e':
            q_text = self.questions[index]
            ans_exp_text = f"{self.answers[index]} because {self.explanations[index]}"
            
            q_tensor = torch.tensor(self.vocab.numericalize(q_text), dtype=torch.long)
            
            tokens = [self.vocab.word2idx['<start>']] + \
                     self.vocab.numericalize(ans_exp_text) + \
                     [self.vocab.word2idx['<end>']]
            target_tensor = torch.tensor(tokens, dtype=torch.long)
            
            return img_tensor, q_tensor, target_tensor


GIAI ĐOẠN 2: ARCHITECTURE UPGRADE (FiLM & Tối ưu Trạng thái khởi tạo)

Mở file src/models/vqa_models.py và src/models/decoder_attention.py để nâng cấp.

2.1. Nâng cấp Trạng thái Khởi tạo (Init States)

Không gán trực tiếp h_0 = fusion. Hãy cho phép mô hình học cách khởi tạo bộ nhớ tế bào (Cell State) một cách phi tuyến tính.
Trong lớp VQAModelD (hoặc C):

# Trong __init__:
self.init_h = nn.Linear(1024, decoder_hidden_size)
self.init_c = nn.Linear(1024, decoder_hidden_size)

# Trong forward:
fusion = img_global_feat * q_feat # (B, 1024)
h_0 = torch.tanh(self.init_h(fusion)).unsqueeze(0).repeat(self.num_layers, 1, 1)
c_0 = torch.tanh(self.init_c(fusion)).unsqueeze(0).repeat(self.num_layers, 1, 1)
hidden = (h_0, c_0)


2.2. Kỹ thuật FiLM (Feature-wise Linear Modulation)

Thay vì dùng phép nhân Hadamard thô sơ, ta dùng vector câu hỏi $q$ dự đoán 2 vector $\gamma$ và $\beta$ để biến đổi ma trận ảnh.
Công thức: $Img_{new} = \gamma(q) \odot Img_{old} + \beta(q)$

# Trong __init__ của VQAModelD:
self.film_gamma = nn.Linear(1024, 1024) # 1024 là dim của q_feat và img_features
self.film_beta = nn.Linear(1024, 1024)

# Trong forward (trước khi đưa vào Decoder):
# q_feat: (B, 1024) -> gamma/beta: (B, 1, 1024) để broadcast với ảnh (B, 49, 1024)
gamma = self.film_gamma(q_feat).unsqueeze(1) 
beta = self.film_beta(q_feat).unsqueeze(1)

# Áp dụng FiLM lên 49 vùng ảnh
img_features_modulated = (gamma * img_features) + beta

# Truyền img_features_modulated vào decoder có attention
logits = self.decoder(hidden, img_features_modulated, target_seq)


GIAI ĐOẠN 3: ĐỈNH CAO RL - SELF-CRITICAL SEQUENCE TRAINING (SCST)

Tạo file mới src/train_rl.py. Đây là phương pháp tối ưu trực tiếp trên thước đo (metric) thay vì hàm loss CrossEntropy.

Thuật toán Lõi (Toán học):

Sinh chuỗi $\hat{Y}$ bằng thuật toán Greedy (Lấy từ có xác suất cao nhất). Tính phần thưởng $R(\hat{Y})$ (VD: dùng hàm BLEU-4).

Sinh chuỗi $Y^s$ bằng cách Lấy mẫu (Multinomial Sampling). Tính phần thưởng $R(Y^s)$. Tính kèm cả log xác suất của các từ được chọn $\log P(y^s_t)$.

Hàm Loss REINFORCE: $\mathcal{L}(\theta) = - (R(Y^s) - R(\hat{Y})) \sum_{t=1}^T \log P(y^s_t | y^s_{<t})$

Mã nguồn PyTorch giả mã (Pseudocode):

def train_step_rl(model, imgs, qs, ground_truth_texts, optimizer):
    model.train()
    
    # 1. Sinh chuỗi Baseline (Greedy) - Không tính đạo hàm
    with torch.no_grad():
        greedy_seqs = model.sample(imgs, qs, method='greedy')
        # Tính phần thưởng (BLEU) so với nhãn gốc
        baseline_rewards = compute_bleu_batch(greedy_seqs, ground_truth_texts)
        # baseline_rewards: tensor shape (B,)
        
    # 2. Sinh chuỗi Lấy mẫu ngẫu nhiên (Multinomial) - CÓ tính đạo hàm
    sample_seqs, log_probs = model.sample(imgs, qs, method='sample')
    # Tính phần thưởng cho câu được lấy mẫu
    sample_rewards = compute_bleu_batch(sample_seqs, ground_truth_texts)
    # sample_rewards: tensor shape (B,)
    
    # 3. Tính Lợi thế (Advantage)
    advantage = sample_rewards - baseline_rewards # (B,)
    
    # 4. Tính Loss bằng REINFORCE
    # log_probs là tổng log xác suất của các từ trong câu sample: shape (B,)
    loss = - (advantage * log_probs).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item(), sample_rewards.mean().item()


(Lưu ý: Bạn phải viết thêm hàm model.sample trong Decoder để hỗ trợ việc lấy mẫu từ hàm phân phối Softmax).

GIAI ĐOẠN 4: LLM-AS-A-JUDGE (Đánh giá Hiện đại)

Thay vì viết code tính Exact Match (điều vô nghĩa với VQA-E), tạo file src/llm_eval.py.

Cài đặt & Mã nguồn:
pip install google-generativeai

import google.generativeai as genai
import json

genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel('gemini-1.5-flash')

def evaluate_with_llm(question, reference, prediction):
    prompt = f"""
    Bạn là một chuyên gia AI làm nhiệm vụ chấm điểm mô hình sinh tự động.
    Bài toán: Visual Question Answering with Explanations.
    
    Câu hỏi: {question}
    Nhãn gốc (Tham chiếu): {reference}
    Dự đoán của mô hình: {prediction}
    
    Hãy chấm điểm dự đoán trên thang điểm từ 1 đến 5 (1: Sai hoàn toàn, 5: Hoàn hảo).
    Tiêu chí: Mức độ tương đồng về ngữ nghĩa đáp án VÀ tính logic của lời giải thích. Bỏ qua các lỗi ngữ pháp hoặc chính tả nhỏ.
    
    Phản hồi DUY NHẤT bằng JSON theo cấu trúc: {{"score": <điểm>, "reason": "<lý do>"}}
    """
    
    response = model.generate_content(prompt)
    try:
        # Làm sạch chuỗi JSON nếu LLM sinh ra markdown code block
        json_str = response.text.strip().replace('```json', '').replace('```', '')
        result = json.loads(json_str)
        return result['score'], result['reason']
    except Exception as e:
        return 0, str(e)

# Ví dụ sử dụng trong vòng lặp validation
# total_score += evaluate_with_llm(q_text, gt_text, pred_text)[0]
# llm_accuracy = (total_score / num_samples) / 5.0 * 100


Lời khuyên Cuối cùng từ Cố vấn:
Đừng làm tất cả cùng lúc. Hãy đi theo đúng trình tự:

Sửa Dữ liệu (Giai đoạn 1) -> Đảm bảo train được 1 epoch bằng CrossEntropy hiện tại.

Thêm FiLM (Giai đoạn 2) -> Train lại, xem Loss có giảm mượt hơn không.

Chạy LLM Eval (Giai đoạn 4) để xem điểm ngữ nghĩa thực tế thay vì Exact Match.

Triển khai RL (Giai đoạn 3) ở những tuần cuối cùng vì nó đòi hỏi kỹ năng gỡ lỗi (debug) toán học rất cao.