import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import h5py
import json
from vocab import Vocabulary 

class VQADataset(Dataset):
    def __init__(self, feature_h5_path, question_json_path, annotations_json_path, vocab_q, vocab_a):
        self.feature_h5_path = feature_h5_path
        self.vocab_q = vocab_q
        self.vocab_a = vocab_a 
        
        # read question and annotations json into ram 
        with open(question_json_path, 'r') as f: 
            self.questions = json.load(f)['questions']

        with open(annotations_json_path, 'r') as f: 
            annotations = json.load(f)['annotations']

        # 2. Xây dựng Hash Map (Dictionary) ánh xạ Question_ID -> Answer
        # Độ phức tạp tra cứu: O(1)
        self.qid2ans = {ann['question_id']: ann['multiple_choice_answer'] for ann in annotations}

        
        # 3. Xây dựng Hash Map ánh xạ Tên ảnh -> Vị trí (Index) trong file H5
        # Tránh việc phải dùng vòng lặp tìm kiếm tên ảnh mỗi khi load 1 sample
        self.name2idx = {}
        with h5py.File(self.feature_h5_path, 'r') as f:
            h5_ids = f['ids'][:]
            for idx, name_bytes in enumerate(h5_ids):
                # Giải mã bytes thành chuỗi (ví dụ: b'COCO_train2014_000000123.jpg' -> 'COCO_train2014_000000123.jpg')
                name_str = name_bytes.decode('utf-8')
                self.name2idx[name_str] = idx

        # H5py File Object (Khởi tạo None để tránh lỗi multiprocessing của DataLoader)
        self.h5_file = None
        
    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        # Mở file h5py ở lần gọi đầu tiên của mỗi worker
        if self.h5_file is None:
            self.h5_file = h5py.File(self.feature_h5_path, 'r')

        # Lấy thông tin thô
        q_info = self.questions[idx]
        q_text = q_info['question']
        q_id = q_info['question_id']
        img_id = q_info['image_id']

        # ---------------------------------------------------------
        # A. XỬ LÝ CÂU HỎI (TEXT -> TENSOR)
        # ---------------------------------------------------------
        q_indices = self.vocab_q.numericalize(q_text)
        q_tensor = torch.tensor(q_indices, dtype=torch.long)

        # ---------------------------------------------------------
        # B. XỬ LÝ CÂU TRẢ LỜI (TEXT -> TENSOR LABEL)
        # ---------------------------------------------------------
        ans_text = self.qid2ans.get(q_id, "")
        # Lấy ID của câu trả lời. Nếu từ chưa học bao giờ, gán thành <unk>
        ans_idx = self.vocab_a.word2idx.get(ans_text, self.vocab_a.word2idx.get('<unk>'))
        ans_tensor = torch.tensor(ans_idx, dtype=torch.long)

        # ---------------------------------------------------------
        # C. XỬ LÝ ẢNH (TRÍCH XUẤT TỪ H5)
        # ---------------------------------------------------------
        # Tái tạo lại tên file chuẩn của bộ COCO train2014 (đệm đủ 12 số 0)
        img_name = f"COCO_train2014_{img_id:012d}.jpg" 
        
        # Lấy index tương ứng trong file h5 (O(1) lookup)
        h5_idx = self.name2idx.get(img_name, 0)
        
        # Cắt lấy tensor ảnh (vector 2048)
        img_tensor = torch.tensor(self.h5_file['features'][h5_idx], dtype=torch.float32)

        return img_tensor, q_tensor, ans_tensor

# =====================================================================
# HÀM COLLATOR: XỬ LÝ ĐỘ DÀI CÂU LỞM CHỞM TRONG CÙNG 1 BATCH
# =====================================================================
def vqa_collate_fn(batch):
    """
    Hàm này can thiệp vào quá trình đóng gói dữ liệu của DataLoader.
    Input: Danh sách các tuple [(img1, q1, ans1), (img2, q2, ans2), ...]
    Output: img_batch, q_batch_padded, ans_batch
    """
    # Tách rời (Unzip) danh sách thành 3 cụm riêng biệt
    imgs, questions, answers = zip(*batch)
    
    # Ảnh và Câu trả lời đã có kích thước cố định, chỉ cần xếp chồng (stack) lên nhau
    # imgs shape -> (Batch, 2048) - đã mean 14x14 trong extract_features
    # answers shape -> (Batch)
    imgs_stacked = torch.stack(imgs, dim=0)
    answers_stacked = torch.stack(answers, dim=0)
    
    # Câu hỏi có độ dài khác nhau.
    # Hàm pad_sequence sẽ tự động tìm câu dài nhất trong TỪNG BATCH.
    # Các câu ngắn hơn sẽ được chèn thêm số 0 (ID của token <pad>) vào cuối.
    # questions_padded shape -> (Batch, Max_Len_Of_This_Batch)
    questions_padded = pad_sequence(questions, batch_first=True, padding_value=0)
    
    return imgs_stacked, questions_padded, answers_stacked