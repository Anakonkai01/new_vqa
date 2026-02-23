"""  
dataset class do: 
input: idx 
output: 1 sample (image_tensor, question_tensor, answer_tensor)

Model A: cnn scratch no attention -> load raw images 
"""



import torch
from  torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from torchvision import transforms
import json
import os 



# helper function 
# collate function 
def vqa_collate_fn(batch):
    """  
    haddle batch that has len of each sequence different with eachother 
    pad_sequence auto find the longest sequence in batch  and pad zeros for short sequence  
    """

    imgs, questions, answers = zip(*batch) # unzip batch 

    # imgs does not need to process so stack (according to batch) it normally 
    imgs_stacked = torch.stack(imgs, dim=0)
    # pad sequence question and answer in batch 
    questions_padded = pad_sequence(questions, batch_first=True)
    answer_padded = pad_sequence(answers, batch_first=True)

    return imgs_stacked, questions_padded, answer_padded
    



class VQADatasetA(Dataset):
    def __init__(self, image_dir, question_json_path,
                 annotations_json_path, vocab_q, vocab_a,
                 split='train2014', max_samples=None):
        """
        split      : 'train2014' hoặc 'val2014' — dùng để tạo đúng tên file ảnh
        max_samples: giới hạn số sample (tiện khi test nhanh trên Kaggle)
        """
        self.image_dir = image_dir
        self.vocab_q   = vocab_q
        self.vocab_a   = vocab_a
        self.split     = split        # 'train2014' | 'val2014'

        # transform: resize to 224, normalize according imagenet 
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # load questions
        with open(question_json_path, 'r') as f:
            self.questions = json.load(f)['questions']

        # giới hạn số sample nếu có (hữu ích khi test pipeline nhanh)
        if max_samples is not None:
            self.questions = self.questions[:max_samples]

        # build dict: question_id -> answer_text
        with open(annotations_json_path, 'r') as f:
            annotations = json.load(f)['annotations']

        self.qid2ans = {ann['question_id']: ann['multiple_choice_answer'] for ann in annotations}
        
    
    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        q_info = self.questions[index]
        q_text = q_info['question']
        q_id = q_info['question_id']
        img_id = q_info['image_id']

        # 1. process question text to indices tensor 
        q_indices = self.vocab_q.numericalize(q_text)
        q_tensor = torch.tensor(q_indices, dtype=torch.long) # int 

        # 2. process annotations to indices tensor
        a_text = self.qid2ans.get(q_id, "") # "" mean if does not found the key, return default ""
        a_indices = self.vocab_a.numericalize(a_text)
        a_tensor = torch.tensor(a_indices, dtype=torch.long)

        # 3. process image
        # tên file phụ thuộc split: COCO_train2014_... hoặc COCO_val2014_...
        img_name = f"COCO_{self.split}_{img_id:012d}.jpg"
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(image)

        return img_tensor, q_tensor, a_tensor
    
    
