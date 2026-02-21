""" 
using autoregressive 
"""



import torch 
import torch.nn.functional as F 
import os, sys, json 
sys.path.append(os.path.dirname(__file__))


from models.vqa_models import VQAmodelA, hadamard_fusion
from vocab import Vocabulary


def greedy_decode(model, image_tensor, question_tensor, vocab_a,
                  max_len=20, device='cpu'):
    """ 
    image_tensor (3, 224, 224)
    question_tensor (max_q_len) 
    return: string answer 
    """

    with torch.no_grad():
        # add batch dim 
        img = image_tensor.unsqueeze(0).to(device) # (1, 3, 224, 224)
        question = question_tensor.unsqueeze(0).to(device) # (1, max_q_len)

        
        # encode 
        img_feat = model.i_encoder(img) # (1, 1024) 1024 is hidden size
        img_feat = F.normalize(img_feat, p=2, dim=1)
        question_feat = model.q_encoder(question) # (1, 1024)

        fusion = hadamard_fusion(img_feat, question_feat) # (1, 1024)

        # prepare h_0 and c_0 for decoder 
        h_0 = fusion.unsqueeze(0).repeat(model.num_layers, 1, 1) # (num_layers, 1, 1024)
        c_0 = torch.zeros_like(h_0)
        
        hidden = (h_0, c_0)
        
        
        # autogression decode 
        start_idx = vocab_a.word2idx['<start>']
        end_idx = vocab_a.word2idx['<end>']

        # input shape (1, 1) first is <start>
        token = torch.tensor([[start_idx]], dtype=torch.long).to(device)

        result = []

        for _ in range(max_len):
            embed = model.decoder.embedding(token) # (1, 1, embed_size)
            output, hidden = model.decoder.lstm(embed, hidden) # output (1, 1, hidden_size)
            logit = model.decoder.fc(output.squeeze(1)) # (1, vocab_a_size), we remove seq_len dim cause input of fc is (batch, hidden_size)
            pred = logit.argmax(dim=-1).item() # greedy 
            
            if pred == end_idx:
                break 
            
            result.append(pred)
            token = torch.tensor([[pred]], dtype=torch.long)

            
        words = [vocab_a.idx2word.get(i, '<unk>') for i in result]

        return ' '.join(words)
        


if __name__ == "__main__":
    from PIL import Image
    from torchvision import transforms

    DEVICE        = 'cpu'
    VOCAB_Q_PATH  = "data/processed/vocab_questions.json"
    VOCAB_A_PATH  = "data/processed/vocab_answers.json"
    CHECKPOINT    = "checkpoints/model_a_epoch10.pth"
    IMAGE_DIR     = "data/raw/images/train2014"
    QUESTION_JSON = "data/raw/vqa_json/v2_OpenEnded_mscoco_train2014_questions.json"

    # Load vocab
    vocab_q = Vocabulary(); vocab_q.load(VOCAB_Q_PATH)
    vocab_a = Vocabulary(); vocab_a.load(VOCAB_A_PATH)

    # Load model
    model = VQAmodelA(vocab_size=len(vocab_q), answer_vocab_size=len(vocab_a))
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))

    # Load 1 sample
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    with open(QUESTION_JSON, 'r') as f:
        questions = json.load(f)['questions']

    sample = questions[0]
    q_text = sample['question']
    img_id = sample['image_id']

    img_path   = os.path.join(IMAGE_DIR, f"COCO_train2014_{img_id:012d}.jpg")
    img_tensor = transform(Image.open(img_path).convert("RGB"))
    q_tensor   = torch.tensor(vocab_q.numericalize(q_text), dtype=torch.long)

    answer = greedy_decode(model, img_tensor, q_tensor, vocab_a, device=DEVICE)
    print(f"Question : {q_text}")
    print(f"Predicted: {answer}")