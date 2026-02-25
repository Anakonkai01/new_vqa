""" 
using autoregressive 
"""



import torch 
import torch.nn.functional as F 
import os, sys, json 
sys.path.append(os.path.dirname(__file__))


from models.vqa_models import VQAModelA, VQAModelB, VQAModelC, VQAModelD, hadamard_fusion
from vocab import Vocabulary


def get_model(model_type, vocab_q_size, vocab_a_size):
    if model_type == 'A':
        return VQAModelA(vocab_size=vocab_q_size, answer_vocab_size=vocab_a_size)
    elif model_type == 'B':
        return VQAModelB(vocab_size=vocab_q_size, answer_vocab_size=vocab_a_size)
    elif model_type == 'C':
        return VQAModelC(vocab_size=vocab_q_size, answer_vocab_size=vocab_a_size)
    elif model_type == 'D':
        return VQAModelD(vocab_size=vocab_q_size, answer_vocab_size=vocab_a_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from A, B, C, D.")


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


def greedy_decode_with_attention(model, image_tensor, question_tensor, vocab_a,
                                 max_len=20, device='cpu'):
    """
    For Model C and D (with Bahdanau attention).
    image_tensor : (3, 224, 224)
    question_tensor: (max_q_len)
    return: string answer
    """
    with torch.no_grad():
        img      = image_tensor.unsqueeze(0).to(device)       # (1, 3, 224, 224)
        question = question_tensor.unsqueeze(0).to(device)    # (1, max_q_len)

        # encode
        img_features  = model.i_encoder(img)                  # (1, 49, 1024) -- keeps spatial
        img_features  = F.normalize(img_features, p=2, dim=-1)
        question_feat = model.q_encoder(question)             # (1, 1024)

        # build image representation as mean of spatial regions
        img_mean = img_features.mean(dim=1)                   # (1, 1024)

        fusion = hadamard_fusion(img_mean, question_feat)     # (1, 1024)

        # initialize decoder hidden state
        h_0 = fusion.unsqueeze(0).repeat(model.num_layers, 1, 1)  # (num_layers, 1, 1024)
        c_0 = torch.zeros_like(h_0)
        hidden = (h_0, c_0)

        start_idx = vocab_a.word2idx['<start>']
        end_idx   = vocab_a.word2idx['<end>']

        token  = torch.tensor([[start_idx]], dtype=torch.long).to(device)  # (1, 1)
        result = []

        for _ in range(max_len):
            # decode_step returns (logit, new_hidden, alpha)
            # img_features passed at each step so attention can compute context
            logit, hidden, alpha = model.decoder.decode_step(token, hidden, img_features)
            pred = logit.argmax(dim=-1).item()

            if pred == end_idx:
                break

            result.append(pred)
            token = torch.tensor([[pred]], dtype=torch.long).to(device)

        words = [vocab_a.idx2word.get(i, '<unk>') for i in result]
        return ' '.join(words)


def batch_greedy_decode(model, img_tensors, q_tensors, vocab_a,
                        max_len=20, device='cpu'):
    """
    Batch greedy decode for models A/B (no attention).

    img_tensors : (B, 3, 224, 224)
    q_tensors   : (B, max_q_len)
    returns     : list of B answer strings
    """
    with torch.no_grad():
        B    = img_tensors.size(0)
        imgs = img_tensors.to(device)
        qs   = q_tensors.to(device)

        img_feat = model.i_encoder(imgs)               # (B, hidden)
        img_feat = F.normalize(img_feat, p=2, dim=1)
        q_feat   = model.q_encoder(qs)                 # (B, hidden)
        fusion   = hadamard_fusion(img_feat, q_feat)

        h = fusion.unsqueeze(0).repeat(model.num_layers, 1, 1)  # (layers, B, hidden)
        c = torch.zeros_like(h)
        hidden = (h, c)

        start_idx = vocab_a.word2idx['<start>']
        end_idx   = vocab_a.word2idx['<end>']

        token    = torch.full((B, 1), start_idx, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        results  = [[] for _ in range(B)]

        for _ in range(max_len):
            embed  = model.decoder.embedding(token)             # (B, 1, embed)
            output, hidden = model.decoder.lstm(embed, hidden)  # (B, 1, hidden)
            logit  = model.decoder.fc(output.squeeze(1))        # (B, vocab)
            preds  = logit.argmax(dim=-1)                       # (B,)

            for i in range(B):
                if not finished[i]:
                    p = preds[i].item()
                    if p == end_idx:
                        finished[i] = True
                    else:
                        results[i].append(p)

            if finished.all():
                break

            token = preds.unsqueeze(1)  # (B, 1)

        return [' '.join(vocab_a.idx2word.get(i, '<unk>') for i in r) for r in results]


def batch_greedy_decode_with_attention(model, img_tensors, q_tensors, vocab_a,
                                       max_len=20, device='cpu'):
    """
    Batch greedy decode for models C/D (Bahdanau attention).

    img_tensors : (B, 3, 224, 224)
    q_tensors   : (B, max_q_len)
    returns     : list of B answer strings
    """
    with torch.no_grad():
        B    = img_tensors.size(0)
        imgs = img_tensors.to(device)
        qs   = q_tensors.to(device)

        img_features = model.i_encoder(imgs)                    # (B, 49, hidden)
        img_features = F.normalize(img_features, p=2, dim=-1)
        q_feat       = model.q_encoder(qs)                      # (B, hidden)
        img_mean     = img_features.mean(dim=1)                 # (B, hidden)
        fusion       = hadamard_fusion(img_mean, q_feat)

        h = fusion.unsqueeze(0).repeat(model.num_layers, 1, 1)
        c = torch.zeros_like(h)
        hidden = (h, c)

        start_idx = vocab_a.word2idx['<start>']
        end_idx   = vocab_a.word2idx['<end>']

        token    = torch.full((B, 1), start_idx, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        results  = [[] for _ in range(B)]

        for _ in range(max_len):
            logit, hidden, _ = model.decoder.decode_step(token, hidden, img_features)
            preds = logit.argmax(dim=-1)  # (B,)

            for i in range(B):
                if not finished[i]:
                    p = preds[i].item()
                    if p == end_idx:
                        finished[i] = True
                    else:
                        results[i].append(p)

            if finished.all():
                break

            token = preds.unsqueeze(1)  # (B, 1)

        return [' '.join(vocab_a.idx2word.get(i, '<unk>') for i in r) for r in results]


if __name__ == "__main__":
    from PIL import Image
    from torchvision import transforms

    MODEL_TYPE    = 'A'   # change to 'B', 'C', 'D' to run other models
    DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    VOCAB_Q_PATH  = "data/processed/vocab_questions.json"
    VOCAB_A_PATH  = "data/processed/vocab_answers.json"
    CHECKPOINT    = f"checkpoints/model_{MODEL_TYPE.lower()}_epoch10.pth"
    IMAGE_DIR     = "data/raw/images/train2014"
    QUESTION_JSON = "data/raw/vqa_json/v2_OpenEnded_mscoco_train2014_questions.json"

    # Load vocab
    vocab_q = Vocabulary(); vocab_q.load(VOCAB_Q_PATH)
    vocab_a = Vocabulary(); vocab_a.load(VOCAB_A_PATH)

    # Load model
    model = get_model(MODEL_TYPE, len(vocab_q), len(vocab_a))
    model.load_state_dict(torch.load(CHECKPOINT, map_location=lambda storage, loc: storage))
    model.to(DEVICE)

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

    # select decode function based on model type
    if MODEL_TYPE in ('A', 'B'):
        answer = greedy_decode(model, img_tensor, q_tensor, vocab_a, device=DEVICE)
    else:
        answer = greedy_decode_with_attention(model, img_tensor, q_tensor, vocab_a, device=DEVICE)

    print(f"Model    : {MODEL_TYPE}")
    print(f"Question : {q_text}")
    print(f"Predicted: {answer}")