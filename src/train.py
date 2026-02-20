import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys 
import tqdm
sys.path.append(os.path.dirname(__file__))

# Import modules
from dataset import VQADatasetA, vqa_collate_fn
from models.vqa_models import VQAmodelA
from vocab import Vocabulary




""" 
because CE only work with 2d prediction: (N, C), target (N) -> sample, class 
so we need to convert from 3d to 2d by concat the second dim to first dim


# logits:  (batch, seq_len, vocab_size)
# targets: (batch, seq_len)
# CrossEntropyLoss cần: (batch*seq_len, vocab_size) và (batch*seq_len)

logits  = logits.view(-1, vocab_size)   # (batch*seq_len, vocab_size)
targets = targets.view(-1)              # (batch*seq_len)
loss = criterion(logits, targets)
"""



# Configurations Hyperparameters 

BATCH_SIZE = 32
EPOCHS = 10 
LEARNING_RATE = 1e-3 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')


IMAGE_DIR        = "data/raw/images/train2014"
QUESTION_JSON    = "data/raw/vqa_json/v2_OpenEnded_mscoco_train2014_questions.json"
ANNOTATION_JSON  = "data/raw/vqa_json/v2_mscoco_train2014_annotations.json"
VOCAB_Q_PATH     = "data/processed/vocab_questions.json"
VOCAB_A_PATH     = "data/processed/vocab_answers.json"


def train():
    # create checkpoints for models 
    os.makedirs("checkpoints", exist_ok=True)

    
    # load vocab
    vocab_q = Vocabulary()
    vocab_q.load(VOCAB_Q_PATH)
    vocab_a = Vocabulary()
    vocab_a.load(VOCAB_A_PATH)

    # Dataset and Dataloader
    dataset = VQADatasetA(
        image_dir=IMAGE_DIR, 
        question_json_path=QUESTION_JSON,
        annotations_json_path=ANNOTATION_JSON,
        vocab_q=vocab_q,
        vocab_a=vocab_a    
    )
    
    # dataloader (conveyor)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, 
        collate_fn=vqa_collate_fn, # function haddle padding
        num_workers=4,
        pin_memory =  True if torch.cuda.is_available() else False
    )

    
    # init model, optimize function, loss function 
    
    # MODEL
    model = VQAmodelA(
        vocab_size=len(vocab_q),
        answer_vocab_size=len(vocab_a)
    ).to(DEVICE)

    # ignore idx = 0, do not compute loss on <PAD> token
    # LOSS  (CrossEntropy loss )
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # OPTIMIZER (Adam)
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    # training loop 
    print("Training on",DEVICE)
    for epoch in tqdm.tqdm(range(EPOCHS)):
        # switch model to train mode 
        model.train() 
        total_loss = 0 # for tracking 
        
        for batch_idx, (imgs, questions, answer) in enumerate(dataloader):
            # move data to DEVICE 
            imgs = imgs.to(DEVICE)
            questions = questions.to(DEVICE)
            answer = answer.to(DEVICE)

            # TEACHER FORCING (one step ahead)
            # input decoder [<start>, w1, w2, ..., wn]
            # target (label) [w1, w2, ..., <end>]
            decoder_input = answer[:, :-1]
            decoder_target = answer[:, 1:]

            # 1. delete grad 
            optimizer.zero_grad()

            # 2. forward 
            logits = model(imgs, questions, decoder_input)
            # (batch, seq_len, vocab_size)

            # 3. loss 
            # reshape for CE
            vocab_size = logits.size(-1) 
            loss = criterion(
                logits.view(-1, vocab_size), # (batch * seq_len, vocab_size)
                decoder_target.contiguous().view(-1) # (batch*seq_len), because we slice so we need contiguous
            )

            # 4. backward 
            loss.backward()

            # Gradient clipping to avoid exploding gradient 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            # 5. update weight 
            optimizer.step()

            total_loss += loss.item()

            # print for tracking
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] | "
                    f"Batch [{batch_idx}/{len(dataloader)}] | "
                    f"Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

        # save model 
        torch.save(model.state_dict(), f"checkpoints/model_a_epoch{epoch+1}.pth")
            

        
        
if __name__ == "__main__":
    train()
        
        
        
    
