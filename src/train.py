import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import tqdm

# Import modules
from dataset import VQADataset, vqa_collate_fn
from model import SimpleVQAModel
from vocab import Vocabulary




# Configurations Hyperparameters 

BATCH_SIZE = 64 
EPOCHS = 10 
LEARNING_RATE = 1e-3 
DEVICE = torch.device('cpu') 



def train():
    # create checkpoints for models 
    os.makedirs("checkpoints", exist_ok=True)

    
    # load data 
    vocab_q = Vocabulary()
    vocab_q.load("data/processed/vocab_questions.json")

    vocab_a = Vocabulary()
    vocab_a.load("data/processed/vocab_answers.json")

    dataset = VQADataset(
        feature_h5_path="data/processed/train_features.h5",
        question_json_path="data/raw/vqa_json/v2_OpenEnded_mscoco_train2014_questions.json",
        annotations_json_path="data/raw/vqa_json/v2_mscoco_train2014_annotations.json",
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
    model = SimpleVQAModel(
        vocab_size=len(vocab_q),
        num_classes=len(vocab_a)
    ).to(DEVICE)

    # LOSS  (CrossEntropy loss )
    criterion = nn.CrossEntropyLoss()

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

            # 1. delete grad 
            optimizer.zero_grad()

            # 2. forward 
            logits = model(imgs, questions)

            # 3. loss 
            loss = criterion(logits, answer)

            # 4. backward 
            loss.backward()

            # 5. update weight 
            optimizer.step()

            total_loss += loss.item()

            # print for tracking
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx}/{len(dataloader)}] | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

        # save model 
        torch.save(model.state_dict(), f"checkpoints/vqa_model_epoch_{epoch+1}.pth")
            

        
        
if __name__ == "__main__":
    train()
        
        
        
    
