import torch
import torch.nn as nn 
import torch.nn.functional as F 


class SimpleVQAModel(nn.Module):
    def __init__(self, vocab_size, num_classes, 
                 embed_size = 512, hidden_size = 1024 , num_layers = 2):
        
        super().__init__()

        
        # IMAGE ENCODER
        # image encoder (ResNet101)
        # input from ResNet (batch, 7, 7, 2048)

        self.visual_linear = nn.Sequential(
            nn.Linear(2048, hidden_size),
            nn.BatchNorm1d(hidden_size),  # learn more stable
            nn.ReLU(),
            nn.Dropout(0.5) # avoid overfitting
        )

        # QUESTION ENCODER
        # input idx of word, take from vocab dict 
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # lstm 
        self.lstm = nn.LSTM(
            input_size = embed_size,
            hidden_size = hidden_size, 
            num_layers = num_layers, 
            batch_first = True, 
            dropout=0.5 if num_layers > 1 else 0
        )

        # CLASSIFIER 
        # input: vector (image * question)
        # output: prob of each class (num_class)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # combine with ReLU to learn non-linearity 
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(hidden_size, num_classes) 
        )
        
    def forward(self, images, questions):
        """  
        images (batch, 7, 7, 2048)
        question (batch, max_len)
        """

        
        # process image feature
        # take mean of 2048 features vector 
        # squeeze the spacial dimension from 14x14 or 7x7 to 1 value (mean)
        # img_feature = images.mean(dim=[1,2])  # -> (batch_size, vector feature)
       
       
        # normalize the vector feature because we only care the direction (which contain the context), we don't care the magnitude of the vector 
        # using l2 norm (which divide the vector with l2 norm of vector) 
        # after that, all the vector now have l2 norm = 1 but still contain the direction
        img_feature = F.normalize(img_feature, p=2, dim=1)

        # mapping from vector feature space to hidden space
        img_feature = self.visual_linear(img_feature)

        
        # HANDLE QUESTION
        embeds = self.embedding(questions) # -> (batch, seq, embed_size)

        # forward the question through LSTM 
        _, (hidden, cell) = self.lstm(embeds) # take the final state(last hidden)
        # hidden (num_layers, batch, hidden_size)

        # feature extraction of top layer of LSTM 
        q_feature = hidden[-1] # (batch, hidden_size)

        # fusion (using hadamard product)
        fusion = img_feature * q_feature # fusion (batch_size, hidden_size)

        # prediction 
        logits = self.classifier(fusion)

        return logits

        
        
        
        
# === KHỐI KIỂM TRA NHANH (DEBUG LOCAL) ===
if __name__ == "__main__":
    print("🛠️ Đang kiểm tra SimpleVQAModel...")
    
    # Giả lập siêu tham số
    BATCH = 4
    VOCAB_SIZE = 1000
    NUM_CLASSES = 3129 # Số lượng câu trả lời thường thấy trong VQA 2.0
    
    model = SimpleVQAModel(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES)
    
    # Giả lập dữ liệu (Dựa theo extract_features.py của bạn đang xuất 14x14)
    dummy_img = torch.randn(BATCH, 14, 14, 2048) 
    dummy_ques = torch.randint(0, VOCAB_SIZE, (BATCH, 15)) # Câu hỏi dài 15 từ
    
    # Forward Pass
    output = model(dummy_img, dummy_ques)
    
    print("✅ Mạng Nơ-ron đã thông suốt!")
    print(f" - Image Input shape: {dummy_img.shape}")
    print(f" - Question Input shape: {dummy_ques.shape}")
    print(f" - Logits Output shape: {output.shape} (Kỳ vọng: [{BATCH}, {NUM_CLASSES}])")


        


