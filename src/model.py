import torch
import torch.nn as nn 
import torch.functional as F 


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
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU()
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
        img_feature = images.mean(dim=[1,2])  # -> (batch_size, vector feature)
       
       
        # normalize the vector feature because we only care the direction (which contain the context), we don't care the magnitude of the vector 
        # using l2 norm (which divide the vector with l2 norm of vector) 
        # after that, all the vector now have l2 norm = 1 but still contain the direction
        img_feature = F.normalize(img_feature, p=2, dim=1)

        # mapping from vector feature space to hidden space
        img_feature = self.visual_linear(img_feature)

        
        # HANDLE QUESTION
        embeds = self.embedding(questions) # -> (batch, seq, embed_size)


        