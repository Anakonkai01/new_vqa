import torch
import torch.nn as nn 


"""  
input: (batch, max_len)
embedding: (batch, max_len, embed_size)
LSTM: (batch, max_len, hidden_size)
output: hidden[-1] (batch, hidden_size)
"""

class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.5):
        super().__init__()

        # embedding: convert token indices to vector embedding 
        # ex: token 42 -> vector [0.1, -0.3, ..., embedding_size]
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=0)
        
        # lstm 
        self.lstm = nn.LSTM(
            input_size=embed_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
    def forward(self, questions):
        # question (batch, max_len) 
        # 1. embeding 
        embeds = self.embedding(questions)
        # embes (batch, maxlen, embed_size)

        # 2. lstm 
        output, (hidden, cell) = self.lstm(embeds)
        # output: all hidden state of last layers (batch, max_len, hidden_size)
        # hidden: last hidden state of each layers (num_layers, batch, hidden_size)
        # cell: last cell state each layers (num_layers, batch, hidden_size)

        # 3. take hidden state of last layer 
        q_feature = hidden[-1]
        # (batch, hidden_size)

        return q_feature
    

# TESTING 
if __name__ == "__main__":
    model = QuestionEncoder(vocab_size=7000, embed_size=512,
                            hidden_size=1024, num_layers=2, dropout=0.5)
    
    q = torch.randint(0, 7000, (4, 20))

    out = model(q)

    print(out.shape) # expect (4, 1024)
        