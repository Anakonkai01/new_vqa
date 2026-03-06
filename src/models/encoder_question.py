import torch
import torch.nn as nn 


"""  
input: (batch, max_len)
embedding: (batch, max_len, embed_size)
LSTM: (batch, max_len, hidden_size)
output: hidden[-1] (batch, hidden_size)
"""

class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers,
                 dropout=0.5, pretrained_embeddings=None):
        super().__init__()

        # Embedding: supports GloVe pretrained vectors
        if pretrained_embeddings is not None:
            glove_dim = pretrained_embeddings.shape[1]
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=False, padding_idx=0
            )
            # If GloVe dim != embed_size, add a learned projection
            if glove_dim != embed_size:
                self.embed_proj = nn.Linear(glove_dim, embed_size)
            else:
                self.embed_proj = None
        else:
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=0
            )
            self.embed_proj = None
        
        # BiLSTM: each direction has hidden_size//2, concat → hidden_size
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, questions):
        # question (batch, max_len)
        # 1. embedding with dropout (consistent with decoder embedding dropout)
        embeds = self.dropout(self.embedding(questions))
        if self.embed_proj is not None:
            embeds = self.embed_proj(embeds)
        # embeds (batch, maxlen, embed_size)

        # 2. BiLSTM 
        output, (hidden, cell) = self.lstm(embeds)
        # output: (batch, max_len, hidden_size) — concat forward+backward
        # hidden: (num_layers*2, batch, hidden_size//2)

        # 3. Concat forward and backward last hidden states
        # forward last layer: hidden[-2], backward last layer: hidden[-1]
        q_feature = torch.cat([hidden[-2], hidden[-1]], dim=1)
        # q_feature: (batch, hidden_size)

        # 4. Return feature vector (for fusion) AND all hidden states (for question attention)
        q_hidden_states = output  # (batch, max_len, hidden_size)

        return q_feature, q_hidden_states
    

# TESTING 
if __name__ == "__main__":
    model = QuestionEncoder(vocab_size=7000, embed_size=512,
                            hidden_size=1024, num_layers=2, dropout=0.5)
    
    q = torch.randint(0, 7000, (4, 20))

    q_feat, q_hidden = model(q)

    print(q_feat.shape)    # expect (4, 1024)
    print(q_hidden.shape)  # expect (4, 20, 1024)
        