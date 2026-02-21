"""
decoder lstm work 
1. inital_hidden =  fusion(img, question), 
    input = <start> token
2. LSTM (<start>, hidden) -> predict "yes"
3. LSTM ("yes", hidden) -> predict <end>

answer = "yes"



TRAINING PHASE: Teacher Forcing 
INFERENCE PHASE: Autoregressive


flow : 
- initial_hidden (num_layers, batch, hidden_size) from fusion 
- input token: (batch, 1): 1 token each step 
- embedding: (batch, 1, embeded_size) 
- lstm output: (batch, 1, hidden_size)
- fc (linear) (batch, 1, vocab_size): logits for each token
"""


import torch
import torch.nn as nn


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.5):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embed_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )

        # projec hidden state -> vocab logits 
        self.fc = nn.Linear(hidden_size, vocab_size)

    
    def forward(self, encoder_hidden, target_seq):
        """  
        Training mode: teacher forcing 
        
        encoder_hidden (num_layers, batch, hidden_size) from fusion 
        target_seq: (batch, max_len): answer token [<start>, w1, w2, ..., <end>]
        
        returns: 
        logits (batch, max_len, vocab_size)
        """

        embeds = self.embedding(target_seq)
        # embeds (batch, maxlen, embed_size)

        # use encoder_hidden to be initial state of lstm 
        outputs, (hidden, cell) = self.lstm(embeds, encoder_hidden)
        # outputs (batch, maxlen, hidden_size)

        # project from hidden to vocab space 
        logits = self.fc(outputs)
        # logits (batch, maxlen, vocab_size)

        return logits 
    
    
    
# TESTING 
if __name__ == "__main__":
    decoder = LSTMDecoder(vocab_size=3000, embed_size=512, 
                          hidden_size=1024, num_layers=2)

    # simulate encoder hidden (tuple, h, c)
    h = torch.zeros(2, 4, 1024)
    c = torch.zeros(2, 4, 1024)

    target = torch.randint(0, 3000, (4,10))

    logits = decoder((h,c), target)

    print(logits.shape) # expect (4, 10, 3000)
    




