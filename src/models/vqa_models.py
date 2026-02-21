"""   
wrapper model 
CNN Encoder      → img_feature  (batch, hidden_size)
Question Encoder → q_feature    (batch, hidden_size)
         ↓
    FUSION = img_feature * q_feature  (Hadamard)
         ↓
    Tạo initial hidden state cho Decoder
         ↓
LSTM Decoder → logits  (batch, seq_len, vocab_size)




"""


import torch
import torch.nn as nn 
import torch.nn.functional as F 
import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.encoder_cnn import SimpleCNN, ResNetEncoder
from models.encoder_question import QuestionEncoder
from models.decoder_lstm import LSTMDecoder







# FUSION 
# Hadamard fusion 
def hadamard_fusion(img_feature, q_feature):
    return img_feature * q_feature






# Model A: no attention + scratch cnn
class VQAmodelA(nn.Module):
    def __init__(self, vocab_size, answer_vocab_size,
                 embed_size=512, hidden_size=1024, num_layers=2):
        super().__init__()

        self.num_layers = num_layers 
        
        # init 3 model 
        self.i_encoder = SimpleCNN(output_size=hidden_size)
        self.q_encoder = QuestionEncoder(vocab_size=vocab_size, embed_size=embed_size,
                                         hidden_size=hidden_size, num_layers=num_layers)
        self.decoder = LSTMDecoder(vocab_size=answer_vocab_size, embed_size=embed_size,
                                   hidden_size=hidden_size, num_layers=num_layers)


    def forward(self, images, questions, target_seq):
        """  
        images: (batch, 3, 224, 224) 
        question: (batch, max_q_len)
        target_seq: (batch, max_a_len) - answer token (teacher forcing)

        return: logits (batch, max_a_len, answer_vocab_size)
        """

        # encode image 
        img_feature = self.i_encoder(images) # (batch, hidden_size)
        img_feature = F.normalize(img_feature, p=2, dim=1) # normalize cause we don't about magnitude, we care direction 
        
        # encode question 
        q_feature = self.q_encoder(questions) # (batch, hidden_size)
        
        
        # hadamard fusion 
        fusion = hadamard_fusion(img_feature, q_feature) 

        
        # create initial hidden for decoder 
        """ 
        because decoder lstm need (h_0, c_0) tuple, each has shape (num_layers, batch, hidden_size)
        """
        h_0 = fusion.unsqueeze(0) 
        h_0 = h_0.repeat(self.num_layers, 1, 1)
        c_0 = torch.zeros_like(h_0) # convention almost paper about image captioning/ vqa 
        

        
        # decode 
        logits = self.decoder((h_0, c_0), target_seq)
        # (batch, max_a_len, answer_a_vocab_size)
        
        return logits

    
    
    
# Model B: Resnet101 + no attention 
class VQAModelB(nn.Module):
    def __init__(self, vocab_size, answer_vocab_size, 
                 embed_size=512, hidden_size=1024, num_layers=2, freeze=True):
        super().__init__()

        # init 2 encoder and 1 decoder 
        
        self.num_layers = num_layers # store for layers use in before add to decoder 
        
        self.i_encoder = ResNetEncoder(output_size=hidden_size, freeze=freeze)
        self.q_encoder = QuestionEncoder(vocab_size=vocab_size, embed_size=embed_size,
                                         hidden_size=hidden_size, num_layers=num_layers)

        self.decoder = LSTMDecoder(vocab_size=answer_vocab_size, embed_size=embed_size,
                                   hidden_size=hidden_size, num_layers=num_layers)

    
    def forward(self, images, questions, target_seq):
        # encode
        img_feature = self.i_encoder(images) # (batch, 1024)
        img_feature = F.normalize(img_feature, p=2, dim=1) # (batch, 1024)
        
        question_feature = self.q_encoder(questions) # (batch, 1024)

        fusion = hadamard_fusion(img_feature, question_feature)

        # decode 
        h_0 = fusion.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)

        logits = self.decoder((h_0, c_0), target_seq) # (batch, max_seq, answer_vocab_size)

        return logits 

        
        
        
        
       