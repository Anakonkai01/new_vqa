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

from models.encoder_cnn import SimpleCNN, ResNetEncoder, SimpleCNNSpatial, ResNetSpatialEncoder
from models.encoder_question import QuestionEncoder
from models.decoder_lstm import LSTMDecoder
from models.decoder_attention import LSTMDecoderWithAttention







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


# Model C: SimpleCNN Spatial + Bahdanau Attention + LSTMDecoder
class VQAModelC(nn.Module):
    def __init__(self, vocab_size, answer_vocab_size,
                 embed_size=512, hidden_size=1024, num_layers=2, attn_dim=512):
        super().__init__()

        self.num_layers = num_layers

        # CNN giữ spatial 7×7=49 vùng → output (batch, 49, hidden_size)
        # Khác Model A: không mean pool, decoder sẽ attend vào 49 vùng này
        self.i_encoder = SimpleCNNSpatial(output_size=hidden_size)

        # Question encoder giống hệt A/B
        self.q_encoder = QuestionEncoder(vocab_size=vocab_size, embed_size=embed_size,
                                         hidden_size=hidden_size, num_layers=num_layers)

        # Decoder có attention — nhận thêm img_features mỗi bước
        self.decoder = LSTMDecoderWithAttention(
            vocab_size=answer_vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            attn_dim=attn_dim
        )

    def forward(self, images, questions, target_seq):
        """
        images    : (batch, 3, 224, 224)
        questions : (batch, max_q_len)
        target_seq: (batch, max_a_len) — teacher forcing input

        returns: logits (batch, max_a_len, answer_vocab_size)
        """
        # ── Encode image ────────────────────────────────────────────────
        # Model A/B: (batch, hidden)   ← 1 vector duy nhất
        # Model C  : (batch, 49, hidden) ← 49 vùng, giữ spatial
        img_features = self.i_encoder(images)  # (batch, 49, hidden_size)

        # Normalize từng vùng (L2 norm theo chiều hidden)
        img_features = F.normalize(img_features, p=2, dim=-1)

        # ── Encode question ─────────────────────────────────────────────
        q_feature = self.q_encoder(questions)  # (batch, hidden_size)

        # ── Fusion để khởi tạo h_0 ──────────────────────────────────────
        # Lấy mean của 49 vùng để tạo 1 vector đại diện cho ảnh
        # Dùng để khởi tạo h_0 giống A/B — attention sẽ tinh chỉnh sau
        img_mean = img_features.mean(dim=1)    # (batch, hidden_size)
        fusion   = hadamard_fusion(img_mean, q_feature)  # (batch, hidden_size)

        h_0 = fusion.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, batch, hidden)
        c_0 = torch.zeros_like(h_0)

        # ── Decode với attention ─────────────────────────────────────────
        # Truyền thêm img_features (49 vùng) để decoder attend vào từng bước
        logits = self.decoder((h_0, c_0), img_features, target_seq)
        # (batch, max_a_len, answer_vocab_size)

        return logits


# Model D: ResNet101 Spatial (pretrained, frozen) + Bahdanau Attention + LSTMDecoder
# Giống Model C về mọi mặt — chỉ đổi SimpleCNNSpatial → ResNetSpatialEncoder
# ResNet pretrained → feature chất lượng cao hơn scratch CNN
class VQAModelD(nn.Module):
    def __init__(self, vocab_size, answer_vocab_size,
                 embed_size=512, hidden_size=1024, num_layers=2,
                 attn_dim=512, freeze_cnn=True):
        super().__init__()

        self.num_layers = num_layers

        # ResNet101 pretrained, bỏ avgpool+fc, giữ spatial (batch, 49, hidden_size)
        self.i_encoder = ResNetSpatialEncoder(output_size=hidden_size, freeze=freeze_cnn)

        # Question encoder — giống hệt A/B/C
        self.q_encoder = QuestionEncoder(vocab_size=vocab_size, embed_size=embed_size,
                                         hidden_size=hidden_size, num_layers=num_layers)

        # Decoder với attention — giống hệt Model C
        self.decoder = LSTMDecoderWithAttention(
            vocab_size=answer_vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            attn_dim=attn_dim
        )

    def forward(self, images, questions, target_seq):
        """
        images    : (batch, 3, 224, 224)
        questions : (batch, max_q_len)
        target_seq: (batch, max_a_len)
        returns   : logits (batch, max_a_len, answer_vocab_size)
        """
        # ResNet spatial: (batch, 49, hidden_size) — pretrained features
        img_features = self.i_encoder(images)
        img_features = F.normalize(img_features, p=2, dim=-1)

        q_feature = self.q_encoder(questions)  # (batch, hidden_size)

        # Mean pool 49 vùng → 1 vector để khởi tạo h_0
        img_mean = img_features.mean(dim=1)    # (batch, hidden_size)
        fusion   = hadamard_fusion(img_mean, q_feature)

        h_0 = fusion.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)

        # Decode với attention — truyền đủ 49 vùng
        logits = self.decoder((h_0, c_0), img_features, target_seq)
        return logits