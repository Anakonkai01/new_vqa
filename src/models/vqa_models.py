"""
Wrapper model:
  CNN Encoder      -> img_feature  (batch, hidden_size)
  Question Encoder -> q_feature    (batch, hidden_size)
           |
      FUSION = img_feature * q_feature  (Hadamard element-wise product)
           |
      Create initial hidden state for Decoder
           |
  LSTM Decoder -> logits  (batch, seq_len, vocab_size)
"""


import torch
import torch.nn as nn 
import torch.nn.functional as F 
import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.encoder_cnn import SimpleCNN, ResNetEncoder, SimpleCNNSpatial, ResNetSpatialEncoder, CLIPViTEncoder
from models.encoder_question import QuestionEncoder
from models.decoder_lstm import LSTMDecoder
from models.decoder_attention import LSTMDecoderWithAttention







# FUSION 
# Hadamard fusion (legacy — kept for reference)
def hadamard_fusion(img_feature, q_feature):
    return img_feature * q_feature


# Gated Fusion — learnable gate decides how much image vs question info to keep
class GatedFusion(nn.Module):
    """Gated multimodal fusion.
    gate = σ(W_g · [img; q])
    output = gate * tanh(W_img(img)) + (1 - gate) * tanh(W_q(q))
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.fc_img  = nn.Linear(hidden_size, hidden_size)
        self.fc_q    = nn.Linear(hidden_size, hidden_size)
        self.fc_gate = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, img_feature, q_feature):
        h_img = torch.tanh(self.fc_img(img_feature))   # (batch, hidden)
        h_q   = torch.tanh(self.fc_q(q_feature))       # (batch, hidden)
        gate  = torch.sigmoid(self.fc_gate(torch.cat([img_feature, q_feature], dim=1)))  # (batch, hidden)
        return gate * h_img + (1 - gate) * h_q          # (batch, hidden)

# FiLM Fusion — Feature-wise Linear Modulation
class FiLMFusion(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Fusion.
    The question feature generates gamma (scale) and beta (shift) vectors.
    These are applied to the image features element-wise.
    """
    def __init__(self, hidden_size):
        super().__init__()
        # MLP to generate gamma and beta from question feature
        # Using a bottleneck to reduce parameters 
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2), # Dropout added as per Master Blueprint
            nn.Linear(hidden_size // 2, hidden_size * 2) # Outputs [gamma, beta]
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, img_features, q_feature):
        """
        img_features: (batch, 49, hidden) or (batch, hidden)
        q_feature   : (batch, hidden)
        """
        film_params = self.mlp(q_feature) # (batch, hidden * 2)
        gamma, beta = torch.chunk(film_params, 2, dim=-1) # Each is (batch, hidden)
        
        # If img_features is spatial (batch, 49, hidden), reshape gamma/beta to broadcast
        if img_features.dim() == 3:
            gamma = gamma.unsqueeze(1) # (batch, 1, hidden)
            beta = beta.unsqueeze(1)   # (batch, 1, hidden)
            
        # Normalize img_features before modulation to keep activations stable
        img_features = self.layer_norm(img_features)
        
        # Modulate
        modulated_img = gamma * img_features + beta
        return modulated_img






# Model A: no attention + scratch CNN
class VQAModelA(nn.Module):
    def __init__(self, vocab_size, answer_vocab_size,
                 embed_size=512, hidden_size=1024, num_layers=2, dropout=0.5,
                 pretrained_q_emb=None, pretrained_a_emb=None):
        super().__init__()

        self.num_layers = num_layers

        # Initialize image encoder, question encoder, and decoder
        self.i_encoder = SimpleCNN(output_size=hidden_size)
        self.q_encoder = QuestionEncoder(vocab_size=vocab_size, embed_size=embed_size,
                                         hidden_size=hidden_size, num_layers=num_layers,
                                         dropout=dropout,
                                         pretrained_embeddings=pretrained_q_emb)
        self.fusion = GatedFusion(hidden_size)
        self.decoder = LSTMDecoder(vocab_size=answer_vocab_size, embed_size=embed_size,
                                   hidden_size=hidden_size, num_layers=num_layers,
                                   dropout=dropout,
                                   pretrained_embeddings=pretrained_a_emb)


    def forward(self, images, questions, target_seq):
        """  
        images: (batch, 3, 224, 224) 
        question: (batch, max_q_len)
        target_seq: (batch, max_a_len) - answer token (teacher forcing)

        return: logits (batch, max_a_len, answer_vocab_size)
        """

        # encode image
        img_feature = self.i_encoder(images)  # (batch, hidden_size)
        img_feature = F.normalize(img_feature, p=2, dim=1)  # L2 normalize: direction matters, not magnitude

        # encode question (BiLSTM returns tuple)
        q_feature, _ = self.q_encoder(questions)  # q_feature: (batch, hidden_size)

        # Gated fusion
        fusion = self.fusion(img_feature, q_feature)

        # Create initial hidden state for decoder.
        # LSTM expects (h_0, c_0) tuple, each shape (num_layers, batch, hidden_size)
        h_0 = fusion.unsqueeze(0)
        h_0 = h_0.repeat(self.num_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)  # zero-init cell state (standard practice)

        # decode
        logits = self.decoder((h_0, c_0), target_seq)
        # (batch, max_a_len, answer_a_vocab_size)
        
        return logits

    
    
    
# Model B: ResNet101 + no attention
class VQAModelB(nn.Module):
    def __init__(self, vocab_size, answer_vocab_size,
                 embed_size=512, hidden_size=1024, num_layers=2, freeze=True, dropout=0.5,
                 pretrained_q_emb=None, pretrained_a_emb=None):
        super().__init__()

        self.num_layers = num_layers  # stored for use when building initial hidden state

        self.i_encoder = ResNetEncoder(output_size=hidden_size, freeze=freeze)
        self.q_encoder = QuestionEncoder(vocab_size=vocab_size, embed_size=embed_size,
                                         hidden_size=hidden_size, num_layers=num_layers,
                                         dropout=dropout,
                                         pretrained_embeddings=pretrained_q_emb)
        self.fusion = GatedFusion(hidden_size)
        self.decoder = LSTMDecoder(vocab_size=answer_vocab_size, embed_size=embed_size,
                                   hidden_size=hidden_size, num_layers=num_layers,
                                   dropout=dropout,
                                   pretrained_embeddings=pretrained_a_emb)

    
    def forward(self, images, questions, target_seq):
        # encode
        img_feature = self.i_encoder(images) # (batch, 1024)
        img_feature = F.normalize(img_feature, p=2, dim=1) # (batch, 1024)
        
        question_feature, _ = self.q_encoder(questions) # (batch, 1024)

        fusion = self.fusion(img_feature, question_feature)

        # decode 
        h_0 = fusion.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)

        logits = self.decoder((h_0, c_0), target_seq) # (batch, max_seq, answer_vocab_size)

        return logits


# Model C: SimpleCNN Spatial + Bahdanau Attention + LSTM Decoder
class VQAModelC(nn.Module):
    def __init__(self, vocab_size, answer_vocab_size,
                 embed_size=512, hidden_size=1024, num_layers=2, attn_dim=512, dropout=0.5,
                 pretrained_q_emb=None, pretrained_a_emb=None,
                 use_coverage=False):
        super().__init__()

        self.num_layers = num_layers

        # CNN preserves spatial 7x7=49 regions -> output (batch, 49, hidden_size)
        # Unlike Model A: no mean pool; decoder attends over all 49 regions
        self.i_encoder = SimpleCNNSpatial(output_size=hidden_size)

        # Question encoder — identical to A/B
        self.q_encoder = QuestionEncoder(vocab_size=vocab_size, embed_size=embed_size,
                                         hidden_size=hidden_size, num_layers=num_layers,
                                         dropout=dropout,
                                         pretrained_embeddings=pretrained_q_emb)
        self.fusion = GatedFusion(hidden_size)

        # Decoder with dual attention — receives img_features + q_hidden at every decode step
        self.decoder = LSTMDecoderWithAttention(
            vocab_size=answer_vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            attn_dim=attn_dim,
            dropout=dropout,
            pretrained_embeddings=pretrained_a_emb,
            use_coverage=use_coverage
        )

    def forward(self, images, questions, target_seq):
        """
        images    : (batch, 3, 224, 224)
        questions : (batch, max_q_len)
        target_seq: (batch, max_a_len) — teacher forcing input

        returns:
          logits        : (batch, max_a_len, answer_vocab_size)
          coverage_loss : scalar tensor (0.0 if coverage disabled)
        """
        # ── Encode image ──────────────────────────────────────────
        # Model A/B: (batch, hidden)      <- single global vector
        # Model C  : (batch, 49, hidden)  <- 49 spatial regions
        img_features = self.i_encoder(images)  # (batch, 49, hidden_size)

        # L2 normalize each region independently
        img_features = F.normalize(img_features, p=2, dim=-1)

        # ── Encode question ────────────────────────────────────────
        q_feature, q_hidden_states = self.q_encoder(questions)
        # q_feature: (batch, hidden_size)  q_hidden_states: (batch, q_len, hidden_size)

        # ── Build h_0 via gated fusion ─────────────────────────────
        img_mean = img_features.mean(dim=1)    # (batch, hidden_size)
        fusion   = self.fusion(img_mean, q_feature)  # (batch, hidden_size)

        h_0 = fusion.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, batch, hidden)
        c_0 = torch.zeros_like(h_0)

        # ── Decode with dual attention ─────────────────────────────
        # Pass img_features (49 regions) + q_hidden_states for dual attention
        logits, coverage_loss = self.decoder((h_0, c_0), img_features, q_hidden_states, target_seq)
        # (batch, max_a_len, answer_vocab_size)

        return logits, coverage_loss


# Model D: ResNet101 Spatial (pretrained, frozen) + Bahdanau Attention + LSTM Decoder
# Same architecture as Model C — only difference: SimpleCNNSpatial -> ResNetSpatialEncoder
# Pretrained ResNet produces higher-quality features than a scratch CNN
class VQAModelD(nn.Module):
    def __init__(self, vocab_size, answer_vocab_size,
                 embed_size=512, hidden_size=1024, num_layers=2,
                 attn_dim=512, freeze_cnn=True, dropout=0.5,
                 pretrained_q_emb=None, pretrained_a_emb=None,
                 use_coverage=False):
        super().__init__()

        self.num_layers = num_layers

        # ResNet101 pretrained, avgpool+fc removed, keeps spatial (batch, 49, hidden_size)
        self.i_encoder = ResNetSpatialEncoder(output_size=hidden_size, freeze=freeze_cnn)

        # Question encoder — identical to A/B/C
        self.q_encoder = QuestionEncoder(vocab_size=vocab_size, embed_size=embed_size,
                                         hidden_size=hidden_size, num_layers=num_layers,
                                         dropout=dropout,
                                         pretrained_embeddings=pretrained_q_emb)
        self.fusion = GatedFusion(hidden_size)

        # Decoder with dual attention — identical to Model C
        self.decoder = LSTMDecoderWithAttention(
            vocab_size=answer_vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            attn_dim=attn_dim,
            dropout=dropout,
            pretrained_embeddings=pretrained_a_emb,
            use_coverage=use_coverage
        )

    def forward(self, images, questions, target_seq):
        """
        images    : (batch, 3, 224, 224)
        questions : (batch, max_q_len)
        target_seq: (batch, max_a_len)
        returns   :
          logits        : (batch, max_a_len, answer_vocab_size)
          coverage_loss : scalar tensor (0.0 if coverage disabled)
        """
        # ResNet spatial: (batch, 49, hidden_size) — high-quality pretrained features
        img_features = self.i_encoder(images)
        img_features = F.normalize(img_features, p=2, dim=-1)

        q_feature, q_hidden_states = self.q_encoder(questions)
        # q_feature: (batch, hidden_size)  q_hidden_states: (batch, q_len, hidden_size)

        # Mean-pool 49 regions -> 1 vector for gated fusion
        img_mean = img_features.mean(dim=1)    # (batch, hidden_size)
        fusion   = self.fusion(img_mean, q_feature)

        h_0 = fusion.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)

        # Decode with dual attention — pass img_features + q_hidden_states
        logits, coverage_loss = self.decoder((h_0, c_0), img_features, q_hidden_states, target_seq)
        return logits, coverage_loss

# Model E: CLIP ViT-B/32 Spatial + FiLM Fusion + Bahdanau Attention + Init Projection
class VQAModelE(nn.Module):
    def __init__(self, vocab_size, answer_vocab_size,
                 embed_size=512, hidden_size=1024, num_layers=2,
                 attn_dim=512, freeze_cnn=True, dropout=0.5,
                 pretrained_q_emb=None, pretrained_a_emb=None,
                 use_coverage=False):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # CLIP ViT-B/32 Spatial Encoder: returns (batch, 49, hidden_size)
        self.i_encoder = CLIPViTEncoder(output_size=hidden_size, freeze=freeze_cnn)

        # Question encoder
        self.q_encoder = QuestionEncoder(vocab_size=vocab_size, embed_size=embed_size,
                                         hidden_size=hidden_size, num_layers=num_layers,
                                         dropout=dropout,
                                         pretrained_embeddings=pretrained_q_emb)
        
        # Question feature normalization
        self.q_norm = nn.LayerNorm(hidden_size)

        # FiLM Fusion
        self.fusion = FiLMFusion(hidden_size)
        
        # Init State Projection (Non-linear projection to generate h_0 and c_0)
        self.init_h_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        self.init_c_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

        # Decoder with dual attention
        self.decoder = LSTMDecoderWithAttention(
            vocab_size=answer_vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            attn_dim=attn_dim,
            dropout=dropout,
            pretrained_embeddings=pretrained_a_emb,
            use_coverage=use_coverage
        )

    def forward(self, images, questions, target_seq):
        """
        images    : (batch, 3, 224, 224)
        questions : (batch, max_q_len)
        target_seq: (batch, max_a_len)
        returns   :
          logits        : (batch, max_a_len, answer_vocab_size)
          coverage_loss : scalar tensor (0.0 if coverage disabled)
        """
        # CLIP spatial: (batch, 49, hidden_size)
        img_features = self.i_encoder(images)
        img_features = F.normalize(img_features, p=2, dim=-1)

        q_feature, q_hidden_states = self.q_encoder(questions)
        q_feature = self.q_norm(q_feature) # Stable FiLM generation

        # Mean-pool 49 regions for the global fusion vector -> (batch, hidden_size)
        img_mean = img_features.mean(dim=1)    
        fusion_global = self.fusion(img_mean, q_feature) # (batch, hidden_size)

        # Create h_0 and c_0 using non-linear projection from fusion state
        h_0_base = self.init_h_proj(fusion_global) # (batch, hidden)
        c_0_base = self.init_c_proj(fusion_global) # (batch, hidden)
        
        h_0 = h_0_base.unsqueeze(0).repeat(self.num_layers, 1, 1) # (num_layers, batch, hidden)
        c_0 = c_0_base.unsqueeze(0).repeat(self.num_layers, 1, 1)

        # Apply FiLM directly to the 49 spatial regions to pass into Dual Attention
        modulated_img_features = self.fusion(img_features, q_feature) # (batch, 49, hidden_size)

        # Decode with dual attention — pass modulated_img_features + q_hidden_states
        logits, coverage_loss = self.decoder((h_0, c_0), modulated_img_features, q_hidden_states, target_seq)
        return logits, coverage_loss