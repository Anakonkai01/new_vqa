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

from models.encoder_cnn import SimpleCNN, ResNetEncoder, SimpleCNNSpatial, ResNetSpatialEncoder, ConvNeXtSpatialEncoder
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






# Model A: no attention + scratch CNN
class VQAModelA(nn.Module):
    def __init__(self, vocab_size, answer_vocab_size,
                 embed_size=512, hidden_size=1024, num_layers=2, dropout=0.5,
                 pretrained_q_emb=None, pretrained_a_emb=None,
                 use_q_highway=False, use_char_cnn=False):
        super().__init__()

        self.num_layers = num_layers

        # Initialize image encoder, question encoder, and decoder
        self.i_encoder = SimpleCNN(output_size=hidden_size)
        self.q_encoder = QuestionEncoder(vocab_size=vocab_size, embed_size=embed_size,
                                         hidden_size=hidden_size, num_layers=num_layers,
                                         dropout=dropout,
                                         pretrained_embeddings=pretrained_q_emb,
                                         use_highway=use_q_highway,
                                         use_char_cnn=use_char_cnn)
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
                 pretrained_q_emb=None, pretrained_a_emb=None,
                 use_q_highway=False, use_char_cnn=False):
        super().__init__()

        self.num_layers = num_layers  # stored for use when building initial hidden state

        self.i_encoder = ResNetEncoder(output_size=hidden_size, freeze=freeze)
        self.q_encoder = QuestionEncoder(vocab_size=vocab_size, embed_size=embed_size,
                                         hidden_size=hidden_size, num_layers=num_layers,
                                         dropout=dropout,
                                         pretrained_embeddings=pretrained_q_emb,
                                         use_highway=use_q_highway,
                                         use_char_cnn=use_char_cnn)
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
                 use_coverage=False, use_layer_norm=False, use_dropconnect=False,
                 use_dcan=False, use_pgn=False, use_q_highway=False, use_char_cnn=False):
        super().__init__()

        self.num_layers = num_layers

        # CNN preserves spatial 7x7=49 regions -> output (batch, 49, hidden_size)
        # Unlike Model A: no mean pool; decoder attends over all 49 regions
        self.i_encoder = SimpleCNNSpatial(output_size=hidden_size)

        # Question encoder
        self.q_encoder = QuestionEncoder(vocab_size=vocab_size, embed_size=embed_size,
                                         hidden_size=hidden_size, num_layers=num_layers,
                                         dropout=dropout,
                                         pretrained_embeddings=pretrained_q_emb,
                                         use_highway=use_q_highway,
                                         use_char_cnn=use_char_cnn)
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
            use_coverage=use_coverage,
            use_layer_norm=use_layer_norm,
            use_dropconnect=use_dropconnect,
            use_dcan=use_dcan,
            use_pgn=use_pgn,
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
        img_features = F.normalize(self.i_encoder(images), p=2, dim=-1)  # (batch, 49, H)
        q_feature, q_hidden_states = self.q_encoder(questions)

        img_mean = img_features.mean(dim=1)
        fusion   = self.fusion(img_mean, q_feature)

        h_0 = fusion.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)

        logits, coverage_loss = self.decoder(
            (h_0, c_0), img_features, q_hidden_states, target_seq,
            q_token_ids=questions,
        )
        return logits, coverage_loss


# Model D: ResNet101 Spatial (pretrained, frozen) + Bahdanau Attention + LSTM Decoder
# Same architecture as Model C — only difference: SimpleCNNSpatial -> ResNetSpatialEncoder
# Pretrained ResNet produces higher-quality features than a scratch CNN
class VQAModelD(nn.Module):
    def __init__(self, vocab_size, answer_vocab_size,
                 embed_size=512, hidden_size=1024, num_layers=2,
                 attn_dim=512, freeze_cnn=True, dropout=0.5,
                 pretrained_q_emb=None, pretrained_a_emb=None,
                 use_coverage=False, use_layer_norm=False, use_dropconnect=False,
                 use_dcan=False, use_pgn=False, use_q_highway=False, use_char_cnn=False):
        super().__init__()

        self.num_layers = num_layers

        # ResNet101 pretrained, avgpool+fc removed, keeps spatial (batch, 49, hidden_size)
        self.i_encoder = ResNetSpatialEncoder(output_size=hidden_size, freeze=freeze_cnn)

        # Question encoder
        self.q_encoder = QuestionEncoder(vocab_size=vocab_size, embed_size=embed_size,
                                         hidden_size=hidden_size, num_layers=num_layers,
                                         dropout=dropout,
                                         pretrained_embeddings=pretrained_q_emb,
                                         use_highway=use_q_highway,
                                         use_char_cnn=use_char_cnn)
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
            use_coverage=use_coverage,
            use_layer_norm=use_layer_norm,
            use_dropconnect=use_dropconnect,
            use_dcan=use_dcan,
            use_pgn=use_pgn,
        )

    def forward(self, images, questions, target_seq):
        img_features = F.normalize(self.i_encoder(images), p=2, dim=-1)
        q_feature, q_hidden_states = self.q_encoder(questions)
        img_mean = img_features.mean(dim=1)
        fusion   = self.fusion(img_mean, q_feature)
        h_0 = fusion.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)
        logits, coverage_loss = self.decoder(
            (h_0, c_0), img_features, q_hidden_states, target_seq,
            q_token_ids=questions,
        )
        return logits, coverage_loss


# ── Tier 4: MUTAN Tucker Fusion ───────────────────────────────────────────────

class MUTANFusion(nn.Module):
    """
    Tier-4: Multimodal Tucker Fusion (MUTAN).
    Replaces GatedFusion for Model E/F.
    Captures multiplicative cross-modal interactions via Tucker decomposition.

    y = T_c x1 (q W_q) x2 (v W_v)  then BN → output
    T_c ∈ R^{t_q × t_v × d_out} is a learnable core tensor (rank-constrained).
    """
    def __init__(self, d_q, d_v, d_out, t_q=360, t_v=360):
        super().__init__()
        self.W_q  = nn.Linear(d_q, t_q, bias=False)
        self.W_v  = nn.Linear(d_v, t_v, bias=False)
        self.T_c  = nn.Parameter(0.01 * torch.randn(t_q, t_v, d_out))
        self.bn   = nn.BatchNorm1d(d_out)
        self.drop = nn.Dropout(0.5)

    def forward(self, q, v):
        """q: (B, d_q), v: (B, d_v) → (B, d_out)"""
        q_proj = self.drop(torch.tanh(self.W_q(q)))            # (B, t_q)
        v_proj = self.drop(torch.tanh(self.W_v(v)))            # (B, t_v)
        inter  = torch.einsum('bi,ijk->bjk', q_proj, self.T_c) # (B, t_v, d_out)
        out    = torch.einsum('bj,bjk->bk', v_proj, inter)     # (B, d_out)
        return self.bn(out)


# ── Model E: ConvNeXt + DCAN + MUTAN + LSTMDecoderWithAttention ──────────────

class VQAModelE(nn.Module):
    """
    Model E: ConvNeXt-Base + DCAN + MUTAN + LSTM+Attn decoder.
    Flags used for training: --model E --dcan --layer_norm --use_mutan --coverage
    Tiers implemented: 3A (ConvNeXt), 4 (MUTAN), 2 (DCAN), 1 (LayerNorm)
    """
    def __init__(self, vocab_size, answer_vocab_size,
                 embed_size=512, hidden_size=1024, num_layers=2,
                 attn_dim=512, freeze_cnn=True, dropout=0.5,
                 use_coverage=False, use_layer_norm=False, use_dropconnect=False,
                 use_dcan=True, use_mutan=True, use_pgn=False,
                 use_q_highway=False, use_char_cnn=False,
                 pretrained_q_emb=None, pretrained_a_emb=None):
        super().__init__()
        self.num_layers = num_layers
        self.model_type = 'E'

        self.i_encoder = ConvNeXtSpatialEncoder(output_size=hidden_size, freeze=freeze_cnn)
        self.q_encoder = QuestionEncoder(vocab_size=vocab_size, embed_size=embed_size,
                                         hidden_size=hidden_size, num_layers=num_layers,
                                         dropout=dropout, pretrained_embeddings=pretrained_q_emb,
                                         use_highway=use_q_highway, use_char_cnn=use_char_cnn)

        self.fusion = MUTANFusion(hidden_size, hidden_size, hidden_size) if use_mutan \
                      else GatedFusion(hidden_size)

        self.decoder = LSTMDecoderWithAttention(
            vocab_size=answer_vocab_size, embed_size=embed_size,
            hidden_size=hidden_size, num_layers=num_layers,
            attn_dim=attn_dim, dropout=dropout,
            pretrained_embeddings=pretrained_a_emb,
            use_coverage=use_coverage,
            use_layer_norm=use_layer_norm,
            use_dropconnect=use_dropconnect,
            use_dcan=use_dcan,
            use_pgn=use_pgn,
        )

    def forward(self, images, questions, target_seq):
        img_features        = F.normalize(self.i_encoder(images), p=2, dim=-1)  # (B, 49, H)
        q_feature, q_hidden = self.q_encoder(questions)
        img_mean            = img_features.mean(dim=1)
        fused               = self.fusion(q_feature, img_mean)   # MUTAN: q first
        h_0 = fused.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)
        logits, cov_loss = self.decoder(
            (h_0, c_0), img_features, q_hidden, target_seq,
            q_token_ids=questions,
        )
        return logits, cov_loss

    def unfreeze_cnn(self):
        self.i_encoder.unfreeze_top_layers()

    def cnn_backbone_params(self):
        return self.i_encoder.backbone_params()


# ── Model F: BUTD Faster R-CNN features + MHCA + MUTAN + LSTM+Attn ───────────

class VQAModelF(nn.Module):
    """
    Model F: Bottom-Up Top-Down (BUTD) features + MHCA + MUTAN + LSTM decoder.

    Identical architecture to VQAModelE but uses BUTDFeatureEncoder instead of
    ConvNeXtSpatialEncoder as the image encoder.

    The key difference: forward() receives pre-extracted feature tensors
    (B, k, feat_dim) instead of raw images (B, 3, 224, 224).
    BUTDFeatureEncoder projects feat_dim → hidden_size with a learned MLP.

    Expected to achieve highest ceiling because Faster R-CNN RoI features are
    object-centric (semantically coherent regions) whereas CNN grid features are
    spatially uniform (may include background, partial objects).

    Training requires pre-extracting features with extract_features_model_f.py first.
    --model F uses BUTDDataset + butd_collate_fn in train.py.
    """
    def __init__(self, vocab_size, answer_vocab_size,
                 feat_dim=1029, embed_size=512, hidden_size=1024, num_layers=2,
                 attn_dim=512, dropout=0.5,
                 use_coverage=False, use_layer_norm=False, use_dropconnect=False,
                 use_mutan=True, use_pgn=False,
                 use_q_highway=False, use_char_cnn=False,
                 pretrained_q_emb=None, pretrained_a_emb=None):
        super().__init__()
        self.num_layers = num_layers
        self.model_type = 'F'

        from models.encoder_cnn import BUTDFeatureEncoder
        self.i_encoder = BUTDFeatureEncoder(feat_dim=feat_dim, output_size=hidden_size)

        self.q_encoder = QuestionEncoder(
            vocab_size=vocab_size, embed_size=embed_size,
            hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout, pretrained_embeddings=pretrained_q_emb,
            use_highway=use_q_highway, use_char_cnn=use_char_cnn)

        self.fusion = MUTANFusion(hidden_size, hidden_size, hidden_size) if use_mutan \
                      else GatedFusion(hidden_size)

        self.decoder = LSTMDecoderWithAttention(
            vocab_size=answer_vocab_size, embed_size=embed_size,
            hidden_size=hidden_size, num_layers=num_layers,
            attn_dim=attn_dim, dropout=dropout,
            pretrained_embeddings=pretrained_a_emb,
            use_coverage=use_coverage,
            use_layer_norm=use_layer_norm,
            use_dropconnect=use_dropconnect,
            use_pgn=use_pgn,
        )

    def forward(self, img_feats, questions, target_seq, img_mask=None):
        """
        img_feats : (B, max_k, feat_dim) — pre-extracted BUTD features (padded)
        questions : (B, q_len)
        target_seq: (B, max_len)
        img_mask  : (B, max_k) bool or None — True = valid region, False = padding.
                    From butd_collate_fn. Required for correct masked mean and
                    masked attention in the decoder.
        """
        img_features        = F.normalize(self.i_encoder(img_feats), p=2, dim=-1)  # (B, k, H)
        q_feature, q_hidden = self.q_encoder(questions)

        # Masked mean: exclude padding zeros from the global representation.
        # Plain mean(dim=1) deflates magnitude by max_k/actual_k (e.g. if an image
        # has 20 real regions but max_k=36, mean is diluted by 36/20 = 1.8×).
        if img_mask is not None:
            valid_counts = img_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
            img_mean = (img_features * img_mask.unsqueeze(-1).float()).sum(dim=1) \
                       / valid_counts                                    # (B, H)
        else:
            img_mean = img_features.mean(dim=1)

        fused = self.fusion(q_feature, img_mean)                        # MUTAN: q first
        h_0 = fused.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)
        logits, cov_loss = self.decoder(
            (h_0, c_0), img_features, q_hidden, target_seq,
            q_token_ids=questions, img_mask=img_mask,
        )
        return logits, cov_loss