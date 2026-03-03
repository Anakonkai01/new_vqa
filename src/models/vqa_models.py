"""
Wrapper model:
  CNN Encoder      -> img_feature  (batch, hidden_size)
  Question Encoder -> q_feature    (batch, hidden_size)
           |
      FUSION = GatedFusion(img_feature, q_feature)  + LayerNorm
           |
      Create initial hidden state for Decoder (layer 0 only)
           |
  LSTM Decoder -> logits  (batch, seq_len, vocab_size)

Changes vs original:
  - hadamard_fusion() replaced by GatedFusion (learned soft gate + LayerNorm)
  - q_feature L2-normalized before fusion (same scale as img_feature)
  - h_0: only layer 0 receives fusion signal; upper layers start from zeros
  - dropout parameter forwarded from CLI down to decoders
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


# ── Legacy Hadamard fusion (kept for reference / backward compat) ──────────
def hadamard_fusion(img_feature, q_feature):
    return img_feature * q_feature


# ── Gated Fusion (replaces Hadamard in all 4 models) ──────────────────────
class GatedFusion(nn.Module):
    """
    Learned soft gate between image and question features.

    Instead of element-wise multiply (Hadamard), a sigmoid gate decides how
    much of each modality to let through:
        g       = sigmoid( W([img; q]) )       gate ∈ (0, 1)^H
        out     = g * img + (1-g) * q          weighted blend
        fusion  = LayerNorm(out)               stabilise h_0 magnitude

    LayerNorm ensures the resulting h_0 fed to the LSTM decoder has
    unit-ish variance regardless of encoder output scale.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, img_feat, q_feat):
        combined = torch.cat([img_feat, q_feat], dim=1)   # (B, 2H)
        g = torch.sigmoid(self.gate(combined))             # (B, H)
        out = g * img_feat + (1 - g) * q_feat             # (B, H)
        return self.norm(out)                              # (B, H)


# ── Shared helper: build h_0 so only layer 0 gets the fusion signal ────────
def _build_h0(fusion, num_layers):
    """
    layer 0  = fusion vector (carries image+question information)
    layer 1+ = zeros         (free to develop its own representation)
    """
    h0 = torch.zeros(num_layers, fusion.size(0), fusion.size(1), device=fusion.device)
    h0[0] = fusion
    return h0


# ── Model A: no attention + scratch CNN ───────────────────────────────────
class VQAModelA(nn.Module):
    def __init__(self, vocab_size, answer_vocab_size,
                 embed_size=512, hidden_size=1024, num_layers=2, dropout=0.3):
        super().__init__()

        self.num_layers = num_layers

        self.i_encoder = SimpleCNN(output_size=hidden_size)
        self.q_encoder = QuestionEncoder(vocab_size=vocab_size, embed_size=embed_size,
                                         hidden_size=hidden_size, num_layers=num_layers)
        self.fusion  = GatedFusion(hidden_size)
        self.decoder = LSTMDecoder(vocab_size=answer_vocab_size, embed_size=embed_size,
                                   hidden_size=hidden_size, num_layers=num_layers,
                                   dropout=dropout)

    def forward(self, images, questions, target_seq):
        img_feature = F.normalize(self.i_encoder(images), p=2, dim=1)   # (B, H)
        q_feature   = F.normalize(self.q_encoder(questions), p=2, dim=1) # (B, H)

        fusion = self.fusion(img_feature, q_feature)   # (B, H)

        h_0 = _build_h0(fusion, self.num_layers)
        c_0 = torch.zeros_like(h_0)

        logits = self.decoder((h_0, c_0), target_seq)
        return logits


# ── Model B: ResNet101 (pretrained, frozen) + no attention ────────────────
class VQAModelB(nn.Module):
    def __init__(self, vocab_size, answer_vocab_size,
                 embed_size=512, hidden_size=1024, num_layers=2,
                 freeze=True, dropout=0.3):
        super().__init__()

        self.num_layers = num_layers

        self.i_encoder = ResNetEncoder(output_size=hidden_size, freeze=freeze)
        self.q_encoder = QuestionEncoder(vocab_size=vocab_size, embed_size=embed_size,
                                         hidden_size=hidden_size, num_layers=num_layers)
        self.fusion  = GatedFusion(hidden_size)
        self.decoder = LSTMDecoder(vocab_size=answer_vocab_size, embed_size=embed_size,
                                   hidden_size=hidden_size, num_layers=num_layers,
                                   dropout=dropout)

    def forward(self, images, questions, target_seq):
        img_feature = F.normalize(self.i_encoder(images), p=2, dim=1)
        q_feature   = F.normalize(self.q_encoder(questions), p=2, dim=1)

        fusion = self.fusion(img_feature, q_feature)

        h_0 = _build_h0(fusion, self.num_layers)
        c_0 = torch.zeros_like(h_0)

        logits = self.decoder((h_0, c_0), target_seq)
        return logits


# ── Model C: SimpleCNN Spatial + Bahdanau Attention ───────────────────────
class VQAModelC(nn.Module):
    def __init__(self, vocab_size, answer_vocab_size,
                 embed_size=512, hidden_size=1024, num_layers=2,
                 attn_dim=512, dropout=0.3):
        super().__init__()

        self.num_layers = num_layers

        self.i_encoder = SimpleCNNSpatial(output_size=hidden_size)
        self.q_encoder = QuestionEncoder(vocab_size=vocab_size, embed_size=embed_size,
                                         hidden_size=hidden_size, num_layers=num_layers)
        self.fusion  = GatedFusion(hidden_size)
        self.decoder = LSTMDecoderWithAttention(
            vocab_size=answer_vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            attn_dim=attn_dim,
            dropout=dropout
        )

    def forward(self, images, questions, target_seq):
        img_features = F.normalize(self.i_encoder(images), p=2, dim=-1)  # (B, 49, H)
        q_feature    = F.normalize(self.q_encoder(questions), p=2, dim=1) # (B, H)

        img_mean = img_features.mean(dim=1)                               # (B, H)
        fusion   = self.fusion(img_mean, q_feature)                       # (B, H)

        h_0 = _build_h0(fusion, self.num_layers)
        c_0 = torch.zeros_like(h_0)

        logits = self.decoder((h_0, c_0), img_features, target_seq)
        return logits


# ── Model D: ResNet101 Spatial (pretrained, frozen) + Bahdanau Attention ──
class VQAModelD(nn.Module):
    def __init__(self, vocab_size, answer_vocab_size,
                 embed_size=512, hidden_size=1024, num_layers=2,
                 attn_dim=512, freeze_cnn=True, dropout=0.3):
        super().__init__()

        self.num_layers = num_layers

        self.i_encoder = ResNetSpatialEncoder(output_size=hidden_size, freeze=freeze_cnn)
        self.q_encoder = QuestionEncoder(vocab_size=vocab_size, embed_size=embed_size,
                                         hidden_size=hidden_size, num_layers=num_layers)
        self.fusion  = GatedFusion(hidden_size)
        self.decoder = LSTMDecoderWithAttention(
            vocab_size=answer_vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            attn_dim=attn_dim,
            dropout=dropout
        )

    def forward(self, images, questions, target_seq):
        img_features = F.normalize(self.i_encoder(images), p=2, dim=-1)  # (B, 49, H)
        q_feature    = F.normalize(self.q_encoder(questions), p=2, dim=1) # (B, H)

        img_mean = img_features.mean(dim=1)
        fusion   = self.fusion(img_mean, q_feature)

        h_0 = _build_h0(fusion, self.num_layers)
        c_0 = torch.zeros_like(h_0)

        logits = self.decoder((h_0, c_0), img_features, target_seq)
        return logits
