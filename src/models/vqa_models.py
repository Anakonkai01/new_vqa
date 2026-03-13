"""VQA model wrappers combining encoder, fusion, and decoder modules.

Five model variants are defined, each combining a different image encoder
with the shared question encoder and one of two decoder types:

    Model A — SimpleCNN       + GatedFusion + LSTMDecoder
    Model B — ResNetEncoder   + GatedFusion + LSTMDecoder
    Model C — SimpleCNNSpatial + GatedFusion + LSTMDecoderWithAttention
    Model D — ResNetSpatial   + GatedFusion + LSTMDecoderWithAttention
    Model E — CLIPViTEncoder  + FiLMFusion  + LSTMDecoderWithAttention  ← flagship

Common Design Pattern (Models A–D)
------------------------------------
1. ``img_feature  = L2_normalize(encoder(image))``
2. ``q_feature    = question_encoder(question)``
3. ``fusion       = GatedFusion(img_feature, q_feature)``  → (B, hidden_size)
4. ``h_0 = fusion.unsqueeze(0).repeat(num_layers, 1, 1)``
5. ``logits = decoder(h_0, target_seq [, img_features, q_hidden_states])``

Model E additionally uses FiLMFusion and non-linear state projections.

Return types
------------
- Models A and B: ``forward()`` returns ``logits`` only — ``FloatTensor (B, T, V)``
- Models C, D, E: ``forward()`` returns ``(logits, coverage_loss)``
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.encoder_cnn import (
    CLIPViTEncoder,
    ResNetEncoder,
    ResNetSpatialEncoder,
    SimpleCNN,
    SimpleCNNSpatial,
)
from models.encoder_question import QuestionEncoder
from models.decoder_lstm import LSTMDecoder
from models.decoder_attention import LSTMDecoderWithAttention


# ── Fusion modules ────────────────────────────────────────────────────────────

class GatedFusion(nn.Module):
    """Gated multimodal fusion module.

    A learnable gate decides how much image vs. question information to retain::

        h_img   = tanh(W_img(img_feature))            # (B, hidden_size)
        h_q     = tanh(W_q(q_feature))                # (B, hidden_size)
        gate    = σ(W_g([img_feature; q_feature]))     # (B, hidden_size)
        output  = gate * h_img + (1 - gate) * h_q     # (B, hidden_size)

    Args:
        hidden_size: Dimensionality of both input feature vectors.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.fc_img  = nn.Linear(hidden_size, hidden_size)
        self.fc_q    = nn.Linear(hidden_size, hidden_size)
        self.fc_gate = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, img_feature: Tensor, q_feature: Tensor) -> Tensor:
        """Fuse image and question features via a learned gate.

        Args:
            img_feature: ``FloatTensor (B, hidden_size)``
            q_feature:   ``FloatTensor (B, hidden_size)``

        Returns:
            ``FloatTensor (B, hidden_size)`` — fused representation.
        """
        h_img = torch.tanh(self.fc_img(img_feature))    # (B, hidden_size)
        h_q   = torch.tanh(self.fc_q(q_feature))        # (B, hidden_size)
        gate  = torch.sigmoid(
            self.fc_gate(torch.cat([img_feature, q_feature], dim=1))
        )                                                 # (B, hidden_size)
        return gate * h_img + (1.0 - gate) * h_q         # (B, hidden_size)


class FiLMFusion(nn.Module):
    """Feature-wise Linear Modulation (FiLM) fusion.

    The question feature generates per-channel scale (γ) and shift (β)
    parameters via a small MLP, which are then applied element-wise to the
    image features::

        [γ, β] = MLP(q_feature)            # each (B, hidden_size)
        output  = γ * LayerNorm(img_feat) + β

    FiLM was introduced by Perez et al. (2018) for visual reasoning tasks.
    It is more expressive than Hadamard fusion because γ and β are
    *individually predicted* from the question — i.e., each image channel
    gets its own question-conditioned scale and shift.

    Supports both global ``(B, hidden_size)`` and spatial ``(B, N, hidden_size)``
    image feature tensors via broadcasting.

    Args:
        hidden_size: Dimensionality of image and question features.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, hidden_size * 2),  # outputs [γ || β]
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, img_features: Tensor, q_feature: Tensor) -> Tensor:
        """Apply FiLM modulation to image features conditioned on question.

        Args:
            img_features: ``FloatTensor (B, hidden_size)`` or
                ``FloatTensor (B, N, hidden_size)`` — image features.
            q_feature:    ``FloatTensor (B, hidden_size)`` — question feature.

        Returns:
            Modulated image features of the same shape as *img_features*.
        """
        film_params = self.mlp(q_feature)            # (B, hidden_size * 2)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)  # each (B, hidden_size)

        # Broadcast γ and β over the spatial dimension when img_features is 3-D.
        if img_features.dim() == 3:
            gamma = gamma.unsqueeze(1)  # (B, 1, hidden_size)
            beta  = beta.unsqueeze(1)   # (B, 1, hidden_size)

        # LayerNorm stabilises activations before modulation.
        modulated = gamma * self.layer_norm(img_features) + beta
        return modulated  # same shape as img_features


# ── Model A ───────────────────────────────────────────────────────────────────

class VQAModelA(nn.Module):
    """VQA Model A — scratch SimpleCNN + GatedFusion + LSTMDecoder (no attention).

    Args:
        vocab_size: Size of the question vocabulary.
        answer_vocab_size: Size of the answer vocabulary.
        embed_size: Token embedding dimension.
        hidden_size: LSTM and image feature dimension.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout rate.
        pretrained_q_emb: Optional GloVe matrix for question embedding.
        pretrained_a_emb: Optional GloVe matrix for answer embedding.
    """

    def __init__(
        self,
        vocab_size: int,
        answer_vocab_size: int,
        embed_size: int = 512,
        hidden_size: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.5,
        pretrained_q_emb: Optional[Tensor] = None,
        pretrained_a_emb: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.i_encoder  = SimpleCNN(output_size=hidden_size)
        self.q_encoder  = QuestionEncoder(
            vocab_size=vocab_size, embed_size=embed_size,
            hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout, pretrained_embeddings=pretrained_q_emb,
        )
        self.fusion  = GatedFusion(hidden_size)
        self.decoder = LSTMDecoder(
            vocab_size=answer_vocab_size, embed_size=embed_size,
            hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout, pretrained_embeddings=pretrained_a_emb,
        )

    def forward(self, images: Tensor, questions: Tensor, target_seq: Tensor) -> Tensor:
        """Teacher-forcing forward pass.

        Args:
            images:     ``FloatTensor (B, 3, 224, 224)``
            questions:  ``LongTensor  (B, Q)``
            target_seq: ``LongTensor  (B, T)`` — ``[<start>, w_1, ..., w_{T-1}]``

        Returns:
            ``FloatTensor (B, T, answer_vocab_size)`` — logits.
        """
        img_feat = F.normalize(self.i_encoder(images), p=2, dim=1)
        # img_feat: (B, hidden_size) — L2-normalised global vector

        q_feat, _ = self.q_encoder(questions)
        # q_feat: (B, hidden_size)

        fusion = self.fusion(img_feat, q_feat)
        # fusion: (B, hidden_size)

        h_0 = fusion.unsqueeze(0).repeat(self.num_layers, 1, 1)
        # h_0: (num_layers, B, hidden_size)
        c_0 = torch.zeros_like(h_0)

        logits = self.decoder((h_0, c_0), target_seq)
        # logits: (B, T, answer_vocab_size)
        return logits


# ── Model B ───────────────────────────────────────────────────────────────────

class VQAModelB(nn.Module):
    """VQA Model B — pretrained ResNet101 + GatedFusion + LSTMDecoder (no attention).

    Args:
        vocab_size: Size of the question vocabulary.
        answer_vocab_size: Size of the answer vocabulary.
        embed_size: Token embedding dimension.
        hidden_size: LSTM and image feature dimension.
        num_layers: Number of stacked LSTM layers.
        freeze: If True, ResNet backbone is frozen at init.
        dropout: Dropout rate.
        pretrained_q_emb: Optional GloVe matrix for question embedding.
        pretrained_a_emb: Optional GloVe matrix for answer embedding.
    """

    def __init__(
        self,
        vocab_size: int,
        answer_vocab_size: int,
        embed_size: int = 512,
        hidden_size: int = 1024,
        num_layers: int = 2,
        freeze: bool = True,
        dropout: float = 0.5,
        pretrained_q_emb: Optional[Tensor] = None,
        pretrained_a_emb: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.i_encoder  = ResNetEncoder(output_size=hidden_size, freeze=freeze)
        self.q_encoder  = QuestionEncoder(
            vocab_size=vocab_size, embed_size=embed_size,
            hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout, pretrained_embeddings=pretrained_q_emb,
        )
        self.fusion  = GatedFusion(hidden_size)
        self.decoder = LSTMDecoder(
            vocab_size=answer_vocab_size, embed_size=embed_size,
            hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout, pretrained_embeddings=pretrained_a_emb,
        )

    def forward(self, images: Tensor, questions: Tensor, target_seq: Tensor) -> Tensor:
        """Teacher-forcing forward pass.

        Args:
            images:     ``FloatTensor (B, 3, 224, 224)``
            questions:  ``LongTensor  (B, Q)``
            target_seq: ``LongTensor  (B, T)``

        Returns:
            ``FloatTensor (B, T, answer_vocab_size)`` — logits.
        """
        img_feat = F.normalize(self.i_encoder(images), p=2, dim=1)
        # img_feat: (B, hidden_size)

        q_feat, _ = self.q_encoder(questions)
        # q_feat: (B, hidden_size)

        fusion = self.fusion(img_feat, q_feat)
        # fusion: (B, hidden_size)

        h_0 = fusion.unsqueeze(0).repeat(self.num_layers, 1, 1)
        # h_0: (num_layers, B, hidden_size)
        c_0 = torch.zeros_like(h_0)

        logits = self.decoder((h_0, c_0), target_seq)
        # logits: (B, T, answer_vocab_size)
        return logits


# ── Model C ───────────────────────────────────────────────────────────────────

class VQAModelC(nn.Module):
    """VQA Model C — scratch SimpleCNNSpatial + GatedFusion + Dual Attention.

    Args:
        vocab_size: Size of the question vocabulary.
        answer_vocab_size: Size of the answer vocabulary.
        embed_size: Token embedding dimension.
        hidden_size: LSTM and image feature dimension.
        num_layers: Number of stacked LSTM layers.
        attn_dim: Internal attention projection dimension.
        dropout: Dropout rate.
        pretrained_q_emb: Optional GloVe matrix for question embedding.
        pretrained_a_emb: Optional GloVe matrix for answer embedding.
        use_coverage: Enable the coverage penalty on image attention.
    """

    def __init__(
        self,
        vocab_size: int,
        answer_vocab_size: int,
        embed_size: int = 512,
        hidden_size: int = 1024,
        num_layers: int = 2,
        attn_dim: int = 512,
        dropout: float = 0.5,
        pretrained_q_emb: Optional[Tensor] = None,
        pretrained_a_emb: Optional[Tensor] = None,
        use_coverage: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.i_encoder  = SimpleCNNSpatial(output_size=hidden_size)
        self.q_encoder  = QuestionEncoder(
            vocab_size=vocab_size, embed_size=embed_size,
            hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout, pretrained_embeddings=pretrained_q_emb,
        )
        self.fusion  = GatedFusion(hidden_size)
        self.decoder = LSTMDecoderWithAttention(
            vocab_size=answer_vocab_size, embed_size=embed_size,
            hidden_size=hidden_size, num_layers=num_layers,
            attn_dim=attn_dim, dropout=dropout,
            pretrained_embeddings=pretrained_a_emb,
            use_coverage=use_coverage,
        )

    def forward(
        self,
        images: Tensor,
        questions: Tensor,
        target_seq: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Teacher-forcing forward pass with dual attention.

        Args:
            images:     ``FloatTensor (B, 3, 224, 224)``
            questions:  ``LongTensor  (B, Q)``
            target_seq: ``LongTensor  (B, T)``

        Returns:
            Tuple of:
                logits        – ``FloatTensor (B, T, answer_vocab_size)``
                coverage_loss – scalar tensor
        """
        img_features = F.normalize(self.i_encoder(images), p=2, dim=-1)
        # img_features: (B, 49, hidden_size) — L2-normalised per region

        q_feat, q_hidden_states = self.q_encoder(questions)
        # q_feat:          (B, hidden_size)
        # q_hidden_states: (B, Q, hidden_size)

        img_mean = img_features.mean(dim=1)          # (B, hidden_size)
        fusion   = self.fusion(img_mean, q_feat)     # (B, hidden_size)

        h_0 = fusion.unsqueeze(0).repeat(self.num_layers, 1, 1)
        # h_0: (num_layers, B, hidden_size)
        c_0 = torch.zeros_like(h_0)

        logits, coverage_loss = self.decoder(
            (h_0, c_0), img_features, q_hidden_states, target_seq
        )
        # logits: (B, T, answer_vocab_size)
        return logits, coverage_loss


# ── Model D ───────────────────────────────────────────────────────────────────

class VQAModelD(nn.Module):
    """VQA Model D — pretrained ResNetSpatial + GatedFusion + Dual Attention.

    Identical architecture to Model C but uses a pretrained ResNet101 backbone
    for higher-quality spatial features.

    Args:
        vocab_size: Size of the question vocabulary.
        answer_vocab_size: Size of the answer vocabulary.
        embed_size: Token embedding dimension.
        hidden_size: LSTM and image feature dimension.
        num_layers: Number of stacked LSTM layers.
        attn_dim: Internal attention projection dimension.
        freeze_cnn: Freeze the ResNet backbone at init.
        dropout: Dropout rate.
        pretrained_q_emb: Optional GloVe matrix for question embedding.
        pretrained_a_emb: Optional GloVe matrix for answer embedding.
        use_coverage: Enable the coverage penalty on image attention.
    """

    def __init__(
        self,
        vocab_size: int,
        answer_vocab_size: int,
        embed_size: int = 512,
        hidden_size: int = 1024,
        num_layers: int = 2,
        attn_dim: int = 512,
        freeze_cnn: bool = True,
        dropout: float = 0.5,
        pretrained_q_emb: Optional[Tensor] = None,
        pretrained_a_emb: Optional[Tensor] = None,
        use_coverage: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.i_encoder  = ResNetSpatialEncoder(output_size=hidden_size, freeze=freeze_cnn)
        self.q_encoder  = QuestionEncoder(
            vocab_size=vocab_size, embed_size=embed_size,
            hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout, pretrained_embeddings=pretrained_q_emb,
        )
        self.fusion  = GatedFusion(hidden_size)
        self.decoder = LSTMDecoderWithAttention(
            vocab_size=answer_vocab_size, embed_size=embed_size,
            hidden_size=hidden_size, num_layers=num_layers,
            attn_dim=attn_dim, dropout=dropout,
            pretrained_embeddings=pretrained_a_emb,
            use_coverage=use_coverage,
        )

    def forward(
        self,
        images: Tensor,
        questions: Tensor,
        target_seq: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Teacher-forcing forward pass with dual attention.

        Args:
            images:     ``FloatTensor (B, 3, 224, 224)``
            questions:  ``LongTensor  (B, Q)``
            target_seq: ``LongTensor  (B, T)``

        Returns:
            Tuple of:
                logits        – ``FloatTensor (B, T, answer_vocab_size)``
                coverage_loss – scalar tensor
        """
        img_features = F.normalize(self.i_encoder(images), p=2, dim=-1)
        # img_features: (B, 49, hidden_size)

        q_feat, q_hidden_states = self.q_encoder(questions)
        # q_feat:          (B, hidden_size)
        # q_hidden_states: (B, Q, hidden_size)

        img_mean = img_features.mean(dim=1)          # (B, hidden_size)
        fusion   = self.fusion(img_mean, q_feat)     # (B, hidden_size)

        h_0 = fusion.unsqueeze(0).repeat(self.num_layers, 1, 1)
        # h_0: (num_layers, B, hidden_size)
        c_0 = torch.zeros_like(h_0)

        logits, coverage_loss = self.decoder(
            (h_0, c_0), img_features, q_hidden_states, target_seq
        )
        # logits: (B, T, answer_vocab_size)
        return logits, coverage_loss


# ── Model E (Flagship) ────────────────────────────────────────────────────────

class VQAModelE(nn.Module):
    """VQA Model E — CLIP ViT-B/32 + FiLMFusion + Dual Attention (flagship).

    This model combines the strongest components from each axis:

    - **Vision**: CLIP ViT-B/32 extracts 49 semantic patch features.
    - **Language**: BiLSTM question encoder.
    - **Fusion**: FiLMFusion — question-conditioned affine modulation of image
      features (more expressive than Hadamard or Gated fusion).
    - **Decoder**: Dual-attention LSTM over both image patches and question tokens.

    FiLM is applied **twice** in ``forward()``:

    1. **Global init fusion** — modulates the mean-pooled image vector to
       produce the decoder's initial hidden state ``h_0`` and cell state ``c_0``
       via non-linear projections (``init_h_proj``, ``init_c_proj``).

    2. **Spatial feature modulation** — modulates all 49 individual patch
       features before passing them to the attention decoder. This allows the
       decoder's attention to operate on question-conditioned image features
       rather than raw CLIP features, improving the alignment between what the
       decoder "looks for" and what the image regions contain.

    Both applications share the same ``FiLMFusion`` module (same learned γ/β
    generator), which is by design: the question encodes a single semantic
    intent that should consistently modulate both the global context and the
    local spatial features.

    Args:
        vocab_size: Size of the question vocabulary.
        answer_vocab_size: Size of the answer vocabulary.
        embed_size: Token embedding dimension.
        hidden_size: LSTM and image feature dimension.
        num_layers: Number of stacked LSTM layers.
        attn_dim: Internal attention projection dimension.
        freeze_cnn: Freeze the CLIP backbone at init.
        dropout: Dropout rate.
        pretrained_q_emb: Optional GloVe matrix for question embedding.
        pretrained_a_emb: Optional GloVe matrix for answer embedding.
        use_coverage: Enable the coverage penalty on image attention.
    """

    def __init__(
        self,
        vocab_size: int,
        answer_vocab_size: int,
        embed_size: int = 512,
        hidden_size: int = 1024,
        num_layers: int = 2,
        attn_dim: int = 512,
        freeze_cnn: bool = True,
        dropout: float = 0.5,
        pretrained_q_emb: Optional[Tensor] = None,
        pretrained_a_emb: Optional[Tensor] = None,
        use_coverage: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers  = num_layers
        self.hidden_size = hidden_size

        self.i_encoder = CLIPViTEncoder(output_size=hidden_size, freeze=freeze_cnn)
        self.q_encoder = QuestionEncoder(
            vocab_size=vocab_size, embed_size=embed_size,
            hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout, pretrained_embeddings=pretrained_q_emb,
        )
        self.q_norm = nn.LayerNorm(hidden_size)  # stabilise FiLM parameter generation

        # Shared FiLM module — applied to both global and spatial features.
        self.fusion = FiLMFusion(hidden_size)

        # Non-linear projections for initial decoder state — more expressive than
        # a simple repeat of the fusion vector (adds model capacity at the boundary
        # between the encoder and the decoder).
        self.init_h_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.init_c_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())

        self.decoder = LSTMDecoderWithAttention(
            vocab_size=answer_vocab_size, embed_size=embed_size,
            hidden_size=hidden_size, num_layers=num_layers,
            attn_dim=attn_dim, dropout=dropout,
            pretrained_embeddings=pretrained_a_emb,
            use_coverage=use_coverage,
        )

    def forward(
        self,
        images: Tensor,
        questions: Tensor,
        target_seq: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Teacher-forcing forward pass with FiLM fusion and dual attention.

        Args:
            images:     ``FloatTensor (B, 3, 224, 224)``
            questions:  ``LongTensor  (B, Q)``
            target_seq: ``LongTensor  (B, T)``

        Returns:
            Tuple of:
                logits        – ``FloatTensor (B, T, answer_vocab_size)``
                coverage_loss – scalar tensor
        """
        img_features = F.normalize(self.i_encoder(images), p=2, dim=-1)
        # img_features: (B, 49, hidden_size) — L2-normalised CLIP patch features

        q_feat, q_hidden_states = self.q_encoder(questions)
        q_feat = self.q_norm(q_feat)
        # q_feat:          (B, hidden_size) — normalised for stable FiLM generation
        # q_hidden_states: (B, Q, hidden_size)

        # ── FiLM application 1: global init state ────────────────────────────
        img_mean       = img_features.mean(dim=1)          # (B, hidden_size)
        fusion_global  = self.fusion(img_mean, q_feat)     # (B, hidden_size)

        h_0_base = self.init_h_proj(fusion_global)         # (B, hidden_size)
        c_0_base = self.init_c_proj(fusion_global)         # (B, hidden_size)

        h_0 = h_0_base.unsqueeze(0).repeat(self.num_layers, 1, 1)
        # h_0: (num_layers, B, hidden_size)
        c_0 = c_0_base.unsqueeze(0).repeat(self.num_layers, 1, 1)
        # c_0: (num_layers, B, hidden_size)

        # ── FiLM application 2: spatial feature modulation ───────────────────
        # Condition all 49 patch features on the question before attention.
        # This aligns the spatial feature space with the question's semantic intent.
        modulated_img = self.fusion(img_features, q_feat)
        # modulated_img: (B, 49, hidden_size)

        logits, coverage_loss = self.decoder(
            (h_0, c_0), modulated_img, q_hidden_states, target_seq
        )
        # logits: (B, T, answer_vocab_size)
        return logits, coverage_loss
