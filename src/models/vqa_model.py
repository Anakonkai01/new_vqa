"""
VQAModel — unified VQA model replacing VQAModelA/B/C/D/E/F.

Design: composition over inheritance.
  i_encoder + q_encoder + fusion + decoder
  assembled from ModelConfig via build_encoder / build_decoder / build_fusion.

Key properties:
  - Single class for all model types (A through G)
  - forward() returns VQAOutput (not the (logits, cov_loss) tuple from legacy classes)
  - encode() separates encoding from decoding — required by SCST
    (encode once, then greedy-decode + sample-decode without re-encoding)
  - load_legacy_checkpoint() loads A-F checkpoints with strict=False
    (same submodule names: i_encoder, q_encoder, fusion, decoder)
  - Registered as 'G' in MODEL_REGISTRY via @register_model decorator

Backward compat:
  - Old models A-F continue to work via _build_legacy() in registry.py
  - Their submodule names match VQAModel's names, so checkpoints load directly
  - strict=False tolerates missing G2/G3/G5 params not in old checkpoints
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure src/ is on path
_SRC = os.path.dirname(os.path.dirname(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from config.model_config import ModelConfig
from models.base import VQAOutput, VQAModelProtocol
from models.registry import register_model, build_encoder, build_decoder, build_fusion


# ---------------------------------------------------------------------------
# QuestionEncoder import (shared across all models)
# ---------------------------------------------------------------------------

def _build_question_encoder(config: ModelConfig,
                             pretrained_q_emb: Optional[torch.Tensor]) -> nn.Module:
    """Build QuestionEncoder from ModelConfig fields."""
    from models.encoder_question import QuestionEncoder
    enc = config.encoder
    return QuestionEncoder(
        vocab_size=enc.q_vocab_size,
        embed_size=config.decoder.embed_size,
        hidden_size=config.decoder.hidden_size,
        num_layers=config.decoder.num_layers,
        dropout=config.decoder.dropout,
        pretrained_embeddings=pretrained_q_emb,
        use_highway=enc.q_highway,
        use_char_cnn=enc.q_char_cnn,
    )


# ---------------------------------------------------------------------------
# VQAModel
# ---------------------------------------------------------------------------

@register_model("G")
class VQAModel(nn.Module, VQAModelProtocol):
    """
    Unified VQA model for all model types (A–G).

    Composition:
      self.i_encoder  — visual encoder (SimpleCNN / ResNet / ConvNeXt / BUTD)
      self.q_encoder  — QuestionEncoder (BiLSTM)
      self.fusion     — GatedFusion or MUTANFusion
      self.decoder    — LSTMDecoder or LSTMDecoderWithAttention

    Returns VQAOutput from forward(), not the (logits, cov_loss) tuple.

    The submodule names (i_encoder, q_encoder, fusion, decoder) intentionally
    match the legacy VQAModelA/B/C/D/E/F — so A-F checkpoints load with
    strict=False and no key remapping.
    """

    def __init__(
        self,
        config: ModelConfig,
        pretrained_q_emb: Optional[torch.Tensor] = None,
        pretrained_a_emb: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.config     = config
        self.num_layers = config.decoder.num_layers
        self.is_butd    = config.encoder.vision_type == "butd"
        self.use_attn   = config.decoder.use_attention

        # ── Sub-modules ────────────────────────────────────────────────────
        self.i_encoder = build_encoder(config)
        self.q_encoder = _build_question_encoder(config, pretrained_q_emb)
        self.fusion    = build_fusion(config)
        self.decoder   = _build_decoder_with_emb(config, pretrained_a_emb)

        # Convenience flag: True when decoder is LSTMDecoderG (G2/G5 args needed)
        self.is_model_g_decoder = (
            config.decoder.use_pgn3 or config.decoder.use_len_cond
        )

        # G3: InfoNCE projection heads (training only — not used at inference)
        self.infonce_heads = None
        if config.infonce:
            from models.infonce import InfoNCEProjectionHeads
            self.infonce_heads = InfoNCEProjectionHeads(
                img_dim=config.encoder.output_size,
                text_dim=config.decoder.hidden_size,
                z_dim=config.infonce_proj_dim,
                tau=config.infonce_tau,
            )

    # -----------------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        pretrained_q_emb: Optional[torch.Tensor] = None,
        pretrained_a_emb: Optional[torch.Tensor] = None,
    ) -> "VQAModel":
        """Explicit factory — same as __init__, kept for API clarity."""
        return cls(config, pretrained_q_emb, pretrained_a_emb)

    # -----------------------------------------------------------------------
    # encode() — called once by SCST before two decode passes
    # -----------------------------------------------------------------------

    def encode(
        self,
        images_or_feats: torch.Tensor,
        questions: torch.Tensor,
        img_mask: Optional[torch.Tensor] = None,
        char_seqs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run visual + question encoders. Returns raw features for downstream use.

        Returns:
            V       (B, k, H)  — projected + L2-normed visual features (spatial)
                    (B, H)     — global feat for non-spatial models (A/B)
            q_feat  (B, H)     — attention-pooled question summary
            Q_H     (B, L, H)  — full BiLSTM hidden sequence for MHCA
        """
        # Visual encoding
        V = F.normalize(self.i_encoder(images_or_feats), p=2, dim=-1)

        # Question encoding: QuestionEncoder returns (q_feat, q_hidden_states)
        q_feat, Q_H = self.q_encoder(questions)

        return V, q_feat, Q_H

    # -----------------------------------------------------------------------
    # _fuse_and_init() — shared between forward() and SCST
    # -----------------------------------------------------------------------

    def _fuse_and_init(
        self,
        V: torch.Tensor,
        q_feat: torch.Tensor,
        img_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute fusion vector and (h_0, c_0) for the LSTM decoder.

        For BUTD (Model F/G): masked mean before fusion.
        For spatial CNN (Model C/D/E): plain mean over 49 fixed regions.
        For global CNN (Model A/B): V is already (B, H), no mean needed.

        Returns h_0, c_0: each (num_layers, B, H).
        """
        if V.dim() == 3:
            # Spatial: (B, k, H) → (B, H) via (masked) mean
            if img_mask is not None:
                valid_counts = img_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
                img_mean = (V * img_mask.unsqueeze(-1).float()).sum(dim=1) / valid_counts
            else:
                img_mean = V.mean(dim=1)
        else:
            # Global: (B, H) already
            img_mean = V

        # Fusion: MUTAN expects (q, v), GatedFusion expects (img, q)
        # Both fusion modules accept (a, b) and apply their own order internally.
        # MUTAN convention: q first (matches VQAModelE/F). GatedFusion: img first.
        if self.config.fusion.fusion_type == "mutan":
            fused = self.fusion(q_feat, img_mean)   # MUTAN: q first
        else:
            fused = self.fusion(img_mean, q_feat)   # GatedFusion: img first

        h_0 = fused.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)
        return h_0, c_0

    # -----------------------------------------------------------------------
    # forward() — full training pass, returns VQAOutput
    # -----------------------------------------------------------------------

    def forward(
        self,
        images_or_feats: torch.Tensor,
        questions: torch.Tensor,
        targets: torch.Tensor,
        img_mask: Optional[torch.Tensor] = None,
        char_seqs: Optional[torch.Tensor] = None,
        length_bin: Optional[torch.Tensor] = None,
        label_tokens: Optional[torch.Tensor] = None,
    ) -> VQAOutput:
        """
        Full forward pass (encode + fuse + decode). Returns VQAOutput.

        Args:
            images_or_feats : (B, 3, H, W) raw images OR (B, k, feat_dim) BUTD feats
            questions       : (B, q_len) token indices
            targets         : (B, t_len) token indices (teacher forcing)
            img_mask        : (B, k) bool — True = valid BUTD region; None for images
            length_bin      : (B,) int64 in {0,1,2} — G5 length conditioning; None if not used
            label_tokens    : (B, k, max_toks) — G2 visual label token indices; None if unused

        Returns:
            VQAOutput with logits always set; optional attention/coverage/pgn/infonce fields.
        """
        V, q_feat, Q_H = self.encode(images_or_feats, questions, img_mask)
        h_0, c_0 = self._fuse_and_init(V, q_feat, img_mask)

        if self.use_attn:
            if self.is_model_g_decoder:
                # LSTMDecoderG: accepts G2 label_tokens + G5 length_bin
                logits, coverage_loss = self.decoder(
                    (h_0, c_0), V, Q_H, targets,
                    q_token_ids=questions,
                    img_mask=img_mask,
                    length_bin=length_bin,
                    label_tokens=label_tokens,
                )
            else:
                # LSTMDecoderWithAttention (Models C-F)
                logits, coverage_loss = self.decoder(
                    (h_0, c_0), V, Q_H, targets,
                    q_token_ids=questions,
                    img_mask=img_mask,
                )
        else:
            # LSTMDecoder (A/B): returns logits tensor directly
            logits = self.decoder((h_0, c_0), targets)
            return VQAOutput(logits=logits)

        # G3: InfoNCE — compute only during training
        infonce_z = None
        if self.infonce_heads is not None and self.training:
            if V.dim() == 3:
                if img_mask is not None:
                    cnt   = img_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
                    v_bar = (V * img_mask.unsqueeze(-1).float()).sum(dim=1) / cnt
                else:
                    v_bar = V.mean(dim=1)
            else:
                v_bar = V
            infonce_z = self.infonce_heads(v_bar, q_feat)

        return VQAOutput(logits=logits, coverage=None, infonce_z=infonce_z)

    # -----------------------------------------------------------------------
    # coverage_loss() — separate from forward() so trainers can use it
    # -----------------------------------------------------------------------

    def forward_with_cov(
        self,
        images_or_feats: torch.Tensor,
        questions: torch.Tensor,
        targets: torch.Tensor,
        img_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[VQAOutput, torch.Tensor]:
        """
        Same as forward() but also returns coverage_loss scalar.
        Used by the trainer's cross-entropy phase when coverage is enabled.
        """
        V, q_feat, Q_H = self.encode(images_or_feats, questions, img_mask)
        h_0, c_0 = self._fuse_and_init(V, q_feat, img_mask)

        if self.use_attn:
            if self.is_model_g_decoder:
                logits, coverage_loss = self.decoder(
                    (h_0, c_0), V, Q_H, targets,
                    q_token_ids=questions, img_mask=img_mask,
                )
            else:
                logits, coverage_loss = self.decoder(
                    (h_0, c_0), V, Q_H, targets,
                    q_token_ids=questions, img_mask=img_mask,
                )
            return VQAOutput(logits=logits), coverage_loss
        else:
            logits = self.decoder((h_0, c_0), targets)
            cov_zero = logits.new_zeros(1)
            return VQAOutput(logits=logits), cov_zero

    # -----------------------------------------------------------------------
    # decode_step() — single autoregressive step for inference
    # -----------------------------------------------------------------------

    def decode_step(
        self,
        token: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        V: torch.Tensor,
        Q_H: Optional[torch.Tensor] = None,
        coverage: Optional[torch.Tensor] = None,
        img_mask: Optional[torch.Tensor] = None,
        q_token_ids: Optional[torch.Tensor] = None,
        length_bin: Optional[torch.Tensor] = None,
        label_tokens: Optional[torch.Tensor] = None,
    ):
        """
        Single autoregressive decode step for greedy/beam inference.

        Args:
            token       : (B, 1) — current input token
            h           : (num_layers, B, H)
            c           : (num_layers, B, H)
            V           : (B, k, H) visual features from encode()
            Q_H         : (B, L, H) question states (None for A/B)
            coverage    : (B, k) accumulated attention or None
            img_mask    : (B, k) bool or None
            q_token_ids : (B, q_len) for PGN; None if not used
            length_bin  : (B,) int64 G5; None defaults to LONG (2)
            label_tokens: (B, k, max_t) int64 G2; None → skip visual copy

        Returns:
            logit    : (B, V_size)
            h_new    : (num_layers, B, H)
            c_new    : (num_layers, B, H)
            img_alpha: (B, k) or None
            coverage : (B, k) or None — updated
        """
        if self.use_attn:
            if self.is_model_g_decoder:
                logit, (h_new, c_new), img_alpha, coverage_new = self.decoder.decode_step(
                    token, (h, c), V, Q_H,
                    coverage=coverage,
                    q_token_ids=q_token_ids,
                    img_mask=img_mask,
                    length_bin=length_bin,
                    label_tokens=label_tokens,
                )
            else:
                logit, (h_new, c_new), img_alpha, coverage_new = self.decoder.decode_step(
                    token, (h, c), V, Q_H,
                    coverage=coverage,
                    q_token_ids=q_token_ids,
                    img_mask=img_mask,
                )
            return logit, h_new, c_new, img_alpha, coverage_new
        else:
            # LSTMDecoder (A/B): single step
            embed = self.decoder.embedding(token)       # (B, 1, E)
            out, (h_new, c_new) = self.decoder.lstm(embed, (h, c))
            logit = self.decoder.fc(out.squeeze(1))     # (B, V)
            return logit, h_new, c_new, None, None

    # -----------------------------------------------------------------------
    # CNN fine-tuning helpers (same API as VQAModelE/F)
    # -----------------------------------------------------------------------

    def unfreeze_cnn(self):
        """Unfreeze top layers of the visual encoder backbone (Phase 2)."""
        if hasattr(self.i_encoder, "unfreeze_top_layers"):
            self.i_encoder.unfreeze_top_layers()

    def cnn_backbone_params(self):
        """Return backbone params for differential LR (lower LR for backbone)."""
        if hasattr(self.i_encoder, "backbone_params"):
            return self.i_encoder.backbone_params()
        return []

    # -----------------------------------------------------------------------
    # Checkpoint loading (backward compat with A-F)
    # -----------------------------------------------------------------------

    def load_legacy_checkpoint(self, ckpt_path: str) -> dict:
        """
        Load a Model A-F checkpoint into this VQAModel.

        Submodule names are identical (i_encoder, q_encoder, fusion, decoder),
        so keys map 1:1. strict=False handles:
          - Missing keys: G2/G3/G5 params not in old checkpoints → random init
          - Unexpected keys: none expected

        Also handles checkpoints where state_dict is nested under 'model' key.

        Returns the result of load_state_dict (with missing/unexpected key lists).
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Unwrap if checkpoint is a training dict (has 'model' or 'state_dict' key)
        if isinstance(ckpt, dict):
            if "model" in ckpt:
                state_dict = ckpt["model"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        result = self.load_state_dict(state_dict, strict=False)
        return result


# ---------------------------------------------------------------------------
# _build_decoder_with_emb — helper to pass pretrained_a_emb to decoder
# ---------------------------------------------------------------------------

def _build_decoder_with_emb(
    config: ModelConfig,
    pretrained_a_emb: Optional[torch.Tensor],
) -> nn.Module:
    """
    Build decoder with GloVe answer embeddings wired in.

    build_decoder() in registry.py doesn't pass pretrained_a_emb because
    registry.py doesn't know the embedding at build time.  VQAModel calls
    this helper instead.
    """
    from models.decoder_lstm import LSTMDecoder
    from models.decoder_attention import LSTMDecoderWithAttention

    dec = config.decoder

    if not dec.use_attention:
        return LSTMDecoder(
            vocab_size=dec.a_vocab_size,
            embed_size=dec.embed_size,
            hidden_size=dec.hidden_size,
            num_layers=dec.num_layers,
            dropout=dec.dropout,
            pretrained_embeddings=pretrained_a_emb,
        )

    # Model G: use LSTMDecoderG when G2 (3-way PGN) or G5 (length cond) is enabled
    if getattr(dec, 'use_pgn3', False) or getattr(dec, 'use_len_cond', False):
        from models.decoders.attention import LSTMDecoderG
        return LSTMDecoderG(
            vocab_size=dec.a_vocab_size,
            embed_size=dec.embed_size,
            hidden_size=dec.hidden_size,
            num_layers=dec.num_layers,
            dropout=dec.dropout,
            pretrained_embeddings=pretrained_a_emb,
            use_coverage=dec.use_coverage,
            use_layer_norm=dec.use_layer_norm,
            use_dropconnect=dec.dropconnect > 0.0,
            len_embed_dim=getattr(dec, 'len_embed_dim', 64),
        )

    return LSTMDecoderWithAttention(
        vocab_size=dec.a_vocab_size,
        embed_size=dec.embed_size,
        hidden_size=dec.hidden_size,
        num_layers=dec.num_layers,
        attn_dim=dec.hidden_size // 2,
        dropout=dec.dropout,
        pretrained_embeddings=pretrained_a_emb,
        use_coverage=dec.use_coverage,
        use_layer_norm=dec.use_layer_norm,
        use_dropconnect=dec.dropconnect > 0.0,
        use_pgn=dec.use_pgn,
    )
