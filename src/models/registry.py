"""
Model Registry — single entry point for model construction.

Replaces all `if args.model == 'A': ... elif args.model == 'B': ...` dispatch
chains that currently exist in 10+ locations across train.py, scst.py, inference.py.

Usage:
    from src.models.registry import build_model
    from src.config.model_config import ModelConfig

    cfg = ModelConfig.model_g_full()
    model = build_model(cfg, pretrained_q_emb=glove_q, pretrained_a_emb=glove_a)

Step A: registry dispatches to existing VQAModelA/B/C/D/E/F via lazy imports.
Step C: VQAModel (unified class) will be registered under key 'G' and eventually
        replace the per-letter classes.
"""

from __future__ import annotations

import sys
import os
from typing import Callable, Dict, Optional, Type
import torch
import torch.nn as nn

# Ensure src/ is on sys.path regardless of CWD
_SRC = os.path.dirname(os.path.dirname(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from config.model_config import ModelConfig, EncoderConfig, DecoderConfig, FusionConfig


# ---------------------------------------------------------------------------
# Registry infrastructure
# ---------------------------------------------------------------------------

# Maps model_type string → constructor callable
# Populated by @register_model decorator OR explicitly at bottom of file.
MODEL_REGISTRY: Dict[str, Callable] = {}


def register_model(name: str):
    """
    Decorator: register a model class or factory function under the given name.

    @register_model("G")
    class VQAModelG(nn.Module):
        ...
    """
    def decorator(cls_or_fn):
        MODEL_REGISTRY[name] = cls_or_fn
        return cls_or_fn
    return decorator


# ---------------------------------------------------------------------------
# Sub-component builders
# ---------------------------------------------------------------------------

def build_encoder(config: ModelConfig) -> nn.Module:
    """
    Build the visual encoder from ModelConfig.

    Returns the encoder nn.Module (not yet the question encoder — that is
    instantiated inside the VQAModel wrapper because it shares params with
    the answer embedding in some configurations).
    """
    from models.encoder_cnn import (
        SimpleCNN, SimpleCNNSpatial,
        ResNetEncoder, ResNetSpatialEncoder,
        ConvNeXtSpatialEncoder, BUTDFeatureEncoder,
    )

    enc = config.encoder
    vt = enc.vision_type

    if vt == "simple_cnn":
        return SimpleCNN(output_size=enc.output_size)

    if vt == "simple_cnn_spatial":
        return SimpleCNNSpatial(output_size=enc.output_size)

    if vt == "resnet":
        return ResNetEncoder(output_size=enc.output_size)

    if vt == "resnet_spatial":
        return ResNetSpatialEncoder(output_size=enc.output_size)

    if vt == "convnext":
        return ConvNeXtSpatialEncoder(output_size=enc.output_size)

    if vt == "butd":
        # feat_dim = roi_feat_dim + geo_dim (appended before projection)
        feat_dim = enc.roi_feat_dim + enc.geo_dim
        if enc.geo_dim == 7:
            # G1: 7-dim spatial geometry → 2-layer MLP encoder
            from models.encoders.butd import BUTDFeatureEncoderG1
            return BUTDFeatureEncoderG1(feat_dim=feat_dim, output_size=enc.output_size)
        return BUTDFeatureEncoder(feat_dim=feat_dim, output_size=enc.output_size)

    raise ValueError(
        f"Unknown vision_type={vt!r}. "
        f"Valid: simple_cnn, simple_cnn_spatial, resnet, resnet_spatial, convnext, butd"
    )


def build_decoder(config: ModelConfig) -> nn.Module:
    """
    Build the LSTM decoder from ModelConfig.

    A/B  → LSTMDecoder (no attention)
    C-G  → LSTMDecoderWithAttention (dual MHCA)
    """
    from models.decoder_lstm import LSTMDecoder
    from models.decoder_attention import LSTMDecoderWithAttention

    dec = config.decoder

    if not dec.use_attention:
        # Models A and B
        return LSTMDecoder(
            vocab_size=dec.a_vocab_size,
            embed_size=dec.embed_size,
            hidden_size=dec.hidden_size,
            num_layers=dec.num_layers,
            dropout=dec.dropout,
        )

    # Model G — LSTMDecoderG (G2: 3-way PGN + G5: length embedding)
    if getattr(dec, 'use_pgn3', False) or getattr(dec, 'use_len_cond', False):
        from models.decoders.attention import LSTMDecoderG
        return LSTMDecoderG(
            vocab_size=dec.a_vocab_size,
            embed_size=dec.embed_size,
            hidden_size=dec.hidden_size,
            num_layers=dec.num_layers,
            dropout=dec.dropout,
            use_coverage=dec.use_coverage,
            use_layer_norm=dec.use_layer_norm,
            use_dropconnect=dec.dropconnect > 0.0,
            len_embed_dim=getattr(dec, 'len_embed_dim', 64),
        )

    # Models C-F — LSTMDecoderWithAttention (2-way PGN, no length emb)
    return LSTMDecoderWithAttention(
        vocab_size=dec.a_vocab_size,
        embed_size=dec.embed_size,
        hidden_size=dec.hidden_size,
        num_layers=dec.num_layers,
        attn_dim=dec.hidden_size // 2,
        dropout=dec.dropout,
        use_coverage=dec.use_coverage,
        use_layer_norm=dec.use_layer_norm,
        use_dropconnect=dec.dropconnect > 0.0,
        use_pgn=dec.use_pgn,
    )


def build_fusion(config: ModelConfig) -> nn.Module:
    """Build GatedFusion or MUTANFusion from ModelConfig."""
    from models.vqa_models import GatedFusion

    fus = config.fusion

    if fus.fusion_type == "gated":
        return GatedFusion(fus.hidden_size)

    if fus.fusion_type == "mutan":
        # Lazy import to avoid circular dep until vqa_models is reorganised in Step C
        try:
            from models.vqa_models import MUTANFusion
            return MUTANFusion(fus.hidden_size, fus.hidden_size, fus.mutan_out)
        except ImportError:
            # Fallback if MUTANFusion hasn't been extracted yet
            return GatedFusion(fus.hidden_size)

    raise ValueError(f"Unknown fusion_type={fus.fusion_type!r}. Valid: gated, mutan")


# ---------------------------------------------------------------------------
# Primary entry point
# ---------------------------------------------------------------------------

def build_model(
    config: ModelConfig,
    pretrained_q_emb: Optional[torch.Tensor] = None,
    pretrained_a_emb: Optional[torch.Tensor] = None,
) -> nn.Module:
    """
    Build and return the VQA model described by `config`.

    Dispatch order:
    1. Check MODEL_REGISTRY[config.model_type] — used by Step C/D for 'G'.
    2. Fall back to legacy VQAModelA/B/C/D/E/F constructors for backward compat.

    Args:
        config:           ModelConfig describing the full architecture.
        pretrained_q_emb: GloVe matrix for question vocab (V_Q, glove_dim) or None.
        pretrained_a_emb: GloVe matrix for answer vocab (V_A, glove_dim) or None.

    Returns:
        nn.Module ready for training (not yet on device).
    """
    model_type = config.model_type

    # --- Registered models (Step C will register 'G') ---
    if model_type in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_type](
            config=config,
            pretrained_q_emb=pretrained_q_emb,
            pretrained_a_emb=pretrained_a_emb,
        )

    # --- Legacy dispatch (Models A–F, backward compat) ---
    return _build_legacy(config, pretrained_q_emb, pretrained_a_emb)


def _build_legacy(
    config: ModelConfig,
    pretrained_q_emb: Optional[torch.Tensor],
    pretrained_a_emb: Optional[torch.Tensor],
) -> nn.Module:
    """
    Construct VQAModelA/B/C/D/E/F using original class signatures.
    Translates ModelConfig fields back to the positional/keyword args
    those classes expect.
    """
    # Lazy import — keeps circular import risk zero
    from models.vqa_models import (
        VQAModelA, VQAModelB, VQAModelC, VQAModelD, VQAModelE, VQAModelF,
    )

    enc = config.encoder
    dec = config.decoder
    fus = config.fusion

    # Shared kwargs used by most models
    common = dict(
        vocab_size=enc.q_vocab_size,
        answer_vocab_size=dec.a_vocab_size,
        embed_size=dec.embed_size,
        hidden_size=dec.hidden_size,
        num_layers=dec.num_layers,
        dropout=dec.dropout,
        pretrained_q_emb=pretrained_q_emb,
        pretrained_a_emb=pretrained_a_emb,
        use_q_highway=enc.q_highway,
        use_char_cnn=enc.q_char_cnn,
    )

    mt = config.model_type

    if mt == "A":
        return VQAModelA(**common)

    if mt == "B":
        return VQAModelB(**common)

    if mt == "C":
        return VQAModelC(**common)

    if mt == "D":
        return VQAModelD(**common)

    if mt == "E":
        return VQAModelE(
            **common,
            use_coverage=dec.use_coverage,
            use_layer_norm=dec.use_layer_norm,
            use_dropconnect=dec.dropconnect > 0.0,
            use_mutan=fus.fusion_type == "mutan",
        )

    if mt == "F":
        feat_dim = enc.roi_feat_dim + enc.geo_dim   # e.g., 2048+5=2053 or 2048+7=2055 (G1)
        return VQAModelF(
            feat_dim=feat_dim,
            **{k: v for k, v in common.items()
               if k not in ("pretrained_q_emb", "pretrained_a_emb")},
            pretrained_q_emb=pretrained_q_emb,
            pretrained_a_emb=pretrained_a_emb,
            use_coverage=dec.use_coverage,
            use_layer_norm=dec.use_layer_norm,
            use_dropconnect=dec.dropconnect > 0.0,
            use_mutan=fus.fusion_type == "mutan",
            use_pgn=dec.use_pgn,
        )

    raise ValueError(
        f"Unknown model_type={mt!r}. "
        f"Valid: A, B, C, D, E, F, G (register G via @register_model in Step C)."
    )


# ---------------------------------------------------------------------------
# Utility: infer ModelConfig from a loaded checkpoint
# ---------------------------------------------------------------------------

def config_from_checkpoint(ckpt_path: str) -> Optional[ModelConfig]:
    """
    Try to load ModelConfig stored inside a checkpoint's 'model_config' key.
    Returns None if the checkpoint predates Step A (legacy format).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("model_config", None)
    if cfg_dict is None:
        return None
    return ModelConfig.from_json(cfg_dict)
