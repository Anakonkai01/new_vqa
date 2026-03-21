"""
ModelConfig — typed dataclasses replacing 30+ loose argparse args for model architecture.

Hierarchy:
  ModelConfig
    ├── EncoderConfig   (visual + question encoder settings)
    ├── DecoderConfig   (LSTM decoder settings, G2/G5 flags)
    └── FusionConfig    (gated | mutan fusion settings)

G1–G5 flags live at the top level of ModelConfig for easy CLI toggling.

Serialization: to_json() / from_json() for checkpoint metadata.
Bridge:        from_args(argparse.Namespace) for backward compat with train.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# EncoderConfig
# ---------------------------------------------------------------------------

@dataclass
class EncoderConfig:
    # --- Visual encoder ---
    vision_type: str = "butd"
    # Choices: "simple_cnn" | "simple_cnn_spatial" | "resnet" | "resnet_spatial"
    #          | "convnext" | "butd"
    output_size: int = 1024           # projected feature dim after encoder
    num_regions: int = 36             # spatial encoders: grid/ROI count
    roi_feat_dim: int = 2048          # BUTD: raw ROI-pooled dim from Faster R-CNN
    # G1: extended spatial geometry (5-dim → 7-dim)
    geo_dim: int = 5                  # set to 7 when ModelConfig.geo7=True

    # --- Question encoder ---
    q_vocab_size: int = 9937
    q_embed_size: int = 512           # word embedding projection dim
    q_hidden_size: int = 1024         # BiLSTM hidden (per direction = 512)
    q_num_layers: int = 2
    q_bidirectional: bool = True
    q_highway: bool = False           # highway connections between BiLSTM layers
    q_char_cnn: bool = False          # character-level CNN (300-dim output)
    char_embed_dim: int = 50
    char_num_filters: int = 100       # per kernel size; 3 kernel sizes → 300-dim total
    char_max_word_len: int = 20

    # --- GloVe ---
    glove_dim: int = 300              # 0 = no GloVe (random init)


# ---------------------------------------------------------------------------
# DecoderConfig
# ---------------------------------------------------------------------------

@dataclass
class DecoderConfig:
    a_vocab_size: int = 11271         # answer vocabulary size
    embed_size: int = 512             # token embedding dim (projected from GloVe if needed)
    hidden_size: int = 1024           # LSTM hidden dim
    num_layers: int = 2

    # Regularization
    dropout: float = 0.3
    dropconnect: float = 0.3          # 0.0 = disabled; applied to W_h in LSTM cell

    # Architecture flags (Model E/F baseline)
    use_layer_norm: bool = True       # LayerNorm per gate (Tier 1A)
    use_highway: bool = True          # highway connections between LSTM layers
    use_attention: bool = True        # dual MHCA (image + question)
    num_heads: int = 4                # MHCA heads; d_k = hidden_size // num_heads = 256
    use_coverage: bool = True         # coverage mechanism on image attention
    coverage_lambda: float = 0.5

    # Pointer-Generator
    use_pgn: bool = True              # 2-way PGN: vocab + question copy (Model F)
    use_pgn3: bool = False            # G2: 3-way PGN: + visual label copy

    # G5: length-conditioned decoding
    use_len_cond: bool = False
    len_embed_dim: int = 64           # dim of length-bin embedding
    min_decode_len: int = 8           # beam search: block <end> before this step


# ---------------------------------------------------------------------------
# FusionConfig
# ---------------------------------------------------------------------------

@dataclass
class FusionConfig:
    fusion_type: str = "mutan"        # "gated" | "mutan"
    hidden_size: int = 1024           # output dim
    # MUTAN-specific
    mutan_rank: int = 360             # projection dim for q and v before Tucker core
    mutan_out: int = 1024             # output dim (= hidden_size for decoder init)


# ---------------------------------------------------------------------------
# ModelConfig — top-level
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    model_type: str = "F"
    # Choices: "A" | "B" | "C" | "D" | "E" | "F" | "G"

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    # --- G-enhancement flags (all False = Model F baseline behavior) ---
    # G1: Extended 7-dim spatial geometry (+2K params)
    geo7: bool = False
    # G2: Three-way Pointer-Generator — copy from visual object labels (+14K params)
    pgn3: bool = False
    # G3: InfoNCE contrastive alignment loss (+524K params, discarded at inference)
    infonce: bool = False
    infonce_tau: float = 0.07
    infonce_beta: float = 0.1         # loss weight: L_total += beta * L_infonce
    infonce_proj_dim: int = 256       # projection head output dim
    # G4: Object Hallucination Penalty in SCST reward (0 params)
    ohp: bool = False
    ohp_lambda: float = 0.3           # reward weight: R -= lambda * OHP
    ohp_threshold: float = 0.5        # delta in max(0, delta - cos_sim)
    # G5: Length-conditioned decoding (+12K params)
    len_cond: bool = False            # mirrors decoder.use_len_cond — set together

    # -----------------------------------------------------------------------
    # Consistency enforcement
    # -----------------------------------------------------------------------

    def __post_init__(self):
        # Sync G-flags into sub-configs so sub-modules don't need to import ModelConfig
        if self.geo7:
            self.encoder.geo_dim = 7
        if self.pgn3:
            self.decoder.use_pgn3 = True
            self.decoder.use_pgn = True   # pgn3 is a superset
        if self.len_cond:
            self.decoder.use_len_cond = True

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    def to_json(self) -> dict:
        """Recursively convert to plain dict (JSON-serializable). Used in checkpoints."""
        return asdict(self)

    def to_json_str(self) -> str:
        return json.dumps(self.to_json(), indent=2)

    @classmethod
    def from_json(cls, d: dict) -> "ModelConfig":
        enc = EncoderConfig(**d.pop("encoder", {}))
        dec = DecoderConfig(**d.pop("decoder", {}))
        fus = FusionConfig(**d.pop("fusion", {}))
        return cls(encoder=enc, decoder=dec, fusion=fus, **d)

    @classmethod
    def from_json_str(cls, s: str) -> "ModelConfig":
        return cls.from_json(json.loads(s))

    # -----------------------------------------------------------------------
    # Argparse bridge (backward compat with existing train.py)
    # -----------------------------------------------------------------------

    @classmethod
    def from_args(cls, args) -> "ModelConfig":
        """
        Build ModelConfig from argparse.Namespace produced by the current train.py.
        Allows gradual migration: existing train.py still works unchanged.
        """
        model_type = getattr(args, "model", "F")

        # Resolve vision_type from model letter
        _vision = {
            "A": "simple_cnn",
            "B": "resnet",
            "C": "simple_cnn_spatial",
            "D": "resnet_spatial",
            "E": "convnext",
            "F": "butd",
            "G": "butd",
        }.get(model_type, "butd")

        enc = EncoderConfig(
            vision_type=_vision,
            q_vocab_size=getattr(args, "vocab_size", 9937),
            q_embed_size=getattr(args, "embed_size", 512),
            q_hidden_size=getattr(args, "hidden_size", 1024),
            q_num_layers=getattr(args, "num_layers", 2),
            q_highway=getattr(args, "q_highway", False),
            q_char_cnn=getattr(args, "char_cnn", False),
            glove_dim=getattr(args, "glove_dim", 300) if getattr(args, "glove", False) else 0,
            geo_dim=7 if getattr(args, "geo7", False) else 5,
        )

        dec = DecoderConfig(
            a_vocab_size=getattr(args, "answer_vocab_size", 11271),
            embed_size=getattr(args, "embed_size", 512),
            hidden_size=getattr(args, "hidden_size", 1024),
            num_layers=getattr(args, "num_layers", 2),
            dropout=getattr(args, "dropout", 0.3),
            use_layer_norm=getattr(args, "layer_norm", True),
            use_highway=getattr(args, "q_highway", False),
            use_attention=model_type in ("C", "D", "E", "F", "G"),
            use_coverage=getattr(args, "coverage", True),
            coverage_lambda=getattr(args, "coverage_lambda", 0.5),
            use_pgn=getattr(args, "pgn", False),
            use_pgn3=getattr(args, "pgn3", False),
            use_len_cond=getattr(args, "len_cond", False),
        )

        fus = FusionConfig(
            fusion_type="mutan" if getattr(args, "use_mutan", False) else "gated",
        )

        return cls(
            model_type=model_type,
            encoder=enc,
            decoder=dec,
            fusion=fus,
            geo7=getattr(args, "geo7", False),
            pgn3=getattr(args, "pgn3", False),
            infonce=getattr(args, "infonce", False),
            infonce_tau=getattr(args, "infonce_tau", 0.07),
            infonce_beta=getattr(args, "infonce_beta", 0.1),
            infonce_proj_dim=getattr(args, "infonce_proj_dim", 256),
            ohp=getattr(args, "ohp", False),
            ohp_lambda=getattr(args, "ohp_lambda", 0.3),
            ohp_threshold=getattr(args, "ohp_threshold", 0.5),
            len_cond=getattr(args, "len_cond", False),
        )

    # -----------------------------------------------------------------------
    # Preset factories
    # -----------------------------------------------------------------------

    @classmethod
    def model_g_full(cls) -> "ModelConfig":
        """Full Model G — all G1–G5 flags enabled."""
        return cls(
            model_type="G",
            encoder=EncoderConfig(
                vision_type="butd",
                geo_dim=7,
                q_highway=True,
                q_char_cnn=True,
                glove_dim=300,
            ),
            decoder=DecoderConfig(
                use_layer_norm=True,
                use_highway=True,
                use_attention=True,
                num_heads=4,
                use_coverage=True,
                use_pgn=True,
                use_pgn3=True,
                use_len_cond=True,
                dropout=0.3,
                dropconnect=0.3,
            ),
            fusion=FusionConfig(fusion_type="mutan"),
            geo7=True,
            pgn3=True,
            infonce=True,
            ohp=True,
            len_cond=True,
        )

    @classmethod
    def model_f_baseline(cls) -> "ModelConfig":
        """Model F baseline — no G-flags."""
        return cls(
            model_type="F",
            encoder=EncoderConfig(
                vision_type="butd",
                q_highway=True,
                q_char_cnn=True,
                glove_dim=300,
            ),
            decoder=DecoderConfig(
                use_layer_norm=True,
                use_highway=True,
                use_attention=True,
                use_coverage=True,
                use_pgn=True,
            ),
            fusion=FusionConfig(fusion_type="mutan"),
        )
