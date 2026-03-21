"""
TrainConfig — typed dataclasses replacing loose training hyperparameters.

Hierarchy:
  TrainConfig
    ├── ModelConfig     (from model_config.py)
    ├── DataConfig      (paths, workers, augmentation)
    ├── OptimizerConfig (lr, scheduler, grad_clip — shared defaults)
    └── List[PhaseConfig]   (per-phase overrides: lr, epochs, data mix, scst…)

Usage:
  cfg = TrainConfig.model_g_default()          # 4-phase Model G
  cfg = TrainConfig.from_args(args)            # bridge from argparse
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from .model_config import ModelConfig


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    # Primary merged dataset (Model G)
    merged_json: str = "data/processed/merged_train_filtered.json"

    # VQA v2.0 files (Phase 1 mix + experience replay)
    vqa_v2_q_json: str = "data/annotations/vqa_v2/v2_OpenEnded_mscoco_train2014_questions.json"
    vqa_v2_a_json: str = "data/annotations/vqa_v2/v2_mscoco_train2014_annotations.json"
    vqa_v2_val_q_json: str = "data/annotations/vqa_v2/v2_OpenEnded_mscoco_val2014_questions.json"
    vqa_v2_val_a_json: str = "data/annotations/vqa_v2/v2_mscoco_val2014_annotations.json"

    # VQA-E val (evaluation)
    vqa_e_val_json: str = "data/annotations/vqa_e/VQA-E_val_set.json"

    # Image directory
    image_dir: str = "data/images"

    # BUTD features (Model F/G)
    butd_feat_dir: Optional[str] = None

    # Vocabulary paths (rebuilt for Model G)
    q_vocab_path: str = "data/processed/vocab_questions.json"
    a_vocab_path: str = "data/processed/vocab_answers.json"

    # DataLoader
    num_workers: int = 8
    pin_memory: bool = True

    # Sampling limits (for debugging)
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None

    # Augmentation (random horizontal flip, color jitter on images)
    augment: bool = True


# ---------------------------------------------------------------------------
# OptimizerConfig — shared defaults, overridden per phase
# ---------------------------------------------------------------------------

@dataclass
class OptimizerConfig:
    optimizer: str = "adamw"          # "adamw" | "adam" | "sgd"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 2.0
    warmup_epochs: int = 2
    scheduler: str = "cosine"         # "cosine" | "plateau" | "constant"
    # For ReduceLROnPlateau
    plateau_patience: int = 2
    plateau_factor: float = 0.5
    # CNN fine-tuning (Phase 2 for B/D/E/F/G)
    finetune_cnn: bool = False
    cnn_lr_factor: float = 0.1        # LR multiplier for CNN backbone params


# ---------------------------------------------------------------------------
# PhaseConfig — per-phase training settings
# ---------------------------------------------------------------------------

@dataclass
class PhaseConfig:
    phase: int = 1                    # 1 | 2 | 3 | 4

    # Epochs
    epochs: int = 15

    # Learning rate (overrides OptimizerConfig.lr for this phase)
    lr: float = 1e-3
    warmup_epochs: int = 2
    scheduler: str = "cosine"
    weight_decay: float = 1e-4

    # Batch
    batch_size: int = 192
    accum_steps: int = 1              # effective batch = batch_size * accum_steps
    grad_clip: float = 2.0

    # Regularization
    dropout: float = 0.3
    label_smoothing: float = 0.1

    # Loss
    focal: bool = True
    focal_gamma: float = 2.0

    # Data mix (fractions must sum to ≤ 1.0)
    mix_vqa: bool = False             # include VQA v2.0 in mix
    mix_vqa_fraction: float = 0.4    # fraction of batch from VQA v2.0 (Phase 1)
    replay_fraction: float = 0.0     # experience replay fraction (Phases 2-3: 0.2)

    # Scheduled sampling (Phase 3)
    scheduled_sampling: bool = False
    ss_k: float = 5.0                # decay constant for epsilon schedule

    # SCST RL (Phase 4)
    scst: bool = False
    scst_lambda: float = 0.5         # (1-lambda)*CE + lambda*SCST
    scst_bleu_weight: float = 0.5
    scst_meteor_weight: float = 0.5
    scst_cider_weight: float = 1.0
    scst_min_len: int = 8            # min decode len for SCST greedy baseline

    # CNN fine-tuning
    finetune_cnn: bool = False
    cnn_lr_factor: float = 0.1

    # Early stopping (0 = disabled)
    early_stopping: int = 0


# ---------------------------------------------------------------------------
# TrainConfig — top-level
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    phases: List[PhaseConfig] = field(default_factory=list)

    # Checkpoint
    resume: Optional[str] = None
    checkpoint_dir: str = "checkpoints"
    reset_best_val_loss: bool = False

    # W&B
    use_wandb: bool = False
    wandb_project: str = "vqa-model-g"
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None

    # Reproducibility
    seed: int = 42

    # Misc
    no_compile: bool = False          # disable torch.compile (debugging)

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    def to_json(self) -> dict:
        return asdict(self)

    def to_json_str(self) -> str:
        return json.dumps(self.to_json(), indent=2)

    @classmethod
    def from_json(cls, d: dict) -> "TrainConfig":
        model = ModelConfig.from_json(d.pop("model", {}))
        data = DataConfig(**d.pop("data", {}))
        opt = OptimizerConfig(**d.pop("optimizer", {}))
        phases = [PhaseConfig(**p) for p in d.pop("phases", [])]
        return cls(model=model, data=data, optimizer=opt, phases=phases, **d)

    # -----------------------------------------------------------------------
    # Argparse bridge
    # -----------------------------------------------------------------------

    @classmethod
    def from_args(cls, args) -> "TrainConfig":
        """Bridge from existing argparse.Namespace in train.py."""
        model_cfg = ModelConfig.from_args(args)

        data = DataConfig(
            image_dir=getattr(args, "image_dir", "data/images"),
            butd_feat_dir=getattr(args, "butd_feat_dir", None),
            num_workers=getattr(args, "num_workers", 8),
            augment=getattr(args, "augment", True),
            max_train_samples=getattr(args, "max_train_samples", None),
            max_val_samples=getattr(args, "max_val_samples", None),
        )

        opt = OptimizerConfig(
            lr=getattr(args, "lr", 1e-3),
            weight_decay=getattr(args, "weight_decay", 1e-4),
            grad_clip=getattr(args, "grad_clip", 2.0),
            warmup_epochs=getattr(args, "warmup_epochs", 2),
            finetune_cnn=getattr(args, "finetune_cnn", False),
            cnn_lr_factor=getattr(args, "cnn_lr_factor", 0.1),
        )

        # Build a single PhaseConfig from flat args (legacy single-phase call)
        phase = PhaseConfig(
            phase=getattr(args, "phase", 1) or 1,
            epochs=getattr(args, "epochs", 10),
            lr=getattr(args, "lr", 1e-3),
            batch_size=getattr(args, "batch_size", 192),
            accum_steps=getattr(args, "accum_steps", 1),
            grad_clip=getattr(args, "grad_clip", 2.0),
            dropout=getattr(args, "dropout", 0.3),
            label_smoothing=getattr(args, "label_smoothing", 0.1),
            focal=getattr(args, "focal", False),
            focal_gamma=getattr(args, "focal_gamma", 2.0),
            mix_vqa=getattr(args, "mix_vqa", False),
            mix_vqa_fraction=getattr(args, "mix_vqa_fraction", 0.4),
            scheduled_sampling=getattr(args, "scheduled_sampling", False),
            ss_k=getattr(args, "ss_k", 5.0),
            scst=getattr(args, "scst", False),
            scst_lambda=getattr(args, "scst_lambda", 0.5),
            scst_bleu_weight=getattr(args, "scst_bleu_weight", 0.5),
            scst_meteor_weight=getattr(args, "scst_meteor_weight", 0.5),
            finetune_cnn=getattr(args, "finetune_cnn", False),
            cnn_lr_factor=getattr(args, "cnn_lr_factor", 0.1),
            early_stopping=getattr(args, "early_stopping", 0),
        )

        return cls(
            model=model_cfg,
            data=data,
            optimizer=opt,
            phases=[phase],
            resume=getattr(args, "resume", None),
            checkpoint_dir="checkpoints",
            reset_best_val_loss=getattr(args, "reset_best_val_loss", False),
            use_wandb=getattr(args, "wandb", False),
            wandb_project=getattr(args, "wandb_project", "vqa-model-g"),
            wandb_run_name=getattr(args, "wandb_run_name", None),
            no_compile=getattr(args, "no_compile", False),
        )

    # -----------------------------------------------------------------------
    # Preset factories
    # -----------------------------------------------------------------------

    @classmethod
    def model_g_default(cls) -> "TrainConfig":
        """
        Full 4-phase Model G training config per Architecture_Specification_v2.md.
        35 epochs total: Phase1=15, Phase2=10, Phase3=7, Phase4=3.
        """
        model = ModelConfig.model_g_full()
        data = DataConfig(butd_feat_dir="data/butd_features")

        phases = [
            PhaseConfig(
                phase=1, epochs=15,
                lr=1e-3, warmup_epochs=2, scheduler="cosine",
                batch_size=192, grad_clip=2.0, weight_decay=1e-4,
                dropout=0.3, label_smoothing=0.1,
                focal=True, focal_gamma=2.0,
                mix_vqa=True, mix_vqa_fraction=0.4,
                replay_fraction=0.0,
                scheduled_sampling=False, scst=False,
                early_stopping=0,
            ),
            PhaseConfig(
                phase=2, epochs=10,
                lr=5e-4, warmup_epochs=0, scheduler="cosine",
                batch_size=192, grad_clip=2.0, weight_decay=1e-4,
                dropout=0.3, label_smoothing=0.1,
                focal=True, focal_gamma=2.0,
                mix_vqa=False, replay_fraction=0.2,
                scheduled_sampling=False, scst=False,
                early_stopping=3,
            ),
            PhaseConfig(
                phase=3, epochs=7,
                lr=2e-4, warmup_epochs=0, scheduler="plateau",
                batch_size=192, grad_clip=2.0, weight_decay=1e-4,
                dropout=0.3, label_smoothing=0.1,
                focal=True, focal_gamma=2.0,
                mix_vqa=False, replay_fraction=0.2,
                scheduled_sampling=True, ss_k=5.0,
                scst=False,
                early_stopping=3,
            ),
            PhaseConfig(
                phase=4, epochs=3,
                lr=5e-5, warmup_epochs=0, scheduler="constant",
                batch_size=64, grad_clip=2.0, weight_decay=1e-5,
                dropout=0.3, label_smoothing=0.0,
                focal=False,
                mix_vqa=False, replay_fraction=0.0,
                scheduled_sampling=False,
                scst=True, scst_lambda=0.5,
                scst_bleu_weight=0.5, scst_meteor_weight=0.5, scst_cider_weight=1.0,
                scst_min_len=8,
                early_stopping=0,
            ),
        ]

        return cls(
            model=model,
            data=data,
            phases=phases,
            checkpoint_dir="checkpoints",
            seed=42,
        )
