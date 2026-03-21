# config package — dataclass-based configuration for Model G
from .model_config import ModelConfig, EncoderConfig, DecoderConfig, FusionConfig
from .train_config import TrainConfig, DataConfig, OptimizerConfig, PhaseConfig

__all__ = [
    "ModelConfig", "EncoderConfig", "DecoderConfig", "FusionConfig",
    "TrainConfig", "DataConfig", "OptimizerConfig", "PhaseConfig",
]
