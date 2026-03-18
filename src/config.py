"""Configuration definitions for Qwen-SDXL Adapter."""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from omegaconf import OmegaConf


@dataclass
class ModelConfig:
    sdxl_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    qwen_model_id: str = "Qwen/Qwen3.5-0.8B-Base"
    # FIX: Added the VAE ID for Phase 2 CPU/GPU juggling
    sdxl_vae_id: str = "madebyollin/sdxl-vae-fp16-fix"


@dataclass
class DataConfig:
    # FIX: Updated to point to our new Aspect Ratio Bucket cache
    cache_dir: str = "./data/arb_cache"
    batch_size: int = 4
    num_workers: int = 4
    caption_dropout_prob: float = 0.1


@dataclass
class TrainingConfig:
    output_dir: str = "./checkpoints"
    max_train_steps: int = 100000
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    validation_steps: int = 500
    checkpointing_steps: int = 500
    validation_prompts: List[str] = field(
        default_factory=lambda: [""]
    )


@dataclass
class LoggingConfig:
    use_wandb: bool = True
    wandb_project: str = "qwen-sdxl-adapter"
    log_interval: int = 10
    track_grad_norms: bool = True


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def dict(self):
        """Convert to standard dictionary for WandB logging."""
        return OmegaConf.to_container(self, resolve=True)


def load_config(config_path: Optional[str] = None) -> ExperimentConfig:
    """Loads and merges the YAML config into the structured dataclass."""
    base_config = OmegaConf.structured(ExperimentConfig)

    if config_path and os.path.exists(config_path):
        yaml_config = OmegaConf.load(config_path)
        # This is where the error triggered! Now it will merge perfectly.
        config = OmegaConf.merge(base_config, yaml_config)
        return config

    return base_config
