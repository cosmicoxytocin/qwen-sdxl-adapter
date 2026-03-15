"""Configuration structures for the Qwen-SDXL Adapter. training pipeline."""

from dataclasses import dataclass, field
from typing import Optional, List
from omegaconf import OmegaConf


@dataclass
class ModelConfig:
    """Configuration for the Causal-to-Spatial Perceiver Bridge model."""

    qwen_model_id: str = "Qwen/Qwen3.5-0.8B-Base"
    sdxl_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    sdxl_single_file_ckpt: Optional[str] = None
    adapter_depth: int = 6
    adapter_dim: int = 1024
    adapter_heads: int = 16
    sdxl_context_dim: int = 2048
    sdxl_pooled_dim: int = 1280
    num_latent_queries: int = 78  # 77 Context + 1 Pooled


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    cache_dir: str = "./data/cache"
    image_resolution: int = 1024
    max_sequence_length: int = 256
    batch_size: int = 4
    num_workers: int = 4
    caption_dropout_prob: float = 0.1  # For Classifier-Free Guidance
    empty_prompt_token: str = ""  # NOTE: When applying 'Caption Dropout', the qwen tokenizer might output a single sequence consisting of
    # just the `<|endoftext|>` token, padding the rest of the 256 slots.
    # Ensure that the tokenizer pipeline enforces than an 'empty' string will resolve
    # to at least ONE valid token (the BOS or EOS token)
    # so the mask has at least one True value, giving the softmax a valid denominator.


@dataclass
class TrainingConfig:
    """Configuration for the training loop and optimizer."""

    output_dir: str = "./checkpoints"
    seed: int = 42
    mixed_precision: str = "bf16"  # ['no', 'fp16', 'bf16']
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    max_train_steps: int = 10000
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"  # TODO: ['linear', 'cosine', "cosine_annealing", 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']
    lr_warmup_steps: int = 500
    checkpointing_steps: int = 1000
    validation_steps: int = 500
    resume_from_checkpoint: Optional[str] = None  # e.g., "./checkpoints/step_5000"
    save_optimizer_state: bool = True


@dataclass
class LoggingConfig:
    """Configuration for Weights & Biases tracking."""

    project_name: str = "qwen-sdxl-adapter"
    run_name: Optional[str] = None
    log_interval: int = 10
    track_grad_norms: bool = True
    validation_prompts: List[str] = field(
        default_factory=lambda: [  # TODO: Add captions from the training subset for better validation insights
            "",
            "",
        ]
    )


@dataclass
class ExperimentConfig:
    """Master configuration object encompassing all sub-configurations."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(
    yaml_path: Optional[str] = None, cli_args: Optional[list[str]] = None
) -> ExperimentConfig:
    """Loads and merges configuration from defaults, YAML, and CLI."""
    config = OmegaConf.structured(ExperimentConfig)

    # Merge YAML config if provided
    if yaml_path:
        yaml_config = OmegaConf.load(yaml_path)
        config = OmegaConf.merge(config, yaml_config)

    # Merge CLI overrides if provided
    if cli_args:
        # Filter out empty strings and non-override arguments
        cli_args = [arg for arg in cli_args if "=" in arg]
        if cli_args:
            cli_config = OmegaConf.from_dotlist(cli_args)
            config = OmegaConf.merge(config, cli_config)

    # Resolve
    return OmegaConf.to_object(config)
