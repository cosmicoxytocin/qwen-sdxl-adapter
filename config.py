"""Configuration dataclasses for the QWEN-SDXL Adapter."""

import argparse
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AdapterConfig:
    """Configuration for the QWEN-SDXL Adapter."""

    source_dim: int = 1024
    target_dim: int = 2048
    pooled_dim: int = 1280
    num_queries: int = 77
    num_layers: int = 6
    num_heads: int = 16
    dropout: float = 0.1


@dataclass
class TrainConfig:
    """Configuration for training the QWEN-SDXL Adapter."""

    image_path: str = ""
    prompt: str = ""
    output_dir: str = "./output"
    num_steps: int = 1000
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    mixed_precision: str = "bf16"
    seed: int = 42
    save_every: int = 500

def parse_args() -> TrainConfig:
    """Parse command-line arguments for training the QWEN-SDXL Adapter."""
    parser = argparse.ArgumentParser(description="Smoke Test Trainer")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for training.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save outputs.")
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])

    args = parser.parse_args()
    return TrainConfig(**vars(args))
