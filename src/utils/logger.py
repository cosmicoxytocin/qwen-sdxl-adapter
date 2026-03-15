"""Weights and Biases (wandb) logger utility."""

from typing import Dict, Any, Optional

import wandb
import torch

from ..config import ExperimentConfig


class WandbLogger:
    """Handles experiment tracking via Weights and Biases (wandb)."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.project_name = config.logging.project_name

        # Flatten config for Wandb
        flattened_config = {
            "model": config.model.__dict__,
            "training": config.training.__dict__,
            "data": config.data.__dict__,
        }

        wandb.init(
            project=self.project_name,
            name=config.logging.run_name,
            config=flattened_config,
            reinit=True,
        )

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Logs a dictionary of metrics to Wandb at a given step."""
        wandb.log(metrics, step=step)

    def log_image(self, tag: str, image: torch.Tensor, step: int, caption: str = "") -> None:
        """Logs a generated validation image."""
        # Convert [-1, 1] tensor to [0, 1] for Wandb
        img_normalized = (image + 1.0) / 2.0
        img_clamped = torch.clamp(img_normalized, 0.0, 1.0)

        # Expects [C, H, W]
        wandb_img = wandb.Image(img_clamped, caption=caption)
        wandb.log({tag: wandb_img}, step=step)

    def finish(self) -> None:
        """Close the Wandb run."""
        wandb.finish()
