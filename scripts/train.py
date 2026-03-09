"""Main training entry point for the QWEN-SDXL-Adapter."""

import argparse
from models.bridge import CausalToSpatialPerceiverBridge
import random
import sys
from typing import List

import numpy as np
import torch
from diffusers import UNet2DConditionModel, DDPMScheduler

from src.config import load_config, ExperimentConfig
from src.data.dataset import create_dataloader
from src.training.loss import Diff2FlowAlignmentLoss
from src.training.trainer import AdapterTrainer
from src.utils.logger import WandbLogger

def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(config: ExperimentConfig) -> None:
    """Main training function."""
    # 1. Device & Reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.traning.seed)

    print(f"Initializing Qwen-SDXL Adapter Training on device: {device}")
    print(f"Mixed Precision: {config.training.mixed_precision}")

    # 2. Logger initialization
    logger = WandbLogger(config)

    # 3. Load frozen SDXL Unet and Noise Scheduler
    print(f"Loading frozen SDXL UNet from: {config.model.sdxl_model_id}...")

    # Load UNet directly into the the target precision to save RAM/VRAM during init
    unet_dtype = torch.bfloat16 if config.training.mixed_precision == "bf16" else torch.float16
    unet = UNet2DConditionModel.from_pretrained(
        config.model.sdxl_model_id,
        subfolder="unet",
        torch_dtype=unet_dtype,
    ).to(device)

    # Freeze UNet parameters
    unet.requires_grad_(False)
    unet.eval()

    # Load the scheduler to extract the beta schedule for Diff2Flow logic
    noise_scheduler = DDPMScheduler.from_pretrained(
        config.model.sdxl_model_id,
        subfolder="scheduler",
    )

    # 4. Instantiate trainable adapter model
    print("Initializing Causal-to-Spatial Perceiver Bridge...")
    adapter = CausalToSpatialPerceiverBridge(
        depth=config.model.adapter_depth,
        qwen_dim=1024,      # Fixed for Qwen3.5-0.8B-Base
        internal_dim=config.model.adapter_dim,
        sdxl_context_dim=config.model.sdxl_context_dim,
        sdxl_pooled_dim=config.model.sdxl_pooled_dim,
        num_queries=config.model.num_latent_queries
    ).to(device)

    # Ensure adapter is in training mode and requires gradients
    adapter.train()
    adapter.requires_grad_(True)

    # 5. Instantiate Diff2Flow Objective
    print("Setting up Diff2Flow Alignment Objective...")
    objective = Diff2FlowAlignmentLoss(
        noise_scheduler=noise_scheduler,
        device=device
    )

    # 6. Initialize DataLoader
    print(f"Loading cached dataset from {config.data.cache_dir}...")
    train_dataloader = create_dataloader(config.data)

    # 7. Initialize & Run Trainer
    print("Starting training loop...")
    trainer = AdapterTrainer(
        config=config,
        adapter=adapter,
        unet=unet,
        objective=objective,
        train_dataloader=train_dataloader,
        device=device,
    )
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model state...")
        trainer.save_checkpoint()
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise e
    finally:
        logger.finish()

def parse_args(args: List[str]) -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Train the Qwen-SDXL Adapter with Diff2Flow Alignment.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Override config parameters (e.g., training.learning_rate=2e-4)"
    )
    return parser.parse_args(args)

if __name__ == "__main__":
    # Parse command line arguments
    parsed_args = parse_args(sys.argv[1:])

    # Load and merge configuration (Defaults <- YAML <- CLI Overrides)
    experiment_config = load_config(
        yaml_path=parsed_args.config,
        cli_args=parsed_args.overrides
    )

    # Execute the training loop
    main(experiment_config)