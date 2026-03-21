"""Phase 2: Diff2Flow Compositional Fine-Tuning.

This script trains the Causal-to-Spatial Perceiver Bridge on image data
using Aspect Ratio Bucketed (ARB) pre-cached safetensors.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.config import load_config
from src.data.dataset import create_dataloader
from src.models.bridge import CausalToSpatialPerceiverBridge
from src.training.loss import Diff2FlowAlignmentLoss
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None, help="Path to phase 1 adapter weights"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print("1. Loading Frozen SDXL UNet to GPU...")
    unet = (
        UNet2DConditionModel.from_pretrained(
            config.model.sdxl_model_id, subfolder="unet", torch_dtype=dtype
        )
        .to(device)
        .eval()
    )
    unet.requires_grad_(False)

    if config.training.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    print("2. Loading Pre-Trained CSPB Adapter to GPU...")
    adapter = CausalToSpatialPerceiverBridge(
        depth=6,
        qwen_dim=1024,
        internal_dim=1024,
        sdxl_context_dim=2048,
        sdxl_pooled_dim=1280,
        num_queries=78,
    ).to(device, dtype=dtype)
    adapter.train()

    if args.resume_from_checkpoint:
        print(f"   -> Resuming adapter from: {args.resume_from_checkpoint}")
        from safetensors.torch import load_file

        adapter.load_state_dict(load_file(args.resume_from_checkpoint))

    print("3. Loading Validation Models (Qwen & VAE) to CPU (Zero VRAM footprint)...")
    tokenizer = AutoTokenizer.from_pretrained(config.model.qwen_model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    qwen = (
        AutoModelForCausalLM.from_pretrained(config.model.qwen_model_id, torch_dtype=dtype)
        .to("cpu")
        .eval()
    )
    vae = (
        AutoencoderKL.from_pretrained(config.model.sdxl_vae_id, torch_dtype=torch.float16)
        .to("cpu")
        .eval()
    )
    qwen.requires_grad_(False)
    vae.requires_grad_(False)

    print("4. Initializing Aspect Ratio Dataloader...")
    dataloader = create_dataloader(config.data)

    print("5. Initializing Diff2Flow Objective & Trainer...")
    noise_scheduler = DDPMScheduler.from_pretrained(
        config.model.sdxl_model_id, subfolder="scheduler"
    )
    objective = Diff2FlowAlignmentLoss(noise_scheduler, device)

    trainer = Trainer(
        unet=unet,
        adapter=adapter,
        qwen=qwen,
        vae=vae,
        tokenizer=tokenizer,
        dataloader=dataloader,
        objective=objective,
        config=config,
        device=device,
        dtype=dtype,
    )

    print("🚀 Starting Phase 2 Training...")
    trainer.train()


if __name__ == "__main__":
    main()
