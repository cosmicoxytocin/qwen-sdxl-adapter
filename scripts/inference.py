"""Inference script for the Qwen-SDXL Adapter."""

import argparse
import os
from typing import Tuple
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file

from src.models.bridge import CausalToSpatialPerceiverBridge
from src.config import ModelConfig
from src.models.sampler import Diff2FlowEulerSampler


@torch.no_grad()
def generate_image(
    prompt: str,
    adapter_ckpt_path: str,
    output_path: str,
    num_inference_steps: int = 20,
    guidance_scale: float = 4.5,
    seed: int = 42,
    device: str = "cuda",
) -> None:
    """Executes the full inference pipeline."""
    torch.manual_seed(seed)
    dtype = torch.bfloat16
    config = ModelConfig()

    print("1. Loading frozen models (Qwen, UNet, VAE)...")
    # LLM
    tokenizer = AutoTokenizer.from_pretrained(config.qwen_model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    qwen = AutoModelForCausalLM.from_pretrained(config.qwen_model_id, torch_dtype=dtype).to(device)
    qwen.eval()

    # VAE & UNet
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype).to(
        device
    )
    vae.eval()

    if getattr(config, "sdxl_single_file_ckpt", None):
        print(f"Loading SDXL UNet from: {config.sdxl_single_file_ckpt}...")
        unet = UNet2DConditionModel.from_single_file(
            config.sdxl_single_file_ckpt, torch_dtype=dtype
        ).to(device)
    else:
        print("Loading SDXL UNet from Hugging Face...")
        unet = UNet2DConditionModel.from_pretrained(
            config.sdxl_model_id, subfolder="unet", torch_dtype=dtype
        ).to(device)
    unet.eval()

    noise_scheduler = DDPMScheduler.from_pretrained(config.sdxl_model_id, subfolder="scheduler")
    sampler = Diff2FlowEulerSampler(noise_scheduler, torch.device(device))

    print(f"2. Loading trained CSPB Adapter from {adapter_ckpt_path}...")
    adapter = CausalToSpatialPerceiverBridge(
        depth=config.adapter_depth,
        qwen_dim=config.adapter_dim,
        internal_dim=config.adapter_dim,
        sdxl_context_dim=config.sdxl_context_dim,
        sdxl_pooled_dim=config.sdxl_pooled_dim,
        num_queries=config.num_latent_queries,
    ).to(device, dtype=dtype)

    adapter.load_state_dict(load_file(adapter_ckpt_path))
    adapter.eval()

    print(f"3. Processing prompt: '{prompt}'")
    # Tokenize conditional and unconditional
    prompts = ["", prompt]
    encoded = tokenizer(
        prompts, padding="max_length", truncation=True, max_length=256, return_tensors="pt"
    ).to(device)

    # Get Qwen Hidden States
    qwen_outputs = qwen.model(
        input_ids=encoded.input_ids, attention_mask=encoded.attention_mask, return_dict=True
    )
    hidden_states = qwen_outputs.last_hidden_state.to(dtype)
    mask = encoded.attention_mask.bool()

    # Pass through Adapter
    adapter_ctx, adapter_pooled = adapter(hidden_states, mask)

    # SDXL Micro-Conditioning: [original_h, original_w, crop_y, crop_x, target_h, target_w]
    micro_conds = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]] * 2, dtype=dtype, device=device)
    added_cond_kwargs = {"text_embeds": adapter_pooled, "time_ids": micro_conds}

    print("4. Executing Diff2Flow Euler Sampling...")
    # Initialize noise (Flow Matching starts at noise t=0)
    x = torch.randn((1, 4, 128, 128), dtype=dtype, device=device)

    # Step size
    dt = 1.0 / num_inference_steps

    for i in tqdm(range(num_inference_steps), desc="Sampling"):
        # Current time t in [0, 1]
        fm_t = torch.tensor([i * dt], dtype=torch.float32, device=device)

        # Expand x and t for CFG batching
        x_in = torch.cat([x, x], dim=0)
        fm_t_in = torch.cat([fm_t, fm_t], dim=0)

        # Convert to DM variables for UNet (Using public method now)
        dm_t, dm_x = sampler.convert_fm_to_dm(fm_t_in, x_in)

        # Predict epsilon
        eps_pred = unet(
            dm_x.to(dtype),
            dm_t.to(dtype),
            encoder_hidden_states=adapter_ctx,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        # Convert epsilon to velocity (Using public method now)
        v_pred = sampler.predict_velocity(dm_t, dm_x, eps_pred)

        # Classifier-Free Guidance (CFG)
        v_uncond, v_cond = v_pred.chunk(2)
        v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)

        # Euler Step (straight line)
        x = x + v_cfg.to(dtype) * dt

    print("5. Decoding Latents via VAE...")
    # Scale by VAE scaling factor
    x = x / vae.config.scaling_factor
    image_tensor = vae.decode(x).sample

    # Convert to PIL Image
    image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
    image_numpy = image_tensor[0].cpu().permute(1, 2, 0).float().numpy()

    # Catch any NaN/Inf from bloat16 VAE decoding
    import numpy as np
    image_numpy = np.nan_to_num(image_numpy, nan=0.0, posinf=1.0, neginf=0.0)
    
    img = Image.fromarray((image_numpy * 255).astype("uint8"))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    img.save(output_path)
    print(f"Success! Image saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diff2Flow Inference for Qwen-SDXL Adapter")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt (supports tag-soup & natural language)",
    )
    parser.add_argument(
        "--adapter_ckpt", type=str, required=True, help="Path to trained adapter .safetensors file"
    )
    parser.add_argument(
        "--output", type=str, default="./output.png", help="Path to save generated image"
    )
    parser.add_argument("--steps", type=int, default=20, help="Number of Euler sampling steps")
    parser.add_argument("--cfg", type=float, default=4.5, help="Classifier-Free Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    generate_image(
        prompt=args.prompt,
        adapter_ckpt_path=args.adapter_ckpt,
        output_path=args.output,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        seed=args.seed,
    )
