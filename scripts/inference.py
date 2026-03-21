"""Stateful Inference script for the Qwen-SDXL Adapter."""

import argparse
import os
from typing import Tuple
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from PIL import Image
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file

from src.models.bridge import CausalToSpatialPerceiverBridge
from src.config import ModelConfig


class Diff2FlowEulerSampler:
    """Euler ODE Solver for Flow Matching trajectories aligned to Diffusion Models."""

    def __init__(self, noise_scheduler: DDPMScheduler, device: torch.device) -> None:
        self.device = device
        self.num_timesteps = noise_scheduler.config.num_train_timesteps

        betas = noise_scheduler.betas.to(dtype=torch.float32, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_full = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod])

        self.sqrt_alphas_cumprod_full = torch.sqrt(alphas_cumprod_full)
        self.sqrt_one_minus_alphas_cumprod_full = torch.sqrt(1.0 - alphas_cumprod_full)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)

        self.rectified_alphas = self.sqrt_alphas_cumprod_full / (
            self.sqrt_alphas_cumprod_full + self.sqrt_one_minus_alphas_cumprod_full
        )
        self.rectified_alphas_flipped = torch.flip(self.rectified_alphas, dims=[0])

    def _convert_fm_to_dm(
        self, fm_t: torch.Tensor, fm_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps Flow Matching (t, x) to Diffusion Model (t, x)."""
        # FIX: The critical clamp that prevents Time-Reversal!
        right_index = torch.searchsorted(self.rectified_alphas_flipped, fm_t, right=True)
        max_idx = len(self.rectified_alphas_flipped) - 1
        right_index = torch.clamp(right_index, 1, max_idx)
        left_index = right_index - 1

        right_value = self.rectified_alphas_flipped[right_index]
        left_value = self.rectified_alphas_flipped[left_index]

        dm_t_flipped = left_index.float() + (fm_t - left_value) / (right_value - left_value)
        dm_t = self.num_timesteps - dm_t_flipped

        # Spatial scaling
        scale = self.sqrt_alphas_cumprod_full + self.sqrt_one_minus_alphas_cumprod_full
        left_idx = torch.floor(dm_t).long().clamp(0, len(scale) - 1)
        right_idx = torch.ceil(dm_t).long().clamp(0, len(scale) - 1)
        scale_t = scale[left_idx] + (dm_t - left_idx.float()) * (scale[right_idx] - scale[left_idx])

        dm_x = fm_x * scale_t.view(-1, 1, 1, 1)
        return dm_t, dm_x

    def _predict_velocity(
        self, dm_t: torch.Tensor, dm_x: torch.Tensor, eps_pred: torch.Tensor
    ) -> torch.Tensor:
        """Converts DM epsilon prediction to FM velocity."""
        t_idx = dm_t.long().clamp(0, self.num_timesteps - 1)
        recip_alpha = self.sqrt_recip_alphas_cumprod[t_idx].view(-1, 1, 1, 1)
        recipm1_alpha = self.sqrt_recipm1_alphas_cumprod[t_idx].view(-1, 1, 1, 1)

        x1_pred = recip_alpha * dm_x - recipm1_alpha * eps_pred
        return x1_pred - eps_pred


class QwenSDXLPipeline:
    """Stateful inference pipeline. Loads models once to save VRAM."""

    def __init__(self, adapter_ckpt_path: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        config = ModelConfig()

        print("1. Loading frozen models (Qwen, UNet, VAE)...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.qwen_model_id, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.qwen = (
            AutoModelForCausalLM.from_pretrained(config.qwen_model_id, torch_dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        self.vae = (
            AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        self.unet = (
            UNet2DConditionModel.from_pretrained(
                config.sdxl_model_id, subfolder="unet", torch_dtype=self.dtype
            )
            .to(self.device)
            .eval()
        )

        noise_scheduler = DDPMScheduler.from_pretrained(config.sdxl_model_id, subfolder="scheduler")
        self.sampler = Diff2FlowEulerSampler(noise_scheduler, self.device)

        print(f"2. Loading trained CSPB Adapter from {adapter_ckpt_path}...")
        self.adapter = CausalToSpatialPerceiverBridge(
            depth=config.adapter_depth,
            qwen_dim=config.adapter_dim,
            internal_dim=config.adapter_dim,
            sdxl_context_dim=config.sdxl_context_dim,
            sdxl_pooled_dim=config.sdxl_pooled_dim,
            num_queries=config.num_latent_queries,
        ).to(self.device, dtype=self.dtype)

        self.adapter.load_state_dict(load_file(adapter_ckpt_path))
        self.adapter.eval()
        print("Pipeline initialized and ready.")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 25,
        guidance_scale: float = 4.5,
        seed: int = 42,
    ) -> Image.Image:

        torch.manual_seed(seed)

        prompts = [negative_prompt, prompt]
        encoded = self.tokenizer(
            prompts, padding="max_length", truncation=True, max_length=256, return_tensors="pt"
        ).to(self.device)

        qwen_outputs = self.qwen.model(
            input_ids=encoded.input_ids, attention_mask=encoded.attention_mask, return_dict=True
        )
        hidden_states = qwen_outputs.last_hidden_state.to(self.dtype)
        mask = encoded.attention_mask.bool()

        adapter_ctx, adapter_pooled = self.adapter(hidden_states, mask)

        micro_conds = torch.tensor(
            [[1024, 1024, 0, 0, 1024, 1024]] * 2, dtype=self.dtype, device=self.device
        )
        added_cond_kwargs = {"text_embeds": adapter_pooled, "time_ids": micro_conds}

        x = torch.randn((1, 4, 128, 128), dtype=self.dtype, device=self.device)
        dt = 1.0 / num_inference_steps

        for i in tqdm(range(num_inference_steps), desc="Sampling"):
            fm_t = torch.tensor([i * dt], dtype=torch.float32, device=self.device)

            x_in = torch.cat([x, x], dim=0)
            fm_t_in = torch.cat([fm_t, fm_t], dim=0)

            dm_t, dm_x = self.sampler._convert_fm_to_dm(fm_t_in, x_in)

            eps_pred = self.unet(
                dm_x.to(self.dtype),
                dm_t.to(self.dtype),
                encoder_hidden_states=adapter_ctx,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

            v_pred = self.sampler._predict_velocity(dm_t, dm_x, eps_pred)

            v_uncond, v_cond = v_pred.chunk(2)
            v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)

            x = x + v_cfg.to(self.dtype) * dt

        x = x / self.vae.config.scaling_factor
        image_tensor = self.vae.decode(x).sample

        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
        image_numpy = image_tensor[0].cpu().permute(1, 2, 0).float().numpy()
        return Image.fromarray((image_numpy * 255).astype("uint8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--adapter_ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, default="./output.png")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pipe = QwenSDXLPipeline(args.adapter_ckpt)
    img = pipe.generate(
        args.prompt, num_inference_steps=args.steps, guidance_scale=args.cfg, seed=args.seed
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    img.save(args.output)
    print(f"Saved to {args.output}")
