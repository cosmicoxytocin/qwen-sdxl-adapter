"""Euler ODE Solver for Flow Matching trajectories.

Used for both runtime validation sampling and standalone inference.
"""

from typing import Tuple
import torch
from diffusers import DDPMScheduler


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

    def convert_fm_to_dm(self, fm_t: torch.Tensor, fm_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps Flow Matching (t, x) to Diffusion Model (t, x)."""
        right_index = torch.searchsorted(self.rectified_alphas_flipped, fm_t, right=True)
        left_index = right_index - 1
        right_value = self.rectified_alphas_flipped[right_index]
        left_value = self.rectified_alphas_flipped[left_index]

        dm_t_flipped = left_index.float() + (fm_t - left_value) / (right_value - left_value)
        dm_t = self.num_timesteps - dm_t_flipped

        scale = self.sqrt_alphas_cumprod_full + self.sqrt_one_minus_alphas_cumprod_full
        left_idx = torch.floor(dm_t).long().clamp(0, len(scale) - 1)
        right_idx = torch.ceil(dm_t).long().clamp(0, len(scale) - 1)
        scale_t = scale[left_idx] + (dm_t - left_idx.float()) * (scale[right_idx] - scale[left_idx])

        dm_x = fm_x * scale_t.view(-1, 1, 1, 1)
        return dm_t, dm_x

    def predict_velocity(self, dm_t: torch.Tensor, dm_x: torch.Tensor, eps_pred: torch.Tensor) -> torch.Tensor:
        """Converts DM epsilon prediction to FM velocity."""
        t_idx = dm_t.long().clamp(0, self.num_timesteps - 1)
        recip_alpha = self.sqrt_recip_alphas_cumprod[t_idx].view(-1, 1, 1, 1)
        recipm1_alpha = self.sqrt_recipm1_alphas_cumprod[t_idx].view(-1, 1, 1, 1)

        x1_pred = recip_alpha * dm_x - recipm1_alpha * eps_pred
        return x1_pred - eps_pred