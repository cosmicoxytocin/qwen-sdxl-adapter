"""Euler ODE Solver for Flow Matching trajectories."""

from typing import Tuple
import torch
from diffusers import DDPMScheduler


class Diff2FlowEulerSampler:
    """Euler ODE Solver for Flow Matching trajectories aligned to Diffusion Models."""

    def __init__(self, noise_scheduler: DDPMScheduler, device: torch.device) -> None:
        """Precomputes the Diff2Flow alignment schedules."""
        self.device = device
        self.num_timesteps = noise_scheduler.config.num_train_timesteps

        # 1. Extract and ENFORCE ZERO-TERMINAL SNR on Betas
        betas = noise_scheduler.betas.to(dtype=torch.float32, device=device)
        betas = self._enforce_zero_terminal_snr(betas)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_full = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod])

        self.sqrt_alphas_cumprod_full = torch.sqrt(alphas_cumprod_full)
        self.sqrt_one_minus_alphas_cumprod_full = torch.sqrt(1.0 - alphas_cumprod_full)

        alphas_cumprod_clamped = alphas_cumprod.clamp(min=1e-7)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod_clamped)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod_clamped - 1.0)

        self.rectified_alphas = self.sqrt_alphas_cumprod_full / (
            self.sqrt_alphas_cumprod_full + self.sqrt_one_minus_alphas_cumprod_full
        )
        self.rectified_alphas_flipped = torch.flip(self.rectified_alphas, dims=[0])

    def _enforce_zero_terminal_snr(self, betas: torch.Tensor) -> torch.Tensor:
        """Rescales the beta schedule to ensure mathematically pure noise at T."""
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_sqrt = torch.sqrt(alphas_bar)

        # Store old limits
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

        # Shift so last timestep is exactly zero
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        # Scale so first timestep remains unchanged
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        alphas_bar = alphas_bar_sqrt**2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])

        return 1.0 - alphas

    def convert_fm_to_dm(
        self, fm_t: torch.Tensor, fm_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def predict_velocity(
        self, dm_t: torch.Tensor, dm_x: torch.Tensor, eps_pred: torch.Tensor
    ) -> torch.Tensor:
        """Converts DM epsilon prediction to FM velocity."""
        t_idx = dm_t.long().clamp(0, self.num_timesteps - 1)
        recip_alpha = self.sqrt_recip_alphas_cumprod[t_idx].view(-1, 1, 1, 1)
        recipm1_alpha = self.sqrt_recipm1_alphas_cumprod[t_idx].view(-1, 1, 1, 1)

        x1_pred = recip_alpha * dm_x - recipm1_alpha * eps_pred
        v_pred = x1_pred - eps_pred

        # Bounding safeguard: replaces any NaN or Inf with zero velocity
        v_pred = torch.nan_to_num(v_pred, nan=0.0, posinf=0.0, neginf=0.0)
        return v_pred
