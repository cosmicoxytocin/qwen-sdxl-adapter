"""Diff2Flow Optimal Transport Loss Objective for SDXL."""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler


class Diff2FlowAlignmentLoss(nn.Module):
    """Calculates the Flow Matching velocity loss using a frozen DM UNet."""

    def __init__(self, noise_scheduler: DDPMScheduler, device: torch.device) -> None:
        super().__init__()
        self.device = device

        # Extract base scheduler parameters
        betas = noise_scheduler.betas.to(dtype=torch.float32, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Append 1.0 for t=0 (full data, no noise)
        alphas_cumprod_full = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod])
        
        self.num_timesteps = len(betas)

        # Precompute square roots for velocity calculation
        self.sqrt_alphas_cumprod_full = torch.sqrt(alphas_cumprod_full)
        self.sqrt_one_minus_alphas_cumprod_full = torch.sqrt(1.0 - alphas_cumprod_full)

        # For eps to x0 prediction (Diffusion space)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)

        # Precompute Rectified alphas for time mapping
        self.rectified_alphas = self.sqrt_alphas_cumprod_full / (
            self.sqrt_alphas_cumprod_full + self.sqrt_one_minus_alphas_cumprod_full
        )

        # Flipped for torch searchsorted (requires monotonically increasing)
        self.rectified_alphas_flipped = torch.flip(self.rectified_alphas, dims=[0])

    def _convert_fm_t_to_dm_t(self, fm_t: torch.Tensor) -> torch.Tensor:
        """Converts Flow Matching time [0, 1] to Diffusion time [0, 1000]."""
        right_index = torch.searchsorted(self.rectified_alphas_flipped, fm_t, right=True)
        left_index = right_index - 1

        right_value = self.rectified_alphas_flipped[right_index]
        left_value = self.rectified_alphas_flipped[left_index]

        # Linear interpolation
        dm_t_flipped = left_index.float() + (fm_t - left_value) / (right_value - left_value)

        # Un-flip the index
        dm_t = self.num_timesteps - dm_t_flipped
        return dm_t
    
    def _convert_fm_xt_to_dm_xt(self, fm_xt: torch.Tensor, dm_t: torch.Tensor) -> torch.Tensor:
        """Scales Flow Matching latent to Diffusion model latent scale."""
        scale = self.sqrt_alphas_cumprod_full + self.sqrt_one_minus_alphas_cumprod_full

        left_idx = torch.floor(dm_t).long().clamp(0, len(scale) - 1)
        right_idx = torch.ceil(dm_t).long().clamp(0, len(scale) - 1)

        left_val = scale[left_idx]
        right_val = scale[right_idx]

        scale_t = left_val + (dm_t - left_idx.float()) * (right_val - left_val)
        return fm_xt * scale_t.view(-1, 1, 1, 1)
    
    def _predict_x1_from_eps(self, dm_xt: torch.Tensor, dm_t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Predicts the original data (x1) from the UNet's epsilon prediction."""
        t_idx = dm_t.long().clamp(0, self.num_timesteps - 1)

        recip_alpha = self.sqrt_recip_alphas_cumprod[t_idx].view(-1, 1, 1, 1)
        recipm1_alpha = self.sqrt_recipm1_alphas_cumprod[t_idx].view(-1, 1, 1, 1)

        return recip_alpha * dm_xt - recipm1_alpha * eps
    
    def forward(
        self,
        unet: nn.Module,
        adapter_ctx: torch.Tensor,
        adapter_pooled: torch.Tensor,
        x1: torch.Tensor,
        micro_conds: torch.Tensor
    ) -> torch.Tensor:
        """Executes the forward pass to compute the Flow Matching alignment calculation.
        
        Args:
            unet: The frozen SDXL UNet.
            adapter_ctx: Translated context embeddings [B, 77, 2048].
            adapter_pooled: Translated pooled embeddings [B, 1280].
            x1: The target VAE latent (data) [B, 4, 128, 128].
            micro_conds: SDXL added time_ids [B, 6].
        
        Returns:
            The scalar MSE loss representing the Flow Matching velocity error.
        """
        b, c, h, w = x1.shape
        device, dtype = x1.device, x1.dtype

        # 1. Sample Noise and Time (Flow Matching space)
        x0 = torch.randn_like(x1)
        fm_t = torch.rand((b,), device=device, dtype=torch.float32)

        # 2. Construct Flow Matching trajectory
        fm_t_exp = fm_t.view(-1, 1, 1, 1).to(dtype)
        fm_xt = fm_t_exp * x1 + (1.0 - fm_t_exp) * x0

        # Target Velocity: Data (x1) - Noise (x0)
        v_target = x1 - x0

        # 3. Align to diffusion space
        dm_t = self._convert_fm_t_to_dm_t(fm_t)
        dm_xt = self._convert_fm_xt_to_dm_xt(fm_xt, dm_t).to(dtype)

        added_cond_kwargs = {
            "text_embeds": adapter_pooled,
            "time_ids": micro_conds
        }

        # 4. Predict Epsilon using the frozen UNet coditioned by the adapter
        # NOTE: Don't track gradients for the UNet, only the Adapter context
        eps_pred = unet(
            dm_xt,
            dm_t.to(dtype),
            encoder_hidden_states=adapter_ctx,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        # 5. Reconstruct Velocity from epsilon
        x1_pred = self._predict_x1_from_eps(dm_xt, dm_t, eps_pred).to(dtype)
        v_pred = x1_pred - eps_pred

        # 6. Loss calculation
        loss = F.mse_loss(v_pred.float(), v_target.float(), reduction="mean")
        return loss
