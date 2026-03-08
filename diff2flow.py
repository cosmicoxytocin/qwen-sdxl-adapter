"""Diff2Flow trajectory conversions for Flow Matching (FT)."""

import torch


class Diff2FlowConverter:
    """Handles time and latent conversions between Flow Matching and Diffusion."""

    def __init__(self, betas: torch.Tensor, num_train_timesteps: int, device: torch.device):
        self.device = device
        self.num_train_timesteps = num_train_timesteps