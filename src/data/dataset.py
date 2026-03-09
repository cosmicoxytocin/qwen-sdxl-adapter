"""Dataset pipeline for QWEN-SDXL-Adapter training."""

import os
import glob
from typing import Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader

from ..config import DataConfig


class CachedAdapterDataset(Dataset):
    """Loads pre-cached VAE latents and LLM hidden states."""

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.files = glob.glob(os.path.join(config.cache_dir, "*.pt"))
        if len(self.files) == 0:
            raise ValueError(f"No cached .pt files found in {config.cache_dir}")
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_path = self.files[idx]
        data = torch.load(file_path, weights_only=True)

        # Classifier-Free Guidance (CFG) Dropout
        # Randomly replaces the conditional prompt with the unconditional prompt
        drop_caption = torch.rand(1).item() < self.config.caption_dropout_prob

        if drop_caption:
            hidden_states = data["uncond_hidden"]
            mask = data["uncond_mask"]
        else:
            hidden_states = data["cond_hidden"]
            mask = data["cond_mask"]
        
        return {
            "vae_latents": data["vae_latents"],
            "micro_conds": data["micro_conds"],
            "qwen_hidden_states": hidden_states,
            "qwen_mask": mask
        }

def create_dataloader(config: DataConfig) -> DataLoader:
    """Instantiates the DataLoader with optimal multithreading parameters."""
    dataset = CachedAdapterDataset(config)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
