"""Dataset pipeline for Qwen-SDXL Adapter Training."""

import os
import json
import random
from typing import Dict, List, Iterator

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from safetensors.torch import load_file

from ..config import DataConfig


class CachedAdapterDataset(Dataset):
    """Loads pre-cached VAE latents and LLM hidden states via safetensors for efficient training."""

    def __init__(self, config: DataConfig):
        self.config = config
        meta_path = os.path.join(config.cache_dir, "meta.json")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Metadata index not found at {meta_path}. Please run the caching script first."
            )

        with open(meta_path, "r") as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item_meta = self.metadata[idx]
        file_path = item_meta["path"]

        data = load_file(file_path)

        # CFG dropout logic
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
            "qwen_mask": mask,
        }


class AspectRatioBatchSampler(Sampler[List[int]]):
    """
    Groups dataset indices by their aspect ratio buckets and yields
    batches that are guaranteed to have identical latent spatial dimensions.
    """

    def __init__(
        self, dataset: CachedAdapterDataset, batch_size: int, drop_last: bool = True
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Group indices by bucket string (e.g., "1024x1024": [0, 5, 12...])
        self.buckets: Dict[str, List[int]] = {}
        for idx, item in enumerate(self.dataset.metadata):
            bucket_key = item["bucket"]
            if bucket_key not in self.buckets:
                self.buckets[bucket_key] = []
            self.buckets[bucket_key].append(idx)

    def __iter__(self) -> Iterator[List[int]]:
        batches = []

        # Create batches for each bucket individually
        for bucket_key, indices in self.buckets.items():
            random.shuffle(indices)  # Shuffle images within the bucket

            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        # Shuffle the order of the batches so the model doesn't overfit to one aspect ratio at a time
        random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        count = 0
        for indices in self.buckets.values():
            if self.drop_last:
                count += len(indices) // self.batch_size
            else:
                count += (len(indices) + self.batch_size - 1) // self.batch_size
        return count


def create_dataloader(config: DataConfig) -> DataLoader:
    """Instantiates the DataLoader utilizing the Aspect Ratio Batch Sampler."""
    dataset = CachedAdapterDataset(config)

    drop_last = len(dataset) >= config.batch_size

    # Because we are using a custom batch sampler, we cannot use shuffle=True or batch_size
    # in the main DataLoader kwargs. The sampler handles both.
    batch_sampler = AspectRatioBatchSampler(
        dataset=dataset, batch_size=config.batch_size, drop_last=drop_last
    )

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
    )
