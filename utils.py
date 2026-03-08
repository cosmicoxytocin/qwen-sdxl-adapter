"""Shared utilities for logging, seeding, and checkpointing."""

import logging
import os
import os.path as osp
import random
import numpy as np
import torch
import logging  # noqa: F811

def set_seed(seed: int) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_logger(name: str) -> logging.Logger:
    """Initializes and returns a logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def save_checkpoint(model: torch.nn.Module, output_dir: str, step: int) -> None:
    """Saves a model checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    path = osp.join(output_dir, f"adapter_step_{step}.pt")
    # Unwrap compiled or DDP models if necessary
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(state_dict, path)
