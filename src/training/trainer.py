"""Training loop for Qwen-SDXL Adapter."""

import os
from typing import Dict, Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import wandb
from tqdm import tqdm

from ..models.bridge import CausalToSpatialPerceiverBridge
from .loss import Diff2FlowAlignmentLoss
from ..config import ExperimentConfig


class AdapterTrainer:
    """Manages the lifecycle of the adapter training process."""

    def __init__(
        self,
        config: ExperimentConfig,
        adapter: CausalToSpatialPerceiverBridge,
        unet: nn.Module,
        objective: Diff2FlowAlignmentLoss,
        train_dataloader: DataLoader,
        device: torch.device
    ) -> None:
        self.config = config
        self.device = device
        self.train_dataloader = train_dataloader

        # Modules
        self.adapter = adapter.to(device)
        self.unet = unet.to(device)
        self.objective = objective.to(device)

        # Optimizer Setup
        self.optimizer = torch.optim.AdamW(
            self.adapter.parameters(),
            lr=config.training.learning_rate,
            betas=config.training.betas,
            weight_decay=config.training.weight_decay,
        )

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=config.training.lr_warmup_steps,
            num_training_steps=config.training.max_train_steps,
        )

        # Mixed Precision Setup
        self.use_amp = config.training.mixed_precision != "no"
        self.dtype = torch.bfloat16 if config.training.mixed_precision == "bf16" else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == torch.float16))

        # State
        self.global_step = 0
        self.epoch = 0
    
    def save_checkpoint(self) -> None:
        """Save the adapter weights using Safetensors."""
        from safetensors.torch import save_file

        os.makedirs(self.config.training.output_dir, exist_ok=True)
        ckpt_path = os.path.join(
            self.config.training.output_dir,
            f"adapter_step_{self.global_step}.safetensors"
        )
        # Only save adapter weights for efficiency
        save_file(self.adapter.state_dict(), ckpt_path)
        print(f"Checkpoint saved at: {ckpt_path}")
    
    def train(self) -> None:
        """Main training loop."""
        print(f"Starting training on {self.device} for {self.config.training.max_train_steps} steps...")

        if len(self.train_dataloader) == 0:
            raise ValueError(
                f"DataLoader is empty! Dataset has {len(self.train_dataloader.dataset)} samples "
                f"but batch_size={self.train_dataloader.batch_size} with drop_last=True. "
                f"Add more data or reduce batch_size."
            )

        self.adapter.train()
        self.unet.eval()  # UNet is frozen

        progress_bar = tqdm(total=self.config.training.max_train_steps, desc="Training")
        micro_step = 0

        while self.global_step < self.config.training.max_train_steps:
            for batch in self.train_dataloader:
                # 1. Fetch AoT Cached Data
                x1 = batch["vae_latents"].to(self.device, non_blocking=True)
                qwen_hidden = batch["qwen_hidden_states"].to(self.device, non_blocking=True)
                qwen_mask = batch["qwen_mask"].to(self.device, non_blocking=True)
                micro_conds = batch["micro_conds"].to(self.device, non_blocking=True)

                # 2. Forward Pass with AMP
                with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
                    # Adapter translates unaligned Qwen Space to topological SDXL space
                    adapter_ctx, adapter_pooled = self.adapter(qwen_hidden, qwen_mask)

                    # Compute Velocity Loss via frozen UNet
                    loss = self.objective(
                        unet=self.unet,
                        adapter_ctx=adapter_ctx,
                        adapter_pooled=adapter_pooled,
                        x1=x1,
                        micro_conds=micro_conds
                    )

                    # Scale loss for gradient accumulation
                    loss = loss / self.config.training.gradient_accumulation_steps

                    # 3. Backpropagation
                    self.scaler.scale(loss).backward()

                    micro_step += 1

                    # 4. Optimizer Step (Applying accumulated gradients)
                    if micro_step % self.config.training.gradient_accumulation_steps == 0:
                        # Unscale gradients for clipping
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.adapter.parameters(),
                            self.config.training.max_grad_norm
                        )

                        # Step optimizer and scheduler
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)

                        self.global_step += 1
                        progress_bar.update(1)

                        # 5. Logging
                        if self.global_step % self.config.logging.log_interval == 0:
                            logs = {
                                "train/loss": loss.item() * self.config.training.gradient_accumulation_steps,
                                "train/lr": self.lr_scheduler.get_last_lr()[0],
                            }
                            if self.config.logging.track_grad_norms:
                                logs["train/grad_norm"] = grad_norm.item()
                            
                            wandb.log(logs, step=self.global_step)
                            progress_bar.set_postfix(**logs)
                        
                        # 6. Checkpointing
                        if self.global_step % self.config.training.checkpointing_steps == 0:
                            self.save_checkpoint()
                        
                        if self.global_step >= self.config.training.max_train_steps:
                            break
            self.epoch += 1
        
        # Save final checkpoint
        self.save_checkpoint()
        progress_bar.close()
        print("Training complete.")
