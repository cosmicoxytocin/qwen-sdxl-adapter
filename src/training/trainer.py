"""Training loop for Qwen-SDXL Adapter."""

import os
from typing import Dict, Any, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import wandb
from tqdm import tqdm

from ..models.bridge import CausalToSpatialPerceiverBridge
from ..models.sampler import Diff2FlowEulerSampler
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
        device: torch.device,
        # Runtime validation
        qwen: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        vae: Optional[nn.Module] = None,
        sampler: Optional[Diff2FlowEulerSampler] = None,
    ) -> None:
        self.config = config
        self.device = device
        self.train_dataloader = train_dataloader

        # Modules
        self.adapter = adapter.to(device)
        self.unet = unet.to(device)
        self.objective = objective.to(device)

        # Runtime validation components (optional)
        self.qwen = qwen
        self.tokenizer = tokenizer
        self.vae = vae
        self.sampler = sampler

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

        if getattr(self.config.training, "resume_from_checkpoint", None):
            self.load_checkpoint(self.config.training.resume_from_checkpoint)

    def save_checkpoint(self) -> None:
        """Saves adapter weights and full optimizer state (Kohya style)."""
        from safetensors.torch import save_file

        os.makedirs(self.config.training.output_dir, exist_ok=True)
        base_path = os.path.join(self.config.training.output_dir, f"step_{self.global_step}")

        # 1. Save Weights
        save_file(self.adapter.state_dict(), f"{base_path}_adapter.safetensors")

        # 2. Save Optimizer State
        if getattr(self.config.training, "save_optimizer_state", True):
            state = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "global_step": self.global_step,
                "epoch": self.epoch,
            }
            torch.save(state, f"{base_path}_state.pt")

        print(f"\n[Checkpoint Saved] -> {base_path}")

    def load_checkpoint(self, checkpoint_prefix: str) -> None:
        """Restores weights and optimizer state to resume training perfectly."""
        from safetensors.torch import load_file

        print(f"\n[Resuming] Loading checkpoint from: {checkpoint_prefix}")

        # 1. Load Weights
        weight_path = f"{checkpoint_prefix}_adapter.safetensors"
        if os.path.exists(weight_path):
            self.adapter.load_state_dict(load_file(weight_path))
        else:
            raise FileNotFoundError(f"Weight file missing: {weight_path}")

        # 2. Load State
        state_path = f"{checkpoint_prefix}_state.pt"
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)
            self.optimizer.load_state_dict(state["optimizer"])
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])
            self.scaler.load_state_dict(state["scaler"])
            self.global_step = state["global_step"]
            self.epoch = state["epoch"]
            print(f"[Resumed] Starting from Step: {self.global_step}, Epoch: {self.epoch}")
        else:
            print("[Warning] No optimizer state found. Resuming weights only.")

    @torch.no_grad()
    def generate_validation_samples(self) -> None:
        """Kohya-ss/sd-scripts runtime image sampling. Evaluates current adapter weights."""
        if not all([self.qwen, self.tokenizer, self.vae, self.sampler]):
            print("[Warning] Missing validation modules. Skipping generation.")
            return

        print(f"\n--- Generating Validation Samples (Step {self.global_step}) ---")
        self.adapter.eval()
        torch.cuda.empty_cache()  # Free memory before inference

        validation_images = {}
        for idx, prompt in enumerate(self.config.logging.validation_prompts):
            if not prompt.strip():
                continue

            prompts = ["", prompt]  # Unconditional, Conditional
            encoded = self.tokenizer(
                prompts, padding="max_length", truncation=True, max_length=256, return_tensors="pt"
            ).to(self.device)

            qwen_outputs = self.qwen.model(
                input_ids=encoded.input_ids, attention_mask=encoded.attention_mask, return_dict=True
            )
            hidden_states = qwen_outputs.last_hidden_state.to(self.dtype)
            mask = encoded.attention_mask.bool()

            with torch.autocast(
                device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp
            ):
                adapter_ctx, adapter_pooled = self.adapter(hidden_states, mask)

            micro_conds = torch.tensor(
                [[1024, 1024, 0, 0, 1024, 1024]] * 2, dtype=self.dtype, device=self.device
            )
            added_cond_kwargs = {"text_embeds": adapter_pooled, "time_ids": micro_conds}

            # Euler ODE Sampling (20 Steps)
            x = torch.randn((1, 4, 128, 128), dtype=self.dtype, device=self.device)
            dt = 1.0 / 20

            for i in range(20):
                fm_t = torch.tensor([i * dt], dtype=torch.float32, device=self.device)
                x_in = torch.cat([x, x], dim=0)
                fm_t_in = torch.cat([fm_t, fm_t], dim=0)

                dm_t, dm_x = self.sampler.convert_fm_to_dm(fm_t_in, x_in)

                with torch.autocast(
                    device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp
                ):
                    eps_pred = self.unet(
                        dm_x.to(self.dtype),
                        dm_t.to(self.dtype),
                        encoder_hidden_states=adapter_ctx,
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample

                v_pred = self.sampler.predict_velocity(dm_t, dm_x, eps_pred)
                v_uncond, v_cond = v_pred.chunk(2)
                v_cfg = v_uncond + 4.5 * (v_cond - v_uncond)  # CFG = 4.5
                x = x + v_cfg.to(self.dtype) * dt

            # VAE Decode
            x = x / self.vae.config.scaling_factor
            with torch.autocast(
                device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp
            ):
                image_tensor = self.vae.decode(x).sample

            # Normalize and format for WandB
            img_normalized = (image_tensor[0] / 2 + 0.5).clamp(0, 1)
            validation_images[f"Validation/Sample_{idx}"] = wandb.Image(
                img_normalized.float().cpu(), caption=prompt
            )

        if validation_images:
            wandb.log(validation_images, step=self.global_step)

        self.adapter.train()
        torch.cuda.empty_cache()

    def train(self) -> None:
        """Main training loop."""
        print(
            f"Starting training on {self.device} for {self.config.training.max_train_steps} steps..."
        )

        if len(self.train_dataloader) == 0:
            raise ValueError("DataLoader is empty!")

        self.adapter.train()
        self.unet.eval()  # UNet is frozen

        # Resume progress bar naturally if resuming from checkpoint
        progress_bar = tqdm(
            total=self.config.training.max_train_steps, desc="Training", initial=self.global_step
        )
        micro_step = 0

        while self.global_step < self.config.training.max_train_steps:
            for batch in self.train_dataloader:
                x1 = batch["vae_latents"].to(self.device, non_blocking=True)
                qwen_hidden = batch["qwen_hidden_states"].to(self.device, non_blocking=True)
                qwen_mask = batch["qwen_mask"].to(self.device, non_blocking=True)
                micro_conds = batch["micro_conds"].to(self.device, non_blocking=True)

                with torch.autocast(
                    device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp
                ):
                    adapter_ctx, adapter_pooled = self.adapter(qwen_hidden, qwen_mask)
                    loss = self.objective(
                        unet=self.unet,
                        adapter_ctx=adapter_ctx,
                        adapter_pooled=adapter_pooled,
                        x1=x1,
                        micro_conds=micro_conds,
                    )
                    loss = loss / self.config.training.gradient_accumulation_steps

                self.scaler.scale(loss).backward()
                micro_step += 1

                if micro_step % self.config.training.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.adapter.parameters(), self.config.training.max_grad_norm
                    )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    self.global_step += 1
                    progress_bar.update(1)

                    if self.global_step % self.config.logging.log_interval == 0:
                        logs = {
                            "train/loss": loss.item()
                            * self.config.training.gradient_accumulation_steps,
                            "train/lr": self.lr_scheduler.get_last_lr()[0],
                        }
                        if self.config.logging.track_grad_norms:
                            logs["train/grad_norm"] = grad_norm.item()

                        wandb.log(logs, step=self.global_step)
                        progress_bar.set_postfix(**logs)

                    # ---------------------------------------------------------
                    # RUNTIME VALIDATION
                    # ---------------------------------------------------------
                    if (
                        self.config.training.validation_steps > 0
                        and self.global_step % self.config.training.validation_steps == 0
                    ):
                        self.generate_validation_samples()

                    # ---------------------------------------------------------
                    # CHECKPOINTING
                    # ---------------------------------------------------------
                    if self.global_step % self.config.training.checkpointing_steps == 0:
                        self.save_checkpoint()

                    if self.global_step >= self.config.training.max_train_steps:
                        break
            self.epoch += 1

        self.save_checkpoint()
        progress_bar.close()
        print("Training complete.")
