"""Main training loop for Phase 2 Diff2Flow."""

import os
import torch
import wandb
from tqdm.auto import tqdm
from safetensors.torch import save_file
import torchvision.transforms.functional as TF
from PIL import Image

from src.models.sampler import Diff2FlowEulerSampler


class Trainer:
    def __init__(
        self, unet, adapter, qwen, vae, tokenizer, dataloader, objective, config, device, dtype
    ):
        self.unet = unet
        self.adapter = adapter
        self.qwen = qwen  # Lives on CPU
        self.vae = vae  # Lives on CPU
        self.tokenizer = tokenizer
        self.train_dataloader = dataloader
        self.objective = objective
        self.config = config
        self.device = device
        self.dtype = dtype

        self.optimizer = torch.optim.AdamW(
            self.adapter.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        total_steps = config.training.max_train_steps
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=1e-6
        )

        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=config.training.mixed_precision == "fp16"
        )
        self.global_step = 0
        self.epoch = 0

        if config.logging.use_wandb:
            from omegaconf import OmegaConf

            wandb.init(
                project=config.logging.wandb_project,
                config=OmegaConf.to_container(config, resolve=True),
            )

    def train(self):
        progress_bar = tqdm(total=self.config.training.max_train_steps, desc="Training")
        micro_step = 0

        # SAFETY NET: Catch Colab cell stops to save the checkpoint
        try:
            while self.global_step < self.config.training.max_train_steps:
                for batch in self.train_dataloader:
                    # 1. Load pre-cached ARB Data
                    x1 = batch["vae_latents"].to(self.device, non_blocking=True)
                    qwen_hidden = batch["qwen_hidden_states"].to(self.device, non_blocking=True)
                    qwen_mask = batch["qwen_mask"].to(self.device, non_blocking=True)
                    micro_conds = batch["micro_conds"].to(self.device, non_blocking=True)

                    # 2. Forward Pass
                    with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                        adapter_ctx, adapter_pooled = self.adapter(qwen_hidden, qwen_mask)

                        loss = self.objective(
                            unet=self.unet,
                            adapter_ctx=adapter_ctx,
                            adapter_pooled=adapter_pooled,
                            x1=x1,
                            micro_conds=micro_conds,
                        )
                        loss = loss / self.config.training.gradient_accumulation_steps

                    # 3. Backward Pass
                    self.scaler.scale(loss).backward()
                    micro_step += 1

                    # 4. Optimizer Step
                    if micro_step % self.config.training.gradient_accumulation_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.adapter.parameters(), self.config.training.max_grad_norm
                        )

                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)

                        self.global_step += 1
                        progress_bar.update(1)

                        if self.global_step % self.config.logging.log_interval == 0:
                            wandb.log(
                                {
                                    "train/loss": loss.item()
                                    * self.config.training.gradient_accumulation_steps,
                                    "train/lr": self.lr_scheduler.get_last_lr()[0],
                                },
                                step=self.global_step,
                            )
                            progress_bar.set_postfix({"loss": loss.item()})

                        # VALIDATION
                        if (
                            self.config.training.validation_steps > 0
                            and self.global_step % self.config.training.validation_steps == 0
                        ):
                            self.generate_validation_samples()

                        # CHECKPOINTING
                        if self.global_step % self.config.training.checkpointing_steps == 0:
                            self.save_checkpoint()

                        if self.global_step >= self.config.training.max_train_steps:
                            break
                self.epoch += 1

        except KeyboardInterrupt:
            print("\n[Interrupt] Caught KeyboardInterrupt! Saving current state safely...")
            self.save_checkpoint()
            progress_bar.close()
            print("Safe exit complete.")
            return

        self.save_checkpoint()
        progress_bar.close()
        print("Training complete.")

    @torch.no_grad()
    def generate_validation_samples(self):
        print("\nRunning Validation...")
        self.adapter.eval()

        # 1. JUGGLE TO GPU
        self.qwen.to(self.device)
        self.vae.to(self.device)

        from diffusers import DDPMScheduler

        scheduler = DDPMScheduler.from_pretrained(
            self.config.model.sdxl_model_id, subfolder="scheduler"
        )
        sampler = Diff2FlowEulerSampler(scheduler, self.device)

        val_images = []
        for prompt in self.config.training.validation_prompts:
            # STRICT Truncation to prevent batch overflow
            encoded = self.tokenizer(
                ["", prompt],
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
            input_ids = encoded.input_ids[:2, :256]
            attention_mask = encoded.attention_mask[:2, :256]

            qwen_out = self.qwen.model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
            hidden_states = qwen_out.last_hidden_state.to(self.dtype)
            mask = attention_mask.bool()

            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                adapter_ctx, adapter_pooled = self.adapter(hidden_states, mask)

            micro_conds = torch.tensor(
                [[1024, 1024, 0, 0, 1024, 1024]] * 2, dtype=self.dtype, device=self.device
            )
            added_cond_kwargs = {"text_embeds": adapter_pooled, "time_ids": micro_conds}

            x = torch.randn((1, 4, 128, 128), dtype=self.dtype, device=self.device)
            dt = 1.0 / 25

            for i in range(25):
                t = i * dt
                fm_t = torch.tensor([t, t], dtype=torch.float32, device=self.device)
                x_in = torch.cat([x, x], dim=0)
                dm_t, dm_x = sampler.convert_fm_to_dm(fm_t, x_in)

                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    eps_pred = self.unet(
                        dm_x.to(self.dtype),
                        dm_t.to(self.dtype),
                        encoder_hidden_states=adapter_ctx,
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample

                v_pred = sampler.predict_velocity(dm_t, dm_x, eps_pred)
                v_uncond, v_cond = v_pred.chunk(2)
                v_cfg = v_uncond + 4.5 * (v_cond - v_uncond)
                x = x + v_cfg.to(self.dtype) * dt

            x = x / self.vae.config.scaling_factor
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                img_tensor = self.vae.decode(x.to(torch.float16)).sample

            img_normalized = (img_tensor[0] / 2 + 0.5).clamp(0, 1)
            img_pil = TF.to_pil_image(img_normalized)
            val_images.append(wandb.Image(img_pil, caption=prompt))

        if self.config.logging.use_wandb:
            wandb.log({"validation/images": val_images}, step=self.global_step)

        # 2. JUGGLE TO CPU & CLEAR VRAM
        self.qwen.to("cpu")
        self.vae.to("cpu")
        torch.cuda.empty_cache()
        self.adapter.train()

    def save_checkpoint(self):
        os.makedirs(self.config.training.output_dir, exist_ok=True)
        ckpt_path = os.path.join(
            self.config.training.output_dir, f"adapter_step_{self.global_step}.safetensors"
        )
        save_file(self.adapter.state_dict(), ckpt_path)
        print(f"\nSaved checkpoint: {ckpt_path}")
