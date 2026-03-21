"""Inference script for the Qwen-SDXL Adapter."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from safetensors.torch import load_file
from tqdm.auto import tqdm

from src.models.bridge import CausalToSpatialPerceiverBridge
from src.models.sampler import Diff2FlowEulerSampler


class QwenSDXLPipeline:
    
    def __init__(
        self,
        qwen_id: str = "Qwen/Qwen3.5-0.8B-Base",
        unet_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
        vae_id: str = "madebyollin/sdxl-vae-fp16-fix",
        adapter_ckpt: str = "",
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.dtype = torch.bfloat16

        print("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16).to(self.device).eval()

        print("Loading Qwen...")
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.qwen = AutoModelForCausalLM.from_pretrained(qwen_id, torch_dtype=self.dtype).to(self.device).eval()

        print("Loading UNet...")
        self.unet = UNet2DConditionModel.from_pretrained(unet_path, subfolder="unet", torch_dtype=self.dtype).to(self.device).eval()

        print("Loading Adapter...")
        self.adapter = CausalToSpatialPerceiverBridge(
            depth=6,
            qwen_dim=1024,
            sdxl_context_dim=2048,
            sdxl_pooled_dim=1280,
            num_queries=78
        ).to(self.device, dtype=self.dtype).eval()

        if adapter_ckpt:
            self.adapter.load_state_dict(load_file(adapter_ckpt))
        
        scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
        self.sampler = Diff2FlowEulerSampler(scheduler, self.device)
        print("Pipeline initialized.")

        @torch.no_grad()
        def generate(
            self,
            prompt: str,
            negative_prompt: str = "",
            width: int = 1024,
            height: int = 1024,
            num_inference_steps: int = 25,
            cfg_scale: float = 4.5,
            seed: int = None,
            output_dir: str = "./outputs"
        ) -> Image.Image:
            
            os.makedirs(output_dir, exist_ok=True)
            if seed is not None:
                torch.manual_seed(seed)
            
            prompts = [negative_prompt, prompt]
            encoded = self.tokenizer(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)

            input_ids = encoded.input_ids[:2, :256]
            attention_mask = encoded.attention_mask[:2, :256]

            qwen_out = self.qwen.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            hidden_states = qwen_out.last_hidden_state.to(self.dtype)
            mask = attention_mask.bool()

            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                adapter_ctx, adapter_pooled = self.adapter(hidden_states, mask)
            
            micro_conds = torch.tensor([[height, width, 0, 0, height, width]] * 2, dtype=self.dtype, device=self.device)
            added_cond_kwargs = {"text_embeds": adapter_pooled, "time_ids": micro_conds}

            # Remove time shifting for now
            x = torch.randn((1, 4, height // 8, width // 8), dtype=self.dtype, device=self.device)
            dt = 1.0 / num_inference_steps

            progress_bar = tqdm(total=num_inference_steps, desc="Generating Image")
            for i in range(num_inference_steps):
                t = i * dt
                fm_t = torch.tensor([t, t], dtype=torch.float32, device=self.device)
                x_in = torch.cat([x, x], dim=0)

                dm_t, dm_x = self.sampler.convert_fm_to_dm(fm_t, x_in)

                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    eps_pred = self.unet(
                        dm_x.to(self.dtype),
                        dm_t.to(self.dtype),
                        encoder_hidden_states=adapter_ctx,
                        added_cond_kwargs=added_cond_kwargs
                    ).sample
                
                v_pred = self.sampler.predict_velocity(dm_t, dm_x, eps_pred)
                v_uncond, v_cond = v_pred.chunk(2)
                v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)

                x = x + v_cfg.to(self.dtype) * dt
                progress_bar.update(1)
            
            x = x / self.vae.config.scaling_factor
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                image_tensor = self.vae.decode(x.to(torch.float16)).sample
            
            img_normalized = (image_tensor[0] / 2 + 0.5).clamp(0, 1)
            img_pil = TF.to_pil_image(img_normalized)

            return img_pil
            
