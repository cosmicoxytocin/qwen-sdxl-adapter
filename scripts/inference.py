"""Inference script for the Qwen-SDXL Adapter."""


import argparse
import os
import time
from datetime import datetime
from typing import Optional, Tuple
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

def apply_flow_shift(t: float, shift: float = 3.0) -> float:
    """Applies time-shift to the Diff2Flow noise schedule."""
    # Invert to standard FM
    u = 1.0 - t
    # Apply shift curve
    u_shifted = (shift * u) / (1.0 + (shift - 1.0) * u)
    # Invert back to Diff2Flow
    return 1.0 - u_shifted


class QwenSDXLPipeline:
    
    def __init__(
        self,
        qwen_id: str = "Qwen/Qwen3.5-0.8B-Base",
        unet_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
        vae_id: str = "madebyollin/sdxl-vae-fp16-fix",
        adapter_ckpt: str = "",
        is_single_file_unet: bool = False,
        device: str = "cuda"
    ):
        self.device = device
        self.dtype = torch.bfloat16

        print("1. Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16).to(self.device).eval()

        print("2. Loading Qwen LLM...")
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.qwen = AutoModelForCausalLM.from_pretrained(qwen_id, torch_dtype=self.dtype).to(self.device).eval()

        print("3. Loading SDXL UNet...")
        if is_single_file_unet or unet_path.endswith(".safetensors"):
            self.unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=self.dtype).to(self.device).eval()
        else:
            self.unet = UNet2DConditionModel.from_pretrained(unet_path, subfolder="unet", torch_dtype=self.dtype).to(self.device).eval()
        
        print("4. Initializing CSPB Adapter...")
        self.adapter = CausalToSpatialPerceiverBridge(
            depth=6,
            qwen_dim=1024,
            internal_dim=1024,
            sdxl_context_dim=2048,
            sdxl_pooled_dim=1280,
            num_queries=78
        ).to(self.device, dtype=self.dtype).eval()

        if adapter_ckpt:
            print(f"  -> Loading adapter weights from {adapter_ckpt}...")
            self.adapter.load_state_dict(load_file(adapter_ckpt))
        
        # Setup standard SDXL sampler to get beta schedules
        scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
        self.sampler = Diff2FlowEulerSampler(scheduler, self.device)

        print("Pipeline initialization complete.")
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 25,
        cfg_scale: float = 4.5,
        shift: float = 3.0,
        seed: Optional[int] = None,
        output_dir: str = "./outputs"
    ) -> Image.Image:
        """Generates an image from the given prompt using the Qwen-SDXL Adapter pipeline."""
        os.makedirs(output_dir, exist_ok=True)

        # Round dimensions to nearest 64
        width = (width // 64) * 64
        height = (height // 64) * 64

        if seed is not None:
            torch.manual_seed(seed)
        
        # 1. Encode Text
        prompts = [negative_prompt, prompt]
        encoded = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.device)

        qwen_out = self.qwen.model(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
            return_dict=True
        )
        hidden_states = qwen_out.last_hidden_state.to(self.dtype)
        mask = encoded.attention_mask.bool()

        # 2. Adapter embeddings
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            adapter_ctx, adapter_pooled = self.adapter(hidden_states, mask)
        
        # 3. Setup micro-conditioning
        # batched for [uncond, cond]
        micro_conds = torch.tensor([
            [height, width, 0, 0, height, width]
        ] * 2, dtype=self.dtype, device=self.device)
        added_cond_kwargs = {"text_embeds": adapter_pooled, "time_ids": micro_conds}

        # 4. Initialize Noise
        shape = (1, 4, height // 8, width // 8)
        x = torch.randn(shape, dtype=self.dtype, device=self.device)
        dt = 1.0 / num_inference_steps
        
        # 5. Euler ODE Loop with Time-Shifting
        progress_bar = tqdm(total=num_inference_steps, desc="Sampling")
        
        for i in range(num_inference_steps):
            t_curr = i / num_inference_steps
            t_next = (i + 1) / num_inference_steps

            fm_t_curr = apply_flow_shift(t_curr, shift=shift)
            fm_t_next = apply_flow_shift(t_next, shift=shift)

            dt_shifted = fm_t_next - fm_t_curr
            
            fm_t_tensor = torch.tensor([fm_t_curr, fm_t_next], dtype=torch.float32, device=self.device)
            x_in = torch.cat([x, x], dim=0)  # Duplicate for uncond/cond

            dm_t, dm_x = self.sampler.convert_fm_to_dm(fm_t_tensor, x_in)

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

            x = x + v_cfg.to(self.dtype) * dt_shifted
            progress_bar.update(1)
            
        progress_bar.close()
            
        # 6. Decode Image
        x = x / self.vae.config.scaling_factor
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            image_tensor = self.vae.decode(x.to(torch.float16)).sample
            
        img_normalized = (image_tensor[0] / 2 + 0.5).clamp(0, 1)
        img_pil = TF.to_pil_image(img_normalized)
        
        # 7. Auto-Index Saving
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        img_pil.save(filepath)
        
        print(f"🖼️ Saved to: {filepath}")
        return img_pil

# --- CLI Execution ---
if __name__ == "__main__":
    import torchvision.transforms.functional as TF
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--adapter_ckpt", type=str, required=True)
    parser.add_argument("--unet_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=4.5)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    
    pipe = QwenSDXLPipeline(
        unet_path=args.unet_path,
        adapter_ckpt=args.adapter_ckpt,
        is_single_file_unet=args.unet_path.endswith(".safetensors")
    )
    
    pipe.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg,
        shift=args.shift,
        seed=args.seed
    )
