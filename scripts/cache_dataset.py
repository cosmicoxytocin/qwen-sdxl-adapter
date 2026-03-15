"""Ahead-of-Time (AoT) caching for Qwen-SDXL Adapter Training."""

import argparse
import os
import glob
import json
from pathlib import Path
from typing import List, Tuple

import torch
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKL
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from tqdm import tqdm


def get_image_paths(data_dir: str) -> List[str]:
    """Find all common image files in the target directory."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(data_dir, ext)))
    return files


@torch.no_grad()
def cache_dataset(
    raw_data_dir: str,
    cache_dir: str,
    qwen_model_id: str = "Qwen/Qwen3.5-0.8B-Base",
    sdxl_vae_id: str = "madebyollin/sdxl-vae-fp16-fix",
    image_size: int = 1024,
    max_seq_length: int = 256,
    device: str = "cuda",
) -> None:
    """Process and cache VAE latents and LLM hidden states."""
    os.makedirs(cache_dir, exist_ok=True)

    print("Loading frozen models...")
    # Load VAE
    vae = AutoencoderKL.from_pretrained(sdxl_vae_id, torch_dtype=torch.float16).to(device)
    vae.eval()

    # Load Qwen
    tokenizer = AutoTokenizer.from_pretrained(qwen_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    qwen = AutoModelForCausalLM.from_pretrained(
        qwen_model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    qwen.eval()

    image_paths = get_image_paths(raw_data_dir)
    print(f"Found {len(image_paths)} images. Caching dataset...")
    for img_path in tqdm(image_paths):
        base_name = Path(img_path).stem
        txt_path = os.path.join(raw_data_dir, f"{base_name}.txt")

        # 1. Image Processing & VAE Latent Caching
        try:
            img = Image.open(img_path).convert("RGB")
            # Center crop to 1024x1024 for Phase 1 PoC
            w, h = img.size
            min_dim = min(w, h)
            img = TF.center_crop(img, [min_dim, min_dim])
            img = TF.resize(img, [image_size, image_size], antialias=True)

            # Convert to tensor [-1, 1] expected by VAE
            img_tensor = TF.to_tensor(img).unsqueeze(0).to(device, dtype=torch.float16) * 2.0 - 1.0

            # Encode to latent [1, 4, 128, 128]
            latent_dist = vae.encode(img_tensor).latent_dist
            latent = (
                latent_dist.sample() * vae.config.scaling_factor
            )  # Scale by VAE factor (default 0.18215)

            # Save SDXL micro-conditioning stats: (original_h, original_w, crop_y, crop_x, target_h, target_w)
            crop_y = (h - min_dim) // 2
            crop_x = (w - min_dim) // 2
            micro_conds = torch.tensor(
                [h, w, crop_y, crop_x, image_size, image_size], dtype=torch.float32
            )
        except Exception as e:
            print(f"Failed to process image {img_path}: {e}")
            continue

        # 2. Text Processing & Qwen Hidden State Caching
        prompt = ""
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()

        # Cache the unconditional ('empty') prompt for CFG
        prompts = [prompt, ""]  # [conditional, unconditional]

        # Tokenize both (pad to max_seq_length, truncate if over)
        encode = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        ).to(device)

        # Forward pass through Qwen to get last hidden state
        # `output.hidden_states` isn't needed if we take the standard output for causal LM
        # but AutoModelForCausalLM outputs logits. Need base model's hidden states
        outputs = qwen.model(
            input_ids=encode.input_ids, attention_mask=encode.attention_mask, return_dict=True
        )
        # [2, 256, 1024]
        hidden_states = outputs.last_hidden_state.detach().cpu()
        attention_mask = encode.attention_mask.detach().cpu().bool()

        # Split into conditional and unconditional parts
        cond_hidden = hidden_states[0]
        uncond_hidden = hidden_states[1]
        cond_mask = attention_mask[0]
        uncond_mask = attention_mask[1]

        # 3. Save to disk as a single dictionary
        cache_data = {
            "vae_latents": latent.squeeze(0).cpu(),  # [4, 128, 128]
            "micro_conds": micro_conds,  # [6]
            "cond_hidden": cond_hidden,  # [256, 1024]
            "cond_mask": cond_mask,  # [256]
            "uncond_hidden": uncond_hidden,  # [256, 1024]
            "uncond_mask": uncond_mask,  # [256]
        }
        out_file = os.path.join(cache_dir, f"{base_name}.pt")
        torch.save(cache_data, out_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        required=True,
        help="Directory containing raw images and prompts.",
    )
    parser.add_argument(
        "--cache_dir", type=str, required=True, help="Output path for cached .pt tensors."
    )
    args = parser.parse_args()
    cache_dataset(args.raw_data_dir, args.cache_dir)
