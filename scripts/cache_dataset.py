"""Aspect Ratio Caching for Qwen-SDXL Adapter Training."""

import argparse
import os
import json
import glob
import uuid
from typing import List, Tuple

import torch
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKL
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from tqdm import tqdm
from safetensors.torch import save_file

# SDXL Aspect Ratio Buckets 
AR_BUCKETS = [
    (1024, 1024),  # 1:1
    (1152, 896),   # 5:4
    (896, 1152),   # 4:5
    (1216, 832),   # 3:2
    (832, 1216),   # 2:3
    (1344, 768),   # 16:9
    (768, 1344),   # 9:16
    (1536, 640),   # 21:9
    (640, 1536),   # 9:21
]

def get_closest_bucket(w: int, h: int) -> Tuple[int, int]:
    """Find the closest aspect ratio bucket for given width and height."""
    aspect_ratio = w / h
    closest_bucket = min(AR_BUCKETS, key=lambda b: abs((b[0] / b[1]) - aspect_ratio))
    return closest_bucket

def get_image_paths(data_dir: str) -> List[str]:
    """Get all image paths from the dataset directory."""
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    files = []
    for ext in extensions:
        files.extend(
            glob.glob(os.path.join(data_dir, ext))
        )
    return files

@torch.no_grad()
def cache_dataset(
    raw_data_dir: str,
    cache_dir: str,
    qwen_model_id: str = "Qwen/Qwen3.5-0.8B-Base",
    sdxl_vae_id: str = "madebyollin/sdxl-vae-fp16-fix",
    max_seq_length: int = 256,
    device: str = "cuda"
) -> None:
    """Cache the dataset by encoding images and tokenizing prompts."""
    os.makedirs(cache_dir, exist_ok=True)
    metadata_index = []

    print("Loading frozen models...")
    vae = AutoencoderKL.from_pretrained(sdxl_vae_id, torch_dtype=torch.float16).to(device)
    vae.eval()

    tokenizer = AutoTokenizer.from_pretrained(qwen_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    qwen = AutoModelForCausalLM.from_pretrained(qwen_model_id, torch_dtype=torch.bfloat16, device_map=device)
    qwen.eval()

    image_paths = get_image_paths(raw_data_dir)
    print(f"Found {len(image_paths)} images. Caching dataset...")

    for img_path in tqdm(image_paths):
        base_name = os.path.basename(img_path).split('.')[0]
        txt_path = os.path.join(raw_data_dir, f"{base_name}.txt")
        file_id = str(uuid.uuid4())

        try:
            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size

            # 1. Determine Bucket & Resize/Crop
            target_w, target_h = get_closest_bucket(orig_w, orig_h)

            # Scale so the smallest dimension fills the bucket, then crop excess
            scale = max(target_w / orig_w, target_h / orig_h)
            new_w, new_h = round(orig_w * scale), round(orig_h * scale)

            img = TF.resize(img, [new_h, new_w], antialias=True)
            img = TF.center_crop(img, [target_h, target_w])

            # SDXL Micro-Conditioning coords
            crop_y = (new_h - target_h) // 2
            crop_x = (new_h - target_w) // 2

            micro_conds = torch.tensor(
                [
                    orig_h, orig_w,
                    crop_y, crop_x,
                    target_h, target_w
                ],                dtype=torch.float32
            )

            # 2. VAE Encode
            img_tensor = TF.to_tensor(img).unsqueeze(0).to(device, dtype=torch.float16) * 2.0 - 1.0
            latent = vae.encode(img_tensor).latent_dist.sample() * vae.config.scaling_factor

            # 3. Qwen encoding (conditional & unconditional for CFG)
            prompt = ""
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    prompt = f.read().strip()
            encoded = tokenizer(
                [prompt, ""], padding="max_length", truncation=True, max_length=max_seq_length, return_tensors="pt"
            ).to(device)

            outputs = qwen.model(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                return_dict=True
            )
            hidden_states = outputs.last_hidden_state.detach().cpu().float()
            masks = encoded.attention_mask.detach().cpu().bool()

            # 4. Save safely to SafeTensors
            cache_data = {
                "vae_latents": latent.squeeze(0).cpu().contiguous(),  # [4, H/8, W/8]
                "micro_conds": micro_conds.cpu().contiguous(),        # [6]
                "cond_hidden": hidden_states[0].contiguous(),         # [256, 1024]
                "cond_mask": masks[0].contiguous(),                   # [256]
                "uncond_hidden": hidden_states[1].contiguous(),       # [256, 1024]
                "uncond_mask": masks[1].contiguous(),                 # [256]
            }

            out_file = os.path.join(cache_dir, f"{file_id}.safetensors")
            save_file(cache_data, out_file)

            # 5. Append to Index
            metadata_index.append({
                "id": file_id,
                "bucket": f"{target_w}x{target_h}",
                "path": out_file
            })
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Save metadata index
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump(metadata_index, f, indent=4)
    print(f"Caching complete. Cached {len(metadata_index)} samples to {cache_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    args = parser.parse_args()
    cache_dataset(args.raw_data_dir, args.cache_dir)
