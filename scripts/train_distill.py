"""Phase 1: Representation Inheritance Distillation Training Script."""

import argparse
import os
from datasets import load_from_disk
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    get_cosine_schedule_with_warmup,
)

from src.models.bridge import CausalToSpatialPerceiverBridge
from src.training.distill_loss import DistillationLoss
from src.config import ExperimentConfig
from safetensors.torch import save_file

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Adapter Distillation")
    parser.add_argument("--dataset_dir", type=str, default="./data/distill_dataset", help="Path to compiled Hugging Face dataset")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/distill", help="Directory to save adapter weights")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for distillation")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Peak learning rate")
    parser.add_argument("--struct_weight", type=float, default=1.0, help="Weight for the ProCLIP structure alignment loss")
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    wandb.init(project="qwen-sdxl-distillation", name="representation-inheritance-phase1")

    print("1. Loading frozen SDXL CLIP models...")
    tokenizer_1 = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer_2 = AutoTokenizer.from_pretrained("laion/CLIP-ViT-BigG-14-laion2B-39B-b160k")

    clip_1 = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype).to(device).eval()
    clip_2 = CLIPTextModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", torch_dtype=dtype).to(device).eval()
    clip_1.requires_grad_(False)
    clip_2.requires_grad_(False)

    print("2. Loading frozen Qwen LLM...")
    qwen_id = "Qwen/Qwen3.5-0.8B-Base"
    qwen_tok = AutoTokenizer.from_pretrained(qwen_id, trust_remote_code=True)
    if qwen_tok.pad_token_id is None:
        qwen_tok.pad_token = qwen_tok.eos_token
    qwen = AutoModelForCausalLM.from_pretrained(qwen_id, torch_dtype=dtype).to(device).eval()
    qwen.requires_grad_(False)

    print("3. Intializing CSPB Adapter...")
    adapter = CausalToSpatialPerceiverBridge(
        depth=6,
        qwen_dim=1024,
        internal_dim=1024,
        sdxl_context_dim=2048,
        sdxl_pooled_dim=1280,
        num_queries=78
    ).to(device, dtype=dtype)
    adapter.train()

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    loss_fn = DistillationLoss(
        struct_weight=args.struct_weight
    ).to(device)

    # 4. Dataloader
    print("4. Loading distillation dataset from disk...")
    full_dataset = load_from_disk(args.dataset_dir)
    dataloader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    num_steps = len(dataloader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_steps * 0.05),
        num_training_steps=num_steps
    )

    # Training Loop
    print("5. Starting distillation training loop...")
    progress_bar = tqdm(total=num_steps, desc="Distilling", mininterval=1.0)

    global_step = 0
    os.makedirs(args.output_dir, exist_ok=True)

    for batch in dataloader:
        batch_texts = batch["prompt"]

        # --- A. Get CLIP Ground Truth (Teacher) ---
        with torch.no_grad():
            # CLIP 1
            tok1 = tokenizer_1(batch_texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
            out1 = clip_1(tok1.input_ids, output_hidden_states=True)
            clip_hidden_1 = out1.hidden_states[-2]
            
            # CLIP 2
            tok2 = tokenizer_2(batch_texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
            out2 = clip_2(tok2.input_ids, output_hidden_states=True)
            clip_hidden_2 = out2.hidden_states[-2]
            clip_pooled = out2.text_embeds

            # SDXL concatenates the hidden states (77x2048)
            target_context = torch.cat([clip_hidden_1, clip_hidden_2], dim=-1).to(dtype)
            target_pooled = clip_pooled.to(dtype)

        # --- B. Get Adapter Prediction (Student) ---
        qwen_enc = qwen_tok(batch_texts, padding="max_length", max_length=256, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            qwen_out = qwen.model(input_ids=qwen_enc.input_ids, attention_mask=qwen_enc.attention_mask, return_dict=True)
            qwen_hidden = qwen_out.last_hidden_state.to(dtype)
        
        qwen_mask = qwen_enc.attention_mask.bool()

        # Forward pass through adapter
        with torch.autocast(device_type="cuda", dtype=dtype):
            adapter_ctx, adapter_pooled = adapter(qwen_hidden, qwen_mask)
            
            # Calculate ProCLIP Distillation Loss
            loss_ctx, metrics_ctx = loss_fn(adapter_ctx, target_context)
            loss_pool, metrics_pool = loss_fn(adapter_pooled, target_pooled)
            
            total_loss = loss_ctx + loss_pool
        
        # --- C. Backpropagation ---
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        # Logging
        wandb.log(
            {
                "train/total_loss": total_loss.item(),
                "train/ctx_ins": metrics_ctx["loss/l_ins"],
                "train/ctx_struct": metrics_ctx["loss/l_struct"],
                "train/pool_struct": metrics_pool["loss/l_struct"],
                "train/lr": scheduler.get_last_lr()[0]
            },            step=global_step
        )

        progress_bar.update(1)
        progress_bar.set_postfix({"loss": f"{total_loss.item():.4f}"})
        global_step += 1

        # Checkpointing
        if global_step % 5000 == 0:
            ckpt_path = os.path.join(args.output_dir, f"adapter_distilled_step_{global_step}.safetensors")
            save_file(adapter.state_dict(), ckpt_path)
        
    # Final Save
    final_path = os.path.join(args.output_dir, "adapter_distilled_final.safetensors")
    save_file(adapter.state_dict(), final_path)
    print(f"Distillation complete. Final adapter weights saved to: {final_path}")

if __name__ == "__main__":
    main()