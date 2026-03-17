"""Data preparation utility for Phase 1: Representation Inheritance.

This script builds a balanced distillation dataset by combining:
1. Natural language prose from Conceptual Captions (CC3M).
2. Cleaned, escaped tag-soup from nyanko7/danbooru2023 metadata (bypassing image blobs).
"""

import argparse
import os
from datasets import load_dataset, concatenate_datasets, Dataset
from huggingface_hub import HfFileSystem

def clean_danbooru_tags(example: dict) -> dict:
    """
    Cleans native Danbooru tag strings for SDXL tokenization.
    Converts space-separated tags to comma-separated, removes underscores, 
    and strictly escapes parentheses.
    """
    raw_text = example.get("tag_string", "")
    if not raw_text:
        raw_text = example.get("tags", "")
        if isinstance(raw_text, list):
            raw_text = " ".join([str(t) for t in raw_text])
            
    if not isinstance(raw_text, str) or not raw_text.strip():
        return {"prompt": ""}

    tags = raw_text.split(" ")
    clean_tags = []
    for t in tags:
        t = t.strip()
        if not t: continue  # noqa: E701
        # Replace underscores with spaces
        t = t.replace("_", " ")
        # Escape parentheses for WebUI compatibility
        t = t.replace("(", "\\(").replace(")", "\\)")
        clean_tags.append(t)
    
    return {"prompt": ", ".join(clean_tags)}

def clean_natural_language(example: dict) -> dict:
    """Extracts and cleans natural language prose from CC3M."""
    raw_text = example.get("caption", "")
    if not isinstance(raw_text, str):
        return {"prompt": ""}
    return {"prompt": raw_text.strip()}

def main():
    parser = argparse.ArgumentParser(description="Build Unified Distillation Dataset")
    parser.add_argument("--output_dir", type=str, default="./data/distill_dataset", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=250000, help="Samples PER DATASET to extract")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("Building Unified Distillation Dataset.")
    print(f"Target: {args.num_samples} CC3M + {args.num_samples} Danbooru = {args.num_samples * 2} total.\n")

    # ---------------------------------------------------------
    # 1. Process Conceptual Captions (CC3M)
    # ---------------------------------------------------------
    print("Streaming Conceptual Captions 3M...")
    cc3m_stream = load_dataset("google-research-datasets/conceptual_captions", split="train", streaming=True)
    
    cc3m_texts = []
    for i, example in enumerate(cc3m_stream):
        if i >= args.num_samples:
            break
        cleaned = clean_natural_language(example)
        if cleaned["prompt"]:
            cc3m_texts.append(cleaned)
            
    cc3m_dataset = Dataset.from_list(cc3m_texts)
    print(f"✅ Extracted {len(cc3m_dataset)} natural language prompts.\n")

    # ---------------------------------------------------------
    # 2. Process Danbooru2023 Metadata
    # ---------------------------------------------------------
    print("Locating Danbooru2023 metadata on Hugging Face Hub...")
    fs = HfFileSystem()
    files = fs.glob("datasets/nyanko7/danbooru2023/metadata/*.json*")
    data_files = ["hf://" + f for f in files]
    
    if not data_files:
        raise ValueError("Could not find metadata JSON files in nyanko7/danbooru2023.")
    
    print("Streaming Danbooru2023 Tags...")
    booru_stream = load_dataset("json", data_files=data_files, split="train", streaming=True)
    
    booru_texts = []
    for i, example in enumerate(booru_stream):
        if i >= args.num_samples:
            break
        cleaned = clean_danbooru_tags(example)
        if cleaned["prompt"]:
            booru_texts.append(cleaned)

    booru_dataset = Dataset.from_list(booru_texts)
    print(f"✅ Extracted {len(booru_dataset)} tag-soup prompts.\n")

    # ---------------------------------------------------------
    # 3. Combine, Shuffle, and Save
    # ---------------------------------------------------------
    print("Combining and shuffling datasets...")
    combined_dataset = concatenate_datasets([cc3m_dataset, booru_dataset])
    
    # Shuffle with a fixed seed to perfectly mix CC3M and Danbooru
    combined_dataset = combined_dataset.shuffle(seed=42)

    print(f"Saving optimized dataset to {args.output_dir}...")
    combined_dataset.save_to_disk(args.output_dir)
    print("✅ Done!\n")
    
    # Sanity Check
    print("Sanity Check (Mixed Distribution):")
    for i in range(10):
        print(f"[{i}]: {combined_dataset[i]['prompt']}")

if __name__ == "__main__":
    main()