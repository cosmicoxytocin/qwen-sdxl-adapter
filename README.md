# QWEN-to-SDXL Flow-Matching Adapter (Smoke Test)

This codebase implements a bidirectional Perceiver Resampler to map QWEN3.5-0.8B-Base text embeddings to SDXL's UNet conditioning space, trained via Flow Matching (Diff2Flow) objective.

## Usage

**Train:**
This will cache the image/prompt and overfit the adapter.
```bash
accelerate launch train.py \
    --image_path "test_image.jpg" \
    --prompt "caption here..." \
    --output_dir "./smoke_test_output"
```

**Eval:**

```bash
python -m evaluate.py \
    --checkpoint_path "./smoke_test_output/checkpoint-final" \
    --prompt "<same as train>" \
    --output_path "reconstruction.png"
```
