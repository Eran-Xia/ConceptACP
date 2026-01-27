# Migration Guide: Original → Block-Level Fine-Tuning

Transition guide for the new efficient block-level approach: **70-90% memory reduction**, **2-5x faster training**, **identical results**.

## Workflow Comparison

### Original
```bash
uv run src/conceptacp/utils/make_data.py unlearn_type=nudity
uv run src/conceptacp/utils/finetune.py data.unlearn_type=nudity
uv run src/conceptacp/utils/inference.py unlearn_type=nudity
```
Memory: 7-10GB VRAM | Speed: Full UNet per batch

### Block-Level (Recommended)
```bash
uv run src/conceptacp/utils/make_block_data.py unlearn_type=nudity
uv run src/conceptacp/utils/finetune_block.py data.unlearn_type=nudity
uv run src/conceptacp/utils/inference.py unlearn_type=nudity
```
Memory: 0.5-2GB VRAM | Speed: 2-5x faster

## Key Differences

**Data format:** Block-level stores (input, target_output) pairs per timestep vs. final reconstructions only.

**Checkpoints:** Same format, block-level uses `_block.pt` suffix.

**Disk space:** Block-level files are larger (~200-800MB vs ~100-500MB) but enable faster training.

## Migration Options

**For new work:** Use block-level by default.

**For existing data:**
- Continue with original approach (no changes needed)
- OR regenerate with `make_block_data.py` for faster training

## Common Scenarios

**Limited VRAM (<8GB):** Must use block-level approach.

**Hyperparameter tuning:** Generate data once with `make_block_data.py`, then iterate quickly.

**Multiple concepts:** Generate all data first, then train in parallel.

## Validation

```bash
uv run src/conceptacp/utils/validate_checkpoints.py \
    --original models/nudity_99_0.2_0.2.pt \
    --block models/nudity_99_0.2_0.2_block.pt
```
