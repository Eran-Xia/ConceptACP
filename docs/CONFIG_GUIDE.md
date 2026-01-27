# Configuration Guide

## Unified Model Path

**Single source of truth:** `src/conceptacp/utils/conf/data/default.yaml`

```yaml
model_path: stabilityai/sdxl-turbo  # Only defined here
```

## Configuration Files

### Data Config (`conf/data/default.yaml`)
Base config with model path, SAE checkpoints, prompts, and data paths.

**Used by:** `make_data.py`, `make_block_data.py`
**Access:** `cfg.model_path`

### Finetune Config (`conf/finetune.yaml`)
Imports data config. Contains training hyperparameters.

**Used by:** `finetune.py`, `finetune_block.py`
**Access:** `cfg.data.model_path`

### Inference Config (`conf/inference.yaml`)
Imports data config. Contains inference settings and checkpoint paths.

**Used by:** `inference.py`
**Access:** `cfg.data.model_path`

## Overriding Model Path

### Command Line (Recommended)

```bash
# Data scripts
uv run src/conceptacp/utils/make_block_data.py model_path=/path/to/sdxl

# Training/inference scripts
uv run src/conceptacp/utils/finetune_block.py data.model_path=/path/to/sdxl
uv run src/conceptacp/utils/inference.py data.model_path=/path/to/sdxl
```

### Edit Config (Permanent)

Edit `conf/data/default.yaml`:
```yaml
model_path: /path/to/sdxl
```

## Key Parameters

**Data config:**
- `model_path`: SDXL model location
- `device`: CUDA device (null = auto)
- `capture_timesteps`: Timesteps to capture (null = all)
- `make_data_scale`: Feature scaling strength

**Finetune config:**
- `lambda_neg_c`: Retain loss weight
- `num_epochs`: Training epochs (default 100)

**Inference config:**
- `finetune_model_path`: Path to fine-tuned checkpoint
- `num_inference_steps`: Denoising steps
- `method`: Erasure method (ConceptACP)
