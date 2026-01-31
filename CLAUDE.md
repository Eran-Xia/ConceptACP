# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ConceptACP (Concept Acupuncturist) is a research implementation for precise concept erasure in diffusion models using Sparse Autoencoders (SAEs). The project enables fine-tuning of Stable Diffusion XL and SD 1.5 models to "unlearn" specific concepts (e.g., nudity, specific objects, artistic styles) by identifying and modifying specific features in the model's attention layers.

**Warning:** This repository includes model outputs that may be offensive.

**NEW:** Supports both SDXL and SD 1.5 models with unified configuration system.

## Environment Setup

This project uses `uv` for dependency management (requires Python 3.13):

```bash
# Install dependencies
uv sync

# All Python commands should be run with uv
uv run src/conceptacp/utils/make_data.py
uv run src/conceptacp/utils/finetune.py
uv run src/conceptacp/utils/inference.py
```

## Core Architecture

### Three-Phase Pipeline

#### Original Approach

1. **Feature Selection** (Interactive): Run `notebooks/0x00_feature_select.ipynb` to identify which SAE features correspond to concepts you want to erase

2. **Data Generation**: `src/conceptacp/utils/make_data.py` generates fine-tuning datasets by:
   - Running prompts through the hooked diffusion pipeline
   - Extracting activations at specific attention blocks
   - Encoding activations through SAEs to get sparse feature maps
   - Scaling selected features (from finetune_config)
   - Saving scaled reconstructions as WebDataset tar.gz files

3. **Fine-tuning**: `src/conceptacp/utils/finetune.py` trains specific attention layers to match the modified reconstructions while preserving behavior on retain data

4. **Inference**: `src/conceptacp/utils/inference.py` generates test images using the fine-tuned model

#### Efficient Block-Level Approach (RECOMMENDED)

**NEW: This approach provides 70-90% memory reduction and 2-5x faster training while producing identical results.**

**0. Feature Selection** (Automated): `src/conceptacp/utils/feature_select.py` automatically identifies discriminative features
   - Runs forgot/retain prompts through pipeline
   - Scores features by comparing activations (normalized mean difference)
   - Selects block with highest discrimination
   - Generates feature configs automatically
   - **Works for both SDXL and SD 1.5**

1. **Block-Level Data Generation**: `src/conceptacp/utils/make_block_data.py` generates efficient training data by:
   - Running prompts through the pipeline ONCE to capture block inputs
   - Recording inputs to the target attention block (hidden_states, encoder_hidden_states, etc.)
   - For FORGOT prompts: Computing SAE-modified target outputs
   - For RETAIN prompts: Using original block outputs
   - Saving (input, target_output) pairs for each timestep as WebDataset
   - **Key advantage**: Captures data for ALL timesteps, not just final state

2. **Efficient Block Fine-tuning**: `src/conceptacp/utils/finetune_block.py` trains in isolation:
   - Loads ONLY the target attention block (~100MB vs ~7GB)
   - Direct forward pass through block without full pipeline
   - Trains on pre-computed input/output pairs
   - **Key advantage**: No need to load VAE, text encoders, or rest of UNet during training

3. **Inference**: Same as original - `src/conceptacp/utils/inference.py`

**Why this works**: Since only the target block has gradients enabled during fine-tuning, we only need to compute gradients through that block. Pre-computing its inputs allows training in complete isolation with identical results.

See `docs/EFFICIENT_FINETUNING.md` for comprehensive documentation.

### Key Components

**HookedStableDiffusionXLPipeline** (`src/conceptacp/sae/SDLens/`):
- Wraps Stable Diffusion XL pipeline with activation caching capabilities
- `run_with_cache()`: Captures inputs/outputs at specified attention blocks during inference
- `run_with_hooks_grad()`: Enables gradient flow through custom hooks for fine-tuning
- Supports gradient-based training on specific layers
- **Note:** Only used for training/data collection. Inference uses standard diffusers pipelines.

**SparseAutoencoder** (`src/conceptacp/sae/SAE/`):
- Top-K autoencoder for decomposing attention activations into interpretable features
- Pre-trained SAEs for different attention blocks stored in `checkpoints/sdxl-ckpts/` (SDXL) and `checkpoints/sd1.5-sae-ckpts/` (SD 1.5)
- Each checkpoint includes encoder/decoder weights, biases, and feature means
- SDXL Format: `{block}_k10_hidden5120_auxk256_bs4096_lr0.0001/final/`
- SD 1.5 Format (from SAeUron): `{block}/cfg.json` and `{block}/sae.safetensors`

**Configuration System** (Hydra-based, redesigned for clarity):

The config system is organized into four categories:

1. **Model configs** (`conf/model/`): Model-specific settings
   - `sdxl.yaml`: SDXL model path, architecture params, SAE checkpoint locations
   - `sd15.yaml`: SD 1.5 model path, architecture params, SAE checkpoint locations
   - Override with: `model=sd15` or `model=sdxl`

2. **Task configs** (`conf/task/`): Task-specific hyperparameters
   - `make_data.yaml`: Data generation settings (num_inference_steps, etc.)
   - `make_block_data.yaml`: Block-level data generation settings
   - `finetune.yaml`: Training hyperparameters (batch_size, lr, epochs, etc.)
   - `finetune_block.yaml`: Block-level training hyperparameters
   - `inference.yaml`: Inference settings (num_steps, checkpoint path)
   - `feature_select.yaml`: Feature selection parameters

3. **Unlearn configs** (`conf/unlearn/`): Concept-specific configurations
   - `nudity.yaml`: Target block, lambda_neg_c, feature scales for nudity erasure
   - `object_church.yaml`: Settings for church object erasure
   - `object_parachute.yaml`: Settings for parachute object erasure
   - `style_vangogh.yaml`: Settings for Van Gogh style erasure
   - Override with: `unlearn=object_church`, `unlearn=style_vangogh`, etc.

4. **Paths config** (`conf/paths.yaml`): Centralized path definitions
   - All data paths, checkpoint locations, output directories

**Per-script configs**: Each script has its own top-level config that composes the above:
- `make_data.yaml`, `make_block_data.yaml`, `finetune.yaml`, `finetune_block.yaml`, `inference.yaml`, `feature_select.yaml`
- These configs flatten the composed groups for backward compatibility (cfg.model_path instead of cfg.model.model_path)

**Common overrides:**
- Switch model: `model=sd15` (default: sdxl)
- Switch concept: `unlearn=object_church` (default: nudity)
- Change model path: `model.model_path=/path/to/model` (affects cfg.model_path after flattening)
- Change batch size: `batch_size=64`
- Change learning rate: `learning_rate=0.0001`

### Block/Layer Mapping

The code uses shorthand codes for attention blocks:
- `down.2.1` → `unet.down_blocks.2.attentions.1` (most common for concept erasure)
- `mid.0` → `unet.mid_block.attentions.0`
- `up.0.1` → `unet.up_blocks.0.attentions.1` (used for style_vangogh)
- `up.0.0` → `unet.up_blocks.0.attentions.0`

These mappings are defined in `code_to_block` in configs.

### Data Organization

- `data/train_prompts/`: Training prompts split into `forgot.json` (concepts to erase) and `retain.json` (concepts to preserve)
- `data/finetune_config/`: Feature selections and scaling strengths for each unlearn type
- `checkpoints/`: Pre-trained SAE checkpoints
  - `sdxl-ckpts/`: SDXL SAE checkpoints for each attention block
  - `sd1.5-sae-ckpts/`: SD 1.5 SAE checkpoints (from SAeUron)
- `models/`: Fine-tuned attention layer checkpoints

### Unlearn Types

The codebase supports multiple erasure targets with different hyperparameters:
- `nudity`: Uses `down.2.1` block, λ_neg_c=0.2
- `object_church`: Uses `down.2.1` block, λ_neg_c=1.8
- `object_parachute`: Uses `down.2.1` block, λ_neg_c=2.0
- `style_vangogh`: Uses `up.0.1` block, λ_neg_c=0.2

## Common Commands

### Complete Workflow (Recommended)

**SDXL:**
```bash
# 0. Feature selection (automatic)
uv run src/conceptacp/utils/feature_select.py unlearn_type=nudity

# 1. Generate block-level training data
uv run src/conceptacp/utils/make_block_data.py unlearn_type=nudity

# 2. Train efficiently
uv run src/conceptacp/utils/finetune_block.py unlearn_type=nudity

# 3. Run inference
uv run src/conceptacp/utils/inference.py unlearn_type=nudity
```

**SD 1.5:**
```bash
# 0. Feature selection (automatic)
uv run src/conceptacp/utils/feature_select.py model=sd15 unlearn_type=nudity

# 1. Generate block-level training data
uv run src/conceptacp/utils/make_block_data.py model=sd15 unlearn_type=nudity

# 2. Train efficiently
uv run src/conceptacp/utils/finetune_block.py model=sd15 unlearn_type=nudity

# 3. Run inference
uv run src/conceptacp/utils/inference.py model=sd15 unlearn_type=nudity
```

### Feature Selection Options

**Process all unlearn types:**
```bash
# SDXL
uv run src/conceptacp/utils/feature_select.py

# SD 1.5
uv run src/conceptacp/utils/feature_select.py model=sd15
```

**Custom parameters:**
```bash
uv run src/conceptacp/utils/feature_select.py \
    unlearn_type=style_vangogh \
    threshold_divisor=3 \
    max_strength=-8.0 \
    num_inference_steps=4
```

See `docs/FEATURE_SELECTION.md` for comprehensive documentation.

### Efficient Block-Level Approach (RECOMMENDED)

**70-90% less memory, 2-5x faster training, identical results**

```bash
# Generate block-level training data
uv run src/conceptacp/utils/make_block_data.py unlearn_type=nudity

# Train efficiently
uv run src/conceptacp/utils/finetune_block.py unlearn_type=nudity

# Run inference
uv run src/conceptacp/utils/inference.py unlearn_type=nudity

# Use local model
uv run src/conceptacp/utils/make_block_data.py model.model_path=/path/to/sdxl unlearn_type=nudity

# Use different unlearn concept
uv run src/conceptacp/utils/make_block_data.py unlearn=object_church

# Custom batch size for training
uv run src/conceptacp/utils/finetune_block.py unlearn=style_vangogh batch_size=64
```

### Timestep Capture Configuration

**NEW: Control which timesteps to capture during data generation**

When using multi-step inference, you can reduce dataset size and training time by capturing only a subset of timesteps:

**Method 1: Capture at intervals (Simplest, recommended)**
```bash
# Capture every 5 steps (80% reduction for 50-step inference)
uv run src/conceptacp/utils/make_block_data.py \
  unlearn_type=nudity \
  num_inference_steps=50 \
  capture_interval=5

# Capture every 10 steps (90% reduction)
uv run src/conceptacp/utils/make_block_data.py \
  unlearn_type=nudity \
  num_inference_steps=50 \
  capture_interval=10
```

**Method 2: Capture specific timesteps (Fine-grained control)**
```bash
# Capture only specific timesteps
uv run src/conceptacp/utils/make_block_data.py \
  unlearn_type=nudity \
  num_inference_steps=50 \
  'capture_timesteps=[0,10,20,30,40,49]'
```

**Method 3: Capture all (Default)**
```bash
# Default: captures all timesteps (single-step is fastest)
uv run src/conceptacp/utils/make_block_data.py unlearn_type=nudity
```

**Priority:** `capture_interval` > `capture_timesteps` > default (all)

See `docs/TIMESTEP_CAPTURE.md` for detailed documentation and examples.

### Original Approach (Full Pipeline)

```bash
uv run src/conceptacp/utils/make_data.py unlearn_type=nudity
uv run src/conceptacp/utils/finetune.py unlearn_type=nudity
uv run src/conceptacp/utils/inference.py unlearn_type=nudity
```

## Implementation Details

- Only the target attention layer is trainable (dual loss: MSE on forgot + weighted MSE on retain)
- Checkpoints: `{model_version}_{unlearn_type}_{epoch}_{lambda_neg_c}_{make_data_scale}_block.pt` (attention block only)
  - Example: `sd15_style_vangogh_99_0.2_1.0_block.pt` or `sdxl_nudity_99_0.2_1.0_block.pt`
- WebDataset format: tar.gz with `recons.pth` and `gen_args.json` (original) or `input.pth`, `target.pth`, `metadata.json` (block-level)
- Base model: `stabilityai/sdxl-turbo`, requires CUDA GPU
