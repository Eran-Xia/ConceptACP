# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ConceptACP (Concept Acupuncturist) is a research implementation for precise concept erasure in diffusion models using Sparse Autoencoders (SAEs). The project enables fine-tuning of Stable Diffusion XL models to "unlearn" specific concepts (e.g., nudity, specific objects, artistic styles) by identifying and modifying specific features in the model's attention layers.

**Warning:** This repository includes model outputs that may be offensive.

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

1. **Feature Selection** (Same as original): Use `notebooks/0x00_feature_select.ipynb`

2. **Block-Level Data Generation**: `src/conceptacp/utils/make_block_data.py` generates efficient training data by:
   - Running prompts through the pipeline ONCE to capture block inputs
   - Recording inputs to the target attention block (hidden_states, encoder_hidden_states, etc.)
   - For FORGOT prompts: Computing SAE-modified target outputs
   - For RETAIN prompts: Using original block outputs
   - Saving (input, target_output) pairs for each timestep as WebDataset
   - **Key advantage**: Captures data for ALL timesteps, not just final state

3. **Efficient Block Fine-tuning**: `src/conceptacp/utils/finetune_block.py` trains in isolation:
   - Loads ONLY the target attention block (~100MB vs ~7GB)
   - Direct forward pass through block without full pipeline
   - Trains on pre-computed input/output pairs
   - **Key advantage**: No need to load VAE, text encoders, or rest of UNet during training

4. **Inference**: Same as original - `src/conceptacp/utils/inference.py`

**Why this works**: Since only the target block has gradients enabled during fine-tuning, we only need to compute gradients through that block. Pre-computing its inputs allows training in complete isolation with identical results.

See `docs/EFFICIENT_FINETUNING.md` for comprehensive documentation.

### Key Components

**HookedStableDiffusionXLPipeline** (`src/conceptacp/sae/SDLens/`):
- Wraps Stable Diffusion XL pipeline with activation caching capabilities
- `run_with_cache()`: Captures inputs/outputs at specified attention blocks during inference
- `run_with_hooks_grad()`: Enables gradient flow through custom hooks for fine-tuning
- Supports gradient-based training on specific layers

**SparseAutoencoder** (`src/conceptacp/sae/SAE/sae.py`):
- Top-K autoencoder for decomposing attention activations into interpretable features
- Pre-trained SAEs for different attention blocks stored in `checkpoints/`
- Each checkpoint includes encoder/decoder weights, biases, and feature means
- Format: `unet.{block}_k10_hidden5120_auxk256_bs4096_lr0.0001/final/`

**Configuration System** (Hydra-based):
- **`src/conceptacp/utils/conf/data/default.yaml`**: Base config with **unified model path** (single source of truth), block mappings, unlearn types, and all data paths
- `src/conceptacp/utils/conf/finetune.yaml`: Fine-tuning specific settings (imports model_path from data config)
- `src/conceptacp/utils/conf/inference.yaml`: Inference settings and checkpoint paths (imports model_path from data config)
- `data/finetune_config/{unlearn_type}.json`: Per-concept feature selections and scaling strengths

**Model Path Configuration:**
- ✨ **Unified location**: Model path is defined ONLY in `conf/data/default.yaml`
- Scripts using data config directly (`make_data.py`, `make_block_data.py`): access via `cfg.model_path`
- Scripts using composed configs (`finetune*.py`, `inference.py`): access via `cfg.data.model_path`
- Override syntax:
  - Data scripts: `model_path=/path/to/model`
  - Training/inference scripts: `data.model_path=/path/to/model`
- See `docs/CONFIG_GUIDE.md` for detailed configuration documentation

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
- `data/recons/`: Generated WebDataset files containing modified reconstructions
- `checkpoints/`: Pre-trained SAE checkpoints for each attention block
- `models/`: Fine-tuned attention layer checkpoints

### Unlearn Types

The codebase supports multiple erasure targets with different hyperparameters:
- `nudity`: Uses `down.2.1` block, λ_neg_c=0.2
- `object_church`: Uses `down.2.1` block, λ_neg_c=1.8
- `object_parachute`: Uses `down.2.1` block, λ_neg_c=2.0
- `style_vangogh`: Uses `up.0.1` block, λ_neg_c=0.2

## Common Commands

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
uv run src/conceptacp/utils/make_block_data.py data.model_path=/path/to/sdxl unlearn_type=nudity
```

### Original Approach (Full Pipeline)

```bash
uv run src/conceptacp/utils/make_data.py unlearn_type=nudity
uv run src/conceptacp/utils/finetune.py unlearn_type=nudity
uv run src/conceptacp/utils/inference.py unlearn_type=nudity
```

## Implementation Details

- Only the target attention layer is trainable (dual loss: MSE on forgot + weighted MSE on retain)
- Checkpoints: `{unlearn_type}_{epoch}_{lambda_neg_c}_{make_data_scale}.pt` (attention block only)
- WebDataset format: tar.gz with `recons.pth` and `gen_args.json` (original) or `input.pth`, `target.pth`, `metadata.json` (block-level)
- Base model: `stabilityai/sdxl-turbo`, requires CUDA GPU
