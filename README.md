# ConceptACP: Precise Concept Erasure for Diffusion Models via Sparse Autoencoders

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-package%20manager-blue)](https://docs.astral.sh/uv/)

**Warning:** This repository includes model outputs that may be offensive.

ConceptACP (Concept Acupuncturist) is a research implementation for precise concept erasure in diffusion models. It uses Sparse Autoencoders (SAEs) to identify and modify specific features in attention layers, enabling fine-grained "unlearning" of unwanted concepts while preserving model quality.

## Key Features

- **Dual Model Support**: Works with both Stable Diffusion XL and SD 1.5
- **Automated Feature Selection**: No manual feature selection required
- **Memory Efficient**: 70-90% less VRAM with block-level fine-tuning
- **Fast Training**: 2-5x faster than naive approaches
- **Precise Control**: Target specific concepts without degrading general capabilities

## Quick Start

### Installation

Requires Python 3.13 and [uv package manager](https://docs.astral.sh/uv/):

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/yourusername/ConceptACP.git
cd ConceptACP
uv sync
```

### Basic Usage

**Erase nudity from SDXL-Turbo/SDXL (default):**

```bash
# 1. Automatically select discriminative features
uv run src/conceptacp/utils/feature_select.py unlearn_type=nudity

# 2. Generate training data
uv run src/conceptacp/utils/make_block_data.py unlearn_type=nudity

# 3. Fine-tune the model
uv run src/conceptacp/utils/finetune_block.py unlearn_type=nudity

# 4. Generate test images
uv run src/conceptacp/utils/inference.py unlearn_type=nudity
```

**Use SD 1.5 instead:**

```bash
# Just add model=sd15 to all commands
uv run src/conceptacp/utils/feature_select.py model=sd15 unlearn_type=nudity
uv run src/conceptacp/utils/make_block_data.py model=sd15 unlearn_type=nudity
uv run src/conceptacp/utils/finetune_block.py model=sd15 unlearn_type=nudity
uv run src/conceptacp/utils/inference.py model=sd15 unlearn_type=nudity
```

### Supported Concepts

Pre-configured erasure targets:

- `nudity`: Removes NSFW content (default: SDXL-Turbo `down.2.1` block, λ=0.2)
- `object_church`: Removes church objects (SDXL-Turbo `down.2.1` block, λ=1.8)
- `object_parachute`: Removes parachutes (SDXL-Turbo `down.2.1` block, λ=2.0)
- `style_vangogh`: Removes Van Gogh artistic style (SDXL-Turbo `up.0.1` block, λ=0.2)

Use any concept with `unlearn_type=<concept>`.

## How It Works

### Three-Phase Pipeline

```
┌─────────────────────┐
│  1. Feature Select  │  Identify which SAE features encode the target concept
│  (Automated)        │  Scores features by activation patterns
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. Data Generation │  Capture attention block inputs/outputs
│  (Block-Level)      │  Compute SAE-modified targets for "forgot" prompts
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  3. Fine-Tuning     │  Train target block to match modified outputs
│  (Efficient)        │  Dual loss: MSE on forgot + weighted MSE on retain
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  4. Inference       │  Generate images with the fine-tuned model
└─────────────────────┘
```

### Key Components

- **HookedStableDiffusionXLPipeline / HookedStableDiffusion15Pipeline**: Wraps diffusion pipelines with activation caching
- **SparseAutoencoder**: Decomposes activations into interpretable features (top-k sparse representations)
- **Block-Level Fine-Tuning**: Trains only the target attention block (~100MB) instead of full model (~7GB)

## Configuration System

Organized into four categories:

1. **Model configs** (`conf/model/`): `sdxl.yaml`, `sd15.yaml`
2. **Task configs** (`conf/task/`): `make_data.yaml`, `finetune.yaml`, `inference.yaml`, etc.
3. **Unlearn configs** (`conf/unlearn/`): `nudity.yaml`, `object_church.yaml`, etc.
4. **Paths config** (`conf/paths.yaml`): Centralized path definitions

### Common Overrides

```bash
# Switch model
model=sd15

# Switch concept
unlearn_type=object_church

# Use local model
model.model_path=/path/to/model

# Adjust training
batch_size=64 learning_rate=0.0001

# Reduce dataset size (capture every 5 timesteps)
capture_interval=5
```

## Advanced Usage

### Custom Concepts

1. Create prompt files in `data/train_prompts/`:
   - `myconcept_forgot.json`: Prompts containing the concept to erase
   - `myconcept_retain.json`: Prompts to preserve model quality

2. Run feature selection:
   ```bash
   uv run src/conceptacp/utils/feature_select.py unlearn_type=myconcept
   ```

3. Follow standard pipeline with `unlearn_type=myconcept`

### Timestep Capture (Reduce Dataset Size)

Capture only a subset of timesteps during data generation:

```bash
# Capture every 5 steps (80% reduction for 50-step inference)
uv run src/conceptacp/utils/make_block_data.py \
  unlearn_type=nudity \
  num_inference_steps=50 \
  capture_interval=5

# Capture specific timesteps
uv run src/conceptacp/utils/make_block_data.py \
  unlearn_type=nudity \
  'capture_timesteps=[0,10,20,30,40,49]'
```

See [docs/TIMESTEP_CAPTURE.md](docs/TIMESTEP_CAPTURE.md) for details.

### Hyperparameter Tuning

Generate data once, then iterate on training:

```bash
# Generate data (slow, do once)
uv run src/conceptacp/utils/make_block_data.py unlearn_type=nudity

# Try different lambda values (fast)
uv run src/conceptacp/utils/finetune_block.py unlearn_type=nudity lambda_neg_c=0.1
uv run src/conceptacp/utils/finetune_block.py unlearn_type=nudity lambda_neg_c=0.5
uv run src/conceptacp/utils/finetune_block.py unlearn_type=nudity lambda_neg_c=1.0
```

## Project Structure

```
ConceptACP/
├── src/conceptacp/
│   ├── sae/
│   │   ├── SAE/                    # Sparse autoencoder implementations
│   │   │   ├── sae.py             # ConceptACP dense SAE (SDXL)
│   │   │   └── sae_saeuron.py     # SAeUron top-k SAE (SD 1.5)
│   │   └── SDLens/                 # Hooked diffusion pipelines
│   │       ├── hooked_sdxl_pipeline.py
│   │       └── hooked_sd15_pipeline.py
│   └── utils/                      # Main scripts
│       ├── feature_select.py       # Automated feature selection
│       ├── make_block_data.py      # Block-level data generation
│       ├── finetune_block.py       # Efficient fine-tuning
│       └── inference.py            # Image generation
├── conf/                           # Hydra configuration
│   ├── model/                      # Model-specific configs
│   ├── task/                       # Task-specific configs
│   ├── unlearn/                    # Concept-specific configs
│   └── paths.yaml                  # Path definitions
├── data/
│   ├── train_prompts/              # Training prompts (forgot/retain)
│   └── finetune_config/            # Feature selections (auto-generated)
├── checkpoints/
│   ├── sdxl-ckpts/                 # SDXL SAE checkpoints
│   └── sd1.5-sae-ckpts/            # SD 1.5 SAE checkpoints
├── models/                         # Fine-tuned model checkpoints
└── docs/                           # Documentation
```

## Documentation

### User Guides

- **[CLAUDE.md](CLAUDE.md)**: Comprehensive project documentation (for Claude Code)
- **[docs/FEATURE_SELECTION.md](docs/FEATURE_SELECTION.md)**: Feature selection guide
- **[docs/TIMESTEP_CAPTURE.md](docs/TIMESTEP_CAPTURE.md)**: Timestep capture configuration

### Technical Documentation

- **[docs/EFFICIENT_FINETUNING.md](docs/EFFICIENT_FINETUNING.md)**: Block-level fine-tuning details
- **[docs/CONFIG_GUIDE.md](docs/CONFIG_GUIDE.md)**: Configuration system overview
- **[docs/CHECKPOINT_REORGANIZATION.md](docs/CHECKPOINT_REORGANIZATION.md)**: Checkpoint organization

## Requirements

- Python 3.13
- CUDA-capable GPU (recommended: 8GB+ VRAM)
- uv package manager

## Checkpoints

Pre-trained SAE checkpoints are required:

- **SDXL-Turbo/SDXL**: Download from [link] and place in `checkpoints/sdxl-ckpts/`
- **SD 1.5**: Download from [SAeUron](https://github.com/yourusername/SAeUron) and place in `checkpoints/sd1.5-sae-ckpts/`

See [checkpoints/README.md](checkpoints/README.md) for details.

## Citation

If you use ConceptACP in your research, please cite:

```bibtex
@article{conceptacp2024,
  title={Concept Acupuncturist: An Efficient and Precise Concept Erasure Method for Diffusion Models Via Sparse Autoencoders},
  author={Your Name},
  year={2024}
}
```

## License

[Your License Here]

## Acknowledgments

- SAE implementation inspired by [SDXL-unbox](https://github.com/surkovv/sdxl-unbox) and [SAeUron](https://github.com/yourusername/SAeUron)
- Built on [Diffusers](https://github.com/huggingface/diffusers) by Hugging Face
