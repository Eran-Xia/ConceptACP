# SAE Checkpoints Directory

This directory contains pre-trained Sparse Autoencoder (SAE) checkpoints for both SDXL and SD 1.5 models.

## Directory Structure

```
checkpoints/
├── sdxl-ckpts/                    # SDXL SAE checkpoints
│   ├── unet.down_blocks.2.attentions.1_k10_hidden5120_auxk256_bs4096_lr0.0001/
│   │   └── final/
│   │       ├── config.json        # SAE configuration
│   │       ├── state_dict.pth     # Model weights
│   │       ├── mean.pt            # Feature means
│   │       └── std.pt             # Feature stds
│   ├── unet.mid_block.attentions.0_k10_hidden5120_auxk256_bs4096_lr0.0001/
│   ├── unet.up_blocks.0.attentions.0_k10_hidden5120_auxk256_bs4096_lr0.0001/
│   └── unet.up_blocks.0.attentions.1_k10_hidden5120_auxk256_bs4096_lr0.0001/
└── sd1.5-sae-ckpts/              # SD 1.5 SAE checkpoints (from SAeUron)
    ├── unet.up_blocks.1.attentions.1/
    │   ├── cfg.json               # SAE configuration (JSON)
    │   └── sae.safetensors        # Model weights (safetensors format)
    └── unet.up_blocks.1.attentions.2/
        ├── cfg.json
        └── sae.safetensors
```

## Checkpoint Formats

### SDXL SAE Checkpoints (ConceptACP Format)

**Directory naming:**
```
{block_name}_k{k}_hidden{num_latents}_auxk{auxk}_bs{batch_size}_lr{learning_rate}/final/
```

**Files:**
- `config.json`: SAE hyperparameters (k, d_model, num_latents, etc.)
- `state_dict.pth`: PyTorch state dict with encoder/decoder weights
- `mean.pt`: Pre-computed feature means for normalization
- `std.pt`: Pre-computed feature standard deviations

**Parameters:**
- `d_model`: 1280 (attention dimension)
- `k`: 10 (top-k sparsity)
- `num_latents`: 5120 (expansion factor = 4)
- `auxk`: 256 (auxiliary loss k)

### SD 1.5 SAE Checkpoints (SAeUron Format)

**Directory naming:**
```
{block_name}/
```

**Files:**
- `cfg.json`: SAE configuration including d_in, expansion_factor, k, etc.
- `sae.safetensors`: Safetensors format with encoder, W_dec, b_dec weights

**Parameters:**
- `d_in`: 1280 (for up_blocks.1.attentions.*)
- `k`: 32 (top-k sparsity)
- `expansion_factor`: 16
- `num_latents`: 20480 (16 × 1280)
- `batch_topk`: true (batch-level top-k selection)

## Usage

### Configuration

**SDXL (default.yaml):**
```yaml
sae_checkpoints: ${work_dir}/checkpoints/sdxl-ckpts
```

**SD 1.5 (sd15.yaml):**
```yaml
sae_checkpoints: ${work_dir}/checkpoints/sd1.5-sae-ckpts
```

### Loading in Code

**SDXL:**
```python
from conceptacp.sae.SAE.sae import SparseAutoencoder

sae = SparseAutoencoder.load_from_disk(
    "checkpoints/sdxl-ckpts/unet.down_blocks.2.attentions.1_k10_hidden5120_auxk256_bs4096_lr0.0001/final"
)
```

**SD 1.5:**
```python
from conceptacp.sae.SAE.sae_saeuron import Sae

sae = Sae.load_from_disk(
    "checkpoints/sd1.5-sae-ckpts/unet.up_blocks.1.attentions.1",
    device="cuda",
    decoder=True
)
```

## Adding New Checkpoints

### For SDXL

1. Train SAE using ConceptACP's training code
2. Place in `sdxl-ckpts/` with proper naming convention
3. Add block mapping to `conf/data/default.yaml`:
   ```yaml
   code_to_block:
     your.code: unet.your.block.path
   ```

### For SD 1.5

1. Train SAE using SAeUron or compatible method
2. Ensure checkpoint has `cfg.json` and `sae.safetensors`
3. Place in `sd1.5-sae-ckpts/` with block name
4. Add block mapping to `conf/data/sd15.yaml`:
   ```yaml
   code_to_block:
     your.code: unet.your.block.path
   ```

## Checkpoint Source

- **SDXL checkpoints**: Trained for ConceptACP project
- **SD 1.5 checkpoints**: From SAeUron project (https://github.com/yourusername/SAeUron)

## Storage Requirements

- **SDXL checkpoint**: ~50 MB per block (uncompressed)
- **SD 1.5 checkpoint**: ~100 MB per block (safetensors format)

Total storage:
- SDXL (4 blocks): ~200 MB
- SD 1.5 (2 blocks): ~200 MB
- **Total**: ~400 MB

## Notes

- Checkpoints are model-specific and not interchangeable between SDXL and SD 1.5
- SAE d_model/d_in must match the attention block dimension
- Top-k (k) parameter affects sparsity and reconstruction quality
- Safetensors format (SD 1.5) provides faster loading and better security
