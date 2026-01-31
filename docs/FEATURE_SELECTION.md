# Feature Selection Guide

## Overview

The automated feature selection script (`src/conceptacp/utils/feature_select.py`) identifies SAE features that distinguish target concepts (forgot) from retain concepts. It works for both SDXL and SD 1.5 models.

## How It Works

1. **Load Models**: Loads the diffusion pipeline and SAE models for specified blocks
2. **Collect Activations**: Runs prompts through the pipeline and extracts SAE activations
3. **Score Features**: Compares forgot vs retain activations using normalized mean difference
4. **Select Block**: Chooses the block with the highest discriminative score
5. **Select Features**: Picks features above a threshold (top_score / 5)
6. **Assign Strengths**: Scales strengths proportionally to scores (max = -5.0)

## Scoring Method

The script uses a normalized activation difference:

```
score(feature) = μ_forgot(feature) / Σμ_forgot - μ_retain(feature) / Σμ_retain
```

Where:
- `μ_forgot`: Mean activation for forgot prompts
- `μ_retain`: Mean activation for retain prompts
- Higher score = feature activates more for target concept

## Usage

### Basic Usage (All Unlearn Types, SDXL)

```bash
uv run src/conceptacp/utils/feature_select.py
```

This will:
- Process all unlearn types found in `data/train_prompts/{forgot,retain}.json`
- Use SDXL model (default)
- Generate configs in `data/finetune_config/{unlearn_type}.json`

### SD 1.5 Feature Selection

```bash
uv run src/conceptacp/utils/feature_select.py data=sd15
```

This will:
- Use SD 1.5 model and SAeUron SAEs
- Generate configs in `data/finetune_config/sd15/{unlearn_type}.json`

### Specific Unlearn Type

```bash
uv run src/conceptacp/utils/feature_select.py unlearn_type=nudity
```

### Custom Parameters

```bash
uv run src/conceptacp/utils/feature_select.py \
    unlearn_type=style_vangogh \
    threshold_divisor=3 \
    max_strength=-8.0 \
    num_inference_steps=4
```

Parameters:
- `threshold_divisor`: Higher = fewer features (default: 5)
- `max_strength`: Maximum negative scaling (default: -5.0)
- `num_inference_steps`: Denoising steps for activation collection (default: 1)
- `display_top_k`: Number of top features to show per block (default: 10)
- `device`: CUDA device (default: cuda:0)

### Use Local Model

```bash
uv run src/conceptacp/utils/feature_select.py \
    model_path=/path/to/local/sdxl
```

## Configuration Files

### Input: Prompt Files

**`data/train_prompts/forgot.json`** - Concepts to erase:
```json
{
    "nudity": {
        "girl": ["A naked girl...", ...],
        "boy": ["A naked boy...", ...]
    },
    "style_vangogh": {
        "landscape": ["A Van Gogh style landscape...", ...]
    }
}
```

**`data/train_prompts/retain.json`** - Concepts to preserve:
```json
{
    "nudity": {
        "girl": ["A girl...", ...],
        "boy": ["A boy...", ...]
    }
}
```

### Output: Feature Configs

**`data/finetune_config/nudity.json`** (SDXL):
```json
{
    "unlearn_type": "nudity",
    "block": ["down.2.1"],
    "finetune_feature": [4913, 4001, 3256],
    "finetune_strength": [-5.0, -3.8, -2.1]
}
```

**`data/finetune_config/sd15/nudity.json`** (SD 1.5):
```json
{
    "unlearn_type": "nudity",
    "block": ["up.1.1"],
    "finetune_feature": [1234, 5678, 9012],
    "finetune_strength": [-5.0, -4.2, -3.0]
}
```

## Example Workflow

### 1. Prepare Prompts

Create or modify `data/train_prompts/{forgot,retain}.json` with your target and retain concepts.

### 2. Run Feature Selection

**For SDXL:**
```bash
uv run src/conceptacp/utils/feature_select.py unlearn_type=my_concept
```

**For SD 1.5:**
```bash
uv run src/conceptacp/utils/feature_select.py data=sd15 unlearn_type=my_concept
```

### 3. Review Results

Check the generated config file:
```bash
cat data/finetune_config/my_concept.json
# or
cat data/finetune_config/sd15/my_concept.json
```

The script will show:
- Top 10 features for each block
- Selected block and its score
- Number of features selected
- Feature indices and their strengths

### 4. Generate Training Data

```bash
# SDXL
uv run src/conceptacp/utils/make_block_data.py unlearn_type=my_concept

# SD 1.5
uv run src/conceptacp/utils/make_block_data.py data=sd15 unlearn_type=my_concept
```

### 5. Train Model

```bash
# SDXL
uv run src/conceptacp/utils/finetune_block.py unlearn_type=my_concept

# SD 1.5
uv run src/conceptacp/utils/finetune_block.py data=sd15 unlearn_type=my_concept
```

## Output Example

```
================================================================================
ConceptACP Feature Selection
================================================================================
Model version: sdxl
Model path: stabilityai/sdxl-turbo
SAE checkpoints: /home/xiahyr/ConceptACP/checkpoints
Device: cuda:0
Blocks to analyze: ['down.2.1', 'mid.0', 'up.0.1', 'up.0.0']

Loading SDXL pipeline from stabilityai/sdxl-turbo...
Loading SAE models for SDXL...
  Loaded SDXL SAE for down.2.1 (d_model=1280)
  Loaded SDXL SAE for mid.0 (d_model=1280)
  Loaded SDXL SAE for up.0.1 (d_model=1280)
  Loaded SDXL SAE for up.0.0 (d_model=1280)

============================================================
Processing unlearn_type: nudity
============================================================
Collecting retain activations...
Retain girl: 100%|██████████| 10/10
Retain boy: 100%|██████████| 10/10
...

Collecting forgot activations...
Forgot girl: 100%|██████████| 10/10
Forgot boy: 100%|██████████| 10/10
...

Scoring features across blocks...

  Block: down.2.1
    Top score: 0.008234
    Top 10 indices: [4913, 4001, 3256, 2145, 1982, ...]
    Top 10 scores: [0.008234, 0.006245, 0.004123, ...]

  Block: mid.0
    Top score: 0.003145
    Top 10 indices: [2341, 1234, 5678, ...]
    Top 10 scores: [0.003145, 0.002876, ...]

✓ Selected block: down.2.1 (score: 0.008234)

✓ Selected 3 features (threshold: 0.001647)
  Features: [4913, 4001, 3256]
  Strengths: [-5.0, -3.8, -2.5]

✓ Config saved to data/finetune_config/nudity.json

================================================================================
Feature Selection Complete!
================================================================================
nudity               → down.2.1   (3 features)

Next steps:
  1. Review generated configs in: data/finetune_config
  2. Generate training data: uv run src/conceptacp/utils/make_block_data.py
  3. Train model: uv run src/conceptacp/utils/finetune_block.py
```

## Troubleshooting

### Memory Issues

If you run out of memory:

1. **Reduce batch size** (process prompts individually):
   - The script already processes prompts one at a time
   - Consider reducing number of prompts in input files

2. **Use fewer blocks**:
   ```bash
   # Create a custom config with fewer blocks
   uv run src/conceptacp/utils/feature_select.py \
       code_to_block='{down.2.1: unet.down_blocks.2.attentions.1}'
   ```

3. **Use smaller inference steps**:
   ```bash
   uv run src/conceptacp/utils/feature_select.py num_inference_steps=1
   ```

### SAE Not Found

**SDXL:**
- Ensure SAE checkpoints exist in `checkpoints/sdxl-ckpts/`
- Format: `{block_name}_k10_hidden5120_auxk256_bs4096_lr0.0001/final/`

**SD 1.5:**
- Ensure SAeUron checkpoints exist at `checkpoints/sd1.5-sae-ckpts/`
- Format: `{block_name}/cfg.json` and `{block_name}/sae.safetensors`
- Or update `sae_checkpoints` in `conf/data/sd15.yaml`

### No Features Selected

If no features are selected (score below threshold):

1. **Lower threshold**:
   ```bash
   uv run src/conceptacp/utils/feature_select.py threshold_divisor=10
   ```

2. **Check prompts**: Ensure forgot and retain prompts are sufficiently different

3. **Try different blocks**: Some concepts may be represented better in different layers

### Different Results Between Runs

This is normal! Slight variations can occur due to:
- Random seed in generation
- Numerical precision
- GPU non-determinism

To get consistent results, the script uses a fixed seed (42) for generation.

## Advanced: Manual Feature Selection

If you want to manually select features:

1. Run the script to see scores:
   ```bash
   uv run src/conceptacp/utils/feature_select.py unlearn_type=my_concept
   ```

2. Note the top features and their scores

3. Manually create/edit the config file:
   ```json
   {
       "unlearn_type": "my_concept",
       "block": ["down.2.1"],
       "finetune_feature": [1234, 5678],  // Your chosen features
       "finetune_strength": [-5.0, -3.0]  // Your chosen strengths
   }
   ```

## Tips for Good Results

1. **Quality Prompts**: Use diverse, representative prompts for both forgot and retain sets
2. **Balanced Sets**: Use similar numbers of prompts for forgot and retain
3. **Minimal Difference**: Retain prompts should differ minimally from forgot prompts
   - Example: "naked girl" (forgot) vs "girl" (retain)
4. **Multiple Concepts**: Include various sub-concepts (girl, boy, woman, man for nudity)
5. **Test Incrementally**: Start with one unlearn_type, validate, then expand

## Comparison: SDXL vs SD 1.5

| Aspect | SDXL | SD 1.5 |
|--------|------|--------|
| **SAE Format** | ConceptACP (dense) | SAeUron (top-k) |
| **Common Blocks** | down.2.1 | up.1.1, up.1.2 |
| **Num Latents** | 5120 | 20480 (16x1280) |
| **Typical Features** | 2-5 selected | 2-5 selected |
| **Config Location** | `data/finetune_config/` | `data/finetune_config/sd15/` |

Both models use the same scoring algorithm, just adapted for different SAE formats.
