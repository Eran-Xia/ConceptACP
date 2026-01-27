# Efficient Block-Level Fine-Tuning

Optimized training approach: **70-90% memory reduction** (7-10GB → 0.5-2GB), **2-5x faster**, **identical results**.

## Architecture

### Data Generation (`make_block_data.py`)

Pre-computes block input/output pairs for each timestep:

```python
# For each training sample at each timestep:
1. Capture block inputs (hidden_states, encoder_hidden_states, attention_mask)
2. For FORGOT: Compute SAE-modified target outputs
3. For RETAIN: Use original block outputs
4. Save (input, target) pairs as WebDataset
```

**Data format** (`.tar.gz`):
```
sample_001_t0/
├── input.pth       # Block inputs dict
├── target.pth      # Target output tensor
└── metadata.json   # Prompt, timestep, type
```

### Fine-Tuning (`finetune_block.py`)

Trains isolated attention block:

```python
block = extract_attention_block(model, target_code)  # Load ~100MB
for inputs, targets in dataloader:
    output = block(**inputs)
    loss = mse_loss(output, targets)
    loss.backward()
```

**No hooks, no full pipeline - direct optimization.**

## Usage

```bash
# Generate data
uv run src/conceptacp/utils/make_block_data.py unlearn_type=nudity

# Train block
uv run src/conceptacp/utils/finetune_block.py data.unlearn_type=nudity

# Run inference
uv run src/conceptacp/utils/inference.py unlearn_type=nudity
```

## Why It Works

Only the target block has gradients enabled, so we only need to train that block. Pre-computing its inputs enables isolated training with identical results.

## Configuration

Key parameters in configs:
- `capture_timesteps`: null (all) or list of indices
- `lambda_neg_c`: Erasure vs preservation weight
- `make_data_scale`: Feature scaling strength

## Troubleshooting

**OOM during data generation:** Still requires full pipeline. Use machine with more VRAM or distribute with `device_map="balanced"`.

**Missing files during training:** Run `make_block_data.py` first.

**Different results:** Verify same seed, epochs, `make_data_scale`, and `lambda_neg_c` values.
