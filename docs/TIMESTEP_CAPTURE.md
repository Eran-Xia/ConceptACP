# Timestep Capture Configuration

This document explains the different methods for capturing timesteps during data generation.

## Overview

When generating training data with `make_block_data.py`, you can control which timesteps to capture. This is useful for:
- **Reducing dataset size** when using many inference steps
- **Focusing on specific denoising stages** (early, middle, or late)
- **Faster training** with smaller datasets

## Configuration Methods

### Method 1: Capture All Timesteps (Default)

Captures data at every timestep. Best for single-step inference or when you need complete coverage.

```yaml
# conf/task/make_block_data.yaml
num_inference_steps: 1
capture_timesteps: null  # or "all"
capture_interval: null
```

**Command:**
```bash
uv run python src/conceptacp/utils/make_block_data.py model=sdxl unlearn_type=nudity
```

**Result:** With 1 step → captures 1 timestep

---

### Method 2: Capture at Regular Intervals (Recommended)

Captures every N steps. Simplest method for multi-step inference.

```yaml
# conf/task/make_block_data.yaml
num_inference_steps: 50
capture_interval: 5  # Capture every 5 steps
```

**Command:**
```bash
uv run python src/conceptacp/utils/make_block_data.py \
  model=sdxl \
  unlearn_type=nudity \
  num_inference_steps=50 \
  capture_interval=5
```

**Result:** Captures timesteps [0, 5, 10, 15, 20, 25, 30, 35, 40, 45] = 10 timesteps (80% reduction)

**Examples:**
- `capture_interval=1` → all timesteps (same as Method 1)
- `capture_interval=5` → every 5th timestep
- `capture_interval=10` → every 10th timestep (90% reduction)

---

### Method 3: Capture Specific Timesteps (Fine-Grained Control)

Specify exact timestep indices. Best when you need precise control.

```yaml
# conf/task/make_block_data.yaml
num_inference_steps: 50
capture_timesteps: [0, 10, 20, 30, 40, 49]  # Specific indices
```

**Command:**
```bash
uv run python src/conceptacp/utils/make_block_data.py \
  model=sdxl \
  unlearn_type=nudity \
  num_inference_steps=50 \
  'capture_timesteps=[0,10,20,30,40,49]'
```

**Result:** Captures exactly timesteps [0, 10, 20, 30, 40, 49] = 6 timesteps

---

## Priority Order

When multiple parameters are set, the priority is:

1. **`capture_interval`** (highest priority)
2. **`capture_timesteps`** (specific list)
3. **Default** (all timesteps)

## Usage Examples

### Example 1: Quick Training with 50-Step Inference

```bash
# Capture every 10 steps (5 timesteps total)
uv run python src/conceptacp/utils/make_block_data.py \
  model=sdxl \
  unlearn_type=nudity \
  num_inference_steps=50 \
  capture_interval=10
```

**Output filename:** `down.2.1_nudity_1.0_block_steps50_interval10.tar.gz`

### Example 2: Focus on Early Denoising

```bash
# Capture first 5 timesteps only
uv run python src/conceptacp/utils/make_block_data.py \
  model=sdxl \
  unlearn_type=nudity \
  num_inference_steps=50 \
  'capture_timesteps=[0,1,2,3,4]'
```

**Output filename:** `down.2.1_nudity_1.0_block_steps50_t0-1-2-3-4.tar.gz`

### Example 3: Single-Step Inference (Fastest)

```bash
# Default: capture all (1 timestep)
uv run python src/conceptacp/utils/make_block_data.py \
  model=sdxl \
  unlearn_type=nudity \
  num_inference_steps=1
```

**Output filename:** `down.2.1_nudity_1.0_block_steps1.tar.gz`

---

## Performance Impact

| Configuration | Timesteps | Data Size | Training Time |
|---------------|-----------|-----------|---------------|
| 50 steps, all | 50 | 100% | 100% |
| 50 steps, interval=5 | 10 | ~20% | ~20% |
| 50 steps, interval=10 | 5 | ~10% | ~10% |
| 1 step, all | 1 | ~2% | ~2% |

**Note:** Single-step inference (num_inference_steps=1) is the fastest option and is recommended for most use cases, as shown in the original paper.

---

## Tips

1. **Start with single-step** (`num_inference_steps=1`): Fastest and often sufficient
2. **Use intervals for multi-step**: `capture_interval=5` or `capture_interval=10` for 50-step inference
3. **Avoid capturing too many**: More timesteps = larger datasets = longer training
4. **Check data size**: Generated tar.gz files show data size before training

---

## Troubleshooting

**Q: Which method should I use?**
- For most cases: Use single-step inference (`num_inference_steps=1`)
- For quality comparison: Use intervals with 50-step inference (`capture_interval=5`)

**Q: Can I change capture settings after data generation?**
- No, you need to regenerate data with the desired settings

**Q: What if I set both capture_interval and capture_timesteps?**
- `capture_interval` takes precedence and `capture_timesteps` is ignored
