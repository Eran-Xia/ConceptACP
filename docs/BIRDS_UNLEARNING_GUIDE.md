# Birds Unlearning Complete Guide

This document provides the complete command workflow for Birds category unlearning.

## Prerequisites

- All dependencies installed (`uv sync`)
- SD 1.5 model available
- Classifier `checkpoints/style50_cls.pth` available

## Complete Workflow

### Step 0: Feature Selection (Automatic)

Automatically identify the most discriminative features and optimal attention block for Birds:

```bash
uv run src/conceptacp/utils/feature_select.py \
    model=sd15 \
    unlearn=uc_object_birds
```

**Output:**
- Auto-updates `feature_scales` and `code` in `src/conceptacp/utils/conf/unlearn/uc_object_birds.yaml`
- Generates feature config to `data/finetune_config/sd15_uc_object_birds.json`

**Optional Parameters:**
```bash
# Custom parameters
uv run src/conceptacp/utils/feature_select.py \
    model=sd15 \
    unlearn=uc_object_birds \
    threshold_divisor=3 \
    max_strength=-8.0 \
    num_inference_steps=4
```

---

### Step 1: Generate Block-Level Training Data

Generate efficient training data (pre-computed block input/output pairs):

```bash
uv run src/conceptacp/utils/make_block_data.py \
    model=sd15 \
    unlearn=uc_object_birds
```

**Output Locations:**
- Forgot data: `data/recons/sd15_uc_object_birds/forgot_*.tar.gz`
- Retain data: `data/recons/sd15_uc_object_birds/retain_*.tar.gz`

**Optional Parameters:**
```bash
# Use local model
uv run src/conceptacp/utils/make_block_data.py \
    model=sd15 \
    unlearn=uc_object_birds \
    model.model_path=/path/to/sd15

# Custom inference steps and capture interval (reduce data size)
uv run src/conceptacp/utils/make_block_data.py \
    model=sd15 \
    unlearn=uc_object_birds \
    num_inference_steps=50 \
    capture_interval=5
```

---

### Step 2: Training (Block-Level)

Use efficient block-level training (low memory usage, fast):

```bash
uv run src/conceptacp/utils/finetune_block.py \
    model=sd15 \
    unlearn=uc_object_birds
```

**Output Location:**
- Checkpoint: `models/sd15_uc_object_birds_99_0.2_1.0_block.pt`

**Optional Parameters:**
```bash
# Custom training parameters
uv run src/conceptacp/utils/finetune_block.py \
    model=sd15 \
    unlearn=uc_object_birds \
    batch_size=64 \
    learning_rate=0.0001 \
    num_epochs=100
```

---

### Step 3: Inference to Generate Test Images

Generate test images using the trained model:

```bash
uv run src/conceptacp/utils/inference.py \
    model=sd15 \
    unlearn=uc_object_birds
```

**Output Locations:**
- Forgot images: `data/output/sd15_uc_object_birds/forgot/`
  - `ConcepACP/`: ConceptACP model generated images
  - `ori/`: Original model generated images
- Unrelated images: `data/output/sd15_uc_object_birds/unrelated/`
  - `ConcepACP/{Architectures,Bears,Butterfly,...}/`: ConceptACP generated other classes
  - `ori/{Architectures,Bears,Butterfly,...}/`: Original model generated other classes

**Optional Parameters:**
```bash
# Custom inference parameters
uv run src/conceptacp/utils/inference.py \
    model=sd15 \
    unlearn=uc_object_birds \
    num_steps=4 \
    guidance_scale=0.0 \
    num_samples_per_prompt=10
```

---

### Step 4: Evaluate Unlearning Accuracy

Use classifier to evaluate unlearning effectiveness:

```bash
# Comparison evaluation (recommended)
uv run src/conceptacp/utils/compare_unlearn_accuracy.py \
    --input_dir data/output/sd15_uc_object_birds \
    --checkpoint_path checkpoints/style50_cls.pth \
    --target_class Birds

# Evaluate ConceptACP alone
uv run src/conceptacp/utils/evaluate_unlearn_accuracy.py \
    --input_dir data/output/sd15_uc_object_birds \
    --checkpoint_path checkpoints/style50_cls.pth \
    --target_class Birds \
    --model_name ConcepACP

# Evaluate original model alone
uv run src/conceptacp/utils/evaluate_unlearn_accuracy.py \
    --input_dir data/output/sd15_uc_object_birds \
    --checkpoint_path checkpoints/style50_cls.pth \
    --target_class Birds \
    --model_name ori
```

**Output Metrics:**
- **UA (Unlearning Accuracy)**: Higher is better
- **IRA (In-class Retain Accuracy)**: Higher is better

---

## Quick Execution (One-Click)

To execute all steps at once, use the following commands:

```bash
# 0. Feature selection
uv run src/conceptacp/utils/feature_select.py model=sd15 unlearn=uc_object_birds

# 1. Generate data
uv run src/conceptacp/utils/make_block_data.py model=sd15 unlearn=uc_object_birds

# 2. Train
uv run src/conceptacp/utils/finetune_block.py model=sd15 unlearn=uc_object_birds

# 3. Inference
uv run src/conceptacp/utils/inference.py model=sd15 unlearn=uc_object_birds

# 4. Evaluate
uv run src/conceptacp/utils/compare_unlearn_accuracy.py \
    --input_dir data/output/sd15_uc_object_birds \
    --checkpoint_path checkpoints/style50_cls.pth \
    --target_class Birds
```

---

## Troubleshooting

### 1. CUDA Out of Memory
```bash
# Reduce batch size
uv run src/conceptacp/utils/make_block_data.py model=sd15 unlearn=uc_object_birds batch_size=1
uv run src/conceptacp/utils/finetune_block.py model=sd15 unlearn=uc_object_birds batch_size=32
```

### 2. Data Generation Too Slow
```bash
# Use fewer inference steps or capture interval
uv run src/conceptacp/utils/make_block_data.py \
    model=sd15 \
    unlearn=uc_object_birds \
    num_inference_steps=4
```

### 3. Check Configuration
```bash
# View current config
cat src/conceptacp/utils/conf/unlearn/uc_object_birds.yaml

# View generated feature config
cat data/finetune_config/sd15_uc_object_birds.json
```

---

## Expected Results

Based on Bears test results, expected Birds unlearning:

- **Unlearning Accuracy (UA)**: Target > 80%
- **In-class Retain Accuracy (IRA)**: Maintain > 75%

If results are not satisfactory, try:
1. Adjust `lambda_neg_c` (in config, 0.2-2.0)
2. Increase training epochs
3. Adjust `max_strength` in feature selection
