# UnlearnCanvas Dataset Adaptations

## Overview

This document summarizes the changes made to ConceptACP to support the UnlearnCanvas dataset. UnlearnCanvas is a concept erasure benchmark with 20 object classes. Tasks are identified by the `uc_` prefix (e.g., `uc_object_bears`, `uc_object_birds`).

## Key Files Modified

### 1. feature_select.py (Lines 378-401)

**Purpose**: Use different retain prompts for feature selection in UnlearnCanvas tasks.

**Changes**:
```python
# Check if this is an UnlearnCanvas task (starts with 'uc_')
is_unlearncanvas = cfg.get('unlearn_type', '').startswith('uc_')

if is_unlearncanvas:
    # For UnlearnCanvas: use retain_featureselect.json for feature selection
    logger.info("UnlearnCanvas task detected - using retain_featureselect.json")
    retain_featureselect_file = pathlib.Path(cfg.work_dir) / "data/train_prompts/retain_featureselect.json"
    if retain_featureselect_file.exists():
        with open(retain_featureselect_file, 'r') as f:
            retain_prompts = json.load(f)
        logger.info(f"  Loaded retain prompts from {retain_featureselect_file}")
    else:
        logger.warning(f"retain_featureselect.json not found, falling back to retain.json")
        with open(cfg.retain_prompts, 'r') as f:
            retain_prompts = json.load(f)
else:
    with open(cfg.retain_prompts, 'r') as f:
        retain_prompts = json.load(f)
```

**Why**: UnlearnCanvas uses specialized retain prompts during feature selection that replace the target concept with "Something" (e.g., "A cat playing" → "Something playing"). This helps identify features that are truly discriminative for the target concept.

### 2. inference.py (Lines 193-214)

**Purpose**: Route UnlearnCanvas output images to correct directory structure.

**Changes**:
```python
for img_class in retain_prompts[cfg.unlearn_type]:
    # Determine if this is an UnlearnCanvas task (starts with 'uc_')
    is_unlearncanvas = cfg.unlearn_type.startswith('uc_')

    if is_unlearncanvas:
        # UnlearnCanvas: save to unrelated/{class}/
        save_path = os.path.join(cfg.output_path,  "unrelated", cfg.method, img_class)
        save_path2 = os.path.join(cfg.output_path,  "unrelated", "ori", img_class)
    elif cfg.unlearn_type == "nudity" or cfg.unlearn_type == "style_vangogh":
        # Original ConceptACP: save to remove_word/
        save_path = os.path.join(cfg.output_path, "remove_word", cfg.method)
        save_path2 = os.path.join(cfg.output_path,  "remove_word", "ori")
    else:
        # Other object erasure tasks: save to unrelated/{class}/
        save_path = os.path.join(cfg.output_path,  "unrelated", cfg.method, img_class)
        save_path2 = os.path.join(cfg.output_path,  "unrelated", "ori", img_class)
```

**Why**: UnlearnCanvas evaluation requires images organized by class in the `unrelated/` directory for proper accuracy calculation.

### 3. make_block_data.py

**No changes needed**. This file correctly loads from `retain.json` and `forgot.json` for training data generation, which is the correct behavior for both ConceptACP and UnlearnCanvas.

### 4. finetune_block.py

**No changes needed**. This file only loads pre-generated training data files (tar.gz) and doesn't interact with prompt files directly.

## Data File Usage

| File | Used By | Purpose |
|------|---------|---------|
| `retain_featureselect.json` | feature_select.py (UnlearnCanvas only) | Feature selection with "Something" placeholders |
| `retain.json` | make_block_data.py, finetune_block.py | Training data generation and fine-tuning |
| `forgot.json` | feature_select.py, make_block_data.py, finetune_block.py | All stages for both datasets |

## Directory Structure for UnlearnCanvas

```
data/output/sd15_uc_object_bears/
├── forgot/
│   ├── ConcepACP/          # Fine-tuned model output
│   │   ├── 0_0.png
│   │   └── ...
│   └── ori/                # Original model output
│       ├── 0_0.png
│       └── ...
└── unrelated/
    ├── ConcepACP/          # Fine-tuned model output (by class)
    │   ├── Architectures/
    │   ├── Birds/
    │   └── ...
    └── ori/                # Original model output (by class)
        ├── Architectures/
        ├── Birds/
        └── ...
```

## Complete Workflow for UnlearnCanvas

```bash
# 1. Feature selection (uses retain_featureselect.json automatically)
uv run src/conceptacp/utils/feature_select.py model=sd15 unlearn_type=uc_object_bears

# 2. Generate training data (uses retain.json and forgot.json)
uv run src/conceptacp/utils/make_block_data.py model=sd15 unlearn_type=uc_object_bears

# 3. Train model (uses pre-generated data)
uv run src/conceptacp/utils/finetune_block.py model=sd15 unlearn_type=uc_object_bears

# 4. Run inference (saves to unrelated/{class}/ structure)
uv run src/conceptacp/utils/inference.py model=sd15 unlearn_type=uc_object_bears

# 5. Evaluate accuracy
uv run src/conceptacp/utils/compare_unlearn_accuracy.py \
    --input_dir data/output/sd15_uc_object_bears \
    --checkpoint_path checkpoints/style50_cls.pth \
    --target_class Bears
```

## Supported UnlearnCanvas Categories

All 10 categories have been configured:

1. **uc_object_bears** - Bears concept erasure
2. **uc_object_birds** - Birds concept erasure
3. **uc_object_cats** - Cats concept erasure
4. **uc_object_dogs** - Dogs concept erasure
5. **uc_object_fishes** - Fishes concept erasure
6. **uc_object_flowers** - Flowers concept erasure
7. **uc_object_frogs** - Frogs concept erasure
8. **uc_object_horses** - Horses concept erasure
9. **uc_object_human** - Human concept erasure
10. **uc_object_sandwiches** - Sandwiches concept erasure

Each category has:
- Config file in `src/conceptacp/utils/conf/unlearn/`
- Execution script in `scripts/`
- Prompts in `data/train_prompts/forgot.json` and `data/train_prompts/retain_featureselect.json`

## Key Design Decisions

1. **Prefix-based detection**: Using `uc_` prefix allows automatic detection of UnlearnCanvas tasks without manual configuration.

2. **Two-file retain strategy**:
   - `retain_featureselect.json`: Used during feature selection with "Something" placeholders to find discriminative features
   - `retain.json`: Used during training with actual retain concepts to preserve model capabilities

3. **Class-based output structure**: UnlearnCanvas evaluation requires organizing unrelated images by class for per-class accuracy calculation.

4. **Minimal code changes**: Only two files (feature_select.py and inference.py) needed modifications. Training and data generation remain unchanged.

## See Also

- `docs/UNLEARNCANVAS_GUIDE.md` - Complete workflow guide for UnlearnCanvas tasks
- `docs/UNLEARNCANVAS_REMOVEWORD.md` - Details on the "Something" placeholder strategy
- `docs/EVALUATION.md` - Evaluation metrics and usage
