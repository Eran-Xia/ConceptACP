# UnlearnCanvas Feature Selection with Remove-Word Strategy

## Overview

For UnlearnCanvas tasks, the feature selection process uses a special "remove-word" strategy to more accurately identify features specific to the target concept (object or style).

## The Problem

When trying to find features that represent a specific concept (e.g., "Bears"), comparing:
- **Forgot set**: "A bear playing violin"
- **Retain set**: "A cat playing violin" (different object)

This can capture features related to both "Bears vs Cats" AND "violin playing". We want features specific to Bears only.

## The Solution: Remove-Word Strategy

Instead, we compare:
- **Forgot set**: "A bear playing violin"
- **Retain set (remove-word)**: "Something playing violin"

This isolates features specific to the concept itself, not contextual differences.

## Implementation

### 1. Prompt Conversion

The `convert_unlearncanvas_prompts.py` script now generates **three** JSON files:

#### `forgot.json`
Original prompts with the target concept:
```json
{
  "uc_object_bears": {
    "Bears": [
      "A bear dressed as a medieval knight",
      "A bear playing a violin",
      ...
    ]
  }
}
```

#### `retain.json` (for training)
Prompts with other objects/styles:
```json
{
  "uc_object_bears": {
    "Cats": ["A cat dressed as...", ...],
    "Dogs": ["A dog playing...", ...],
    "Birds": ["A bird flying...", ...]
  }
}
```

#### `retain_featureselect.json` (for feature selection)
Remove-word prompts:
```json
{
  "uc_object_bears": {
    "remove_word": [
      "Something dressed as a medieval knight",
      "Something playing a violin",
      ...
    ]
  }
}
```

### 2. Automatic Selection in feature_select.py

When `feature_select.py` detects an UnlearnCanvas task (via `target_type` field), it automatically:
1. Checks for `retain_featureselect.json`
2. Uses remove-word prompts as the retain set
3. Falls back to standard `retain.json` if not found

### 3. Training Still Uses Original Retain Set

During `make_block_data.py` and `finetune_block.py`:
- **Forgot set**: Still uses original prompts with target concept
- **Retain set**: Uses other objects/styles from `retain.json`

This ensures the model learns to preserve other concepts while erasing the target.

## Word Removal Rules

### For Objects (e.g., Bears)
- `"A bear ..."` → `"Something ..."`
- `"The bear ..."` → `"Something ..."`
- `"Bears ..."` → `"Things ..."`
- `"bears ..."` → `"things ..."`

Examples:
- "A bear dressed as a knight" → "Something dressed as a knight"
- "Bears are running" → "Things are running"

### For Styles (e.g., Van_Gogh)
- `"... in Van_Gogh style"` → `"..."`

Examples:
- "An image of Bears in Van_Gogh style" → "An image of Bears"
- "A flower in Impressionism style" → "A flower"

## Usage

### Step 1: Convert Prompts (generates all three files)
```bash
uv run scripts/convert_unlearncanvas_prompts.py --type object --name Bears
```

Output:
```
Converting object prompts for: Bears
  Loaded 70 forgot prompts
  Generated retain prompts for 3 categories (for training)
  Generated 70 remove_word prompts (for feature selection)
  Saved to data/train_prompts/
    - forgot.json: uc_object_bears
    - retain.json: uc_object_bears (for training)
    - retain_featureselect.json: uc_object_bears (for feature selection)
```

### Step 2: Feature Selection (automatically uses remove-word)
```bash
CUDA_VISIBLE_DEVICES=0 uv run src/conceptacp/utils/feature_select.py \
    model=sd15 \
    unlearn=uc_object_bears
```

Log output will show:
```
Loading prompts...
UnlearnCanvas task detected - using retain_featureselect.json
```

### Step 3: Training (uses original retain set)
```bash
# make_block_data.py and finetune_block.py use retain.json automatically
CUDA_VISIBLE_DEVICES=0 uv run src/conceptacp/utils/make_block_data.py \
    model=sd15 \
    unlearn=uc_object_bears
```

## Benefits

1. **More Accurate Features**: Features selected are specific to the target concept, not confounded by context
2. **Better Erasure**: Training focuses on the exact features that represent the concept
3. **Automatic**: No manual intervention needed - the system detects and uses the right prompts
4. **Backward Compatible**: Falls back gracefully if remove-word prompts are not available

## File Structure

```
data/train_prompts/
├── forgot.json                    # Target concept prompts
├── retain.json                    # Other concepts (for training)
└── retain_featureselect.json      # Remove-word prompts (for feature selection)
```

## Technical Details

The detection logic in `feature_select.py`:

```python
# Check if this is an UnlearnCanvas task
is_unlearncanvas = cfg.get('target_type') is not None

if is_unlearncanvas:
    # Use remove-word prompts for feature selection
    retain_featureselect_file = Path("data/train_prompts/retain_featureselect.json")
    if retain_featureselect_file.exists():
        with open(retain_featureselect_file) as f:
            retain_prompts = json.load(f)
    else:
        # Fallback to standard retain.json
        with open(cfg.retain_prompts) as f:
            retain_prompts = json.load(f)
```

## Example: Full Workflow

```bash
# 1. Convert prompts (generates remove-word variants)
uv run scripts/convert_unlearncanvas_prompts.py --type object --name Bears

# 2. Feature selection (uses remove-word automatically)
CUDA_VISIBLE_DEVICES=0 uv run src/conceptacp/utils/feature_select.py \
    model=sd15 \
    unlearn=uc_object_bears

# 3. Generate training data (uses original retain set)
CUDA_VISIBLE_DEVICES=0 uv run src/conceptacp/utils/make_block_data.py \
    model=sd15 \
    unlearn=uc_object_bears

# 4. Train (uses original retain set)
CUDA_VISIBLE_DEVICES=0 uv run src/conceptacp/utils/finetune_block.py \
    model=sd15 \
    unlearn=uc_object_bears

# 5. Test
CUDA_VISIBLE_DEVICES=0 uv run src/conceptacp/utils/inference.py \
    model=sd15 \
    unlearn=uc_object_bears \
    unlearncanvas_resources_path=/path/to/UnlearnCanvas_resources
```

The system automatically handles which retain set to use at each stage!
