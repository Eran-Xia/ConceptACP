# UnlearnCanvas Integration Guide

This document explains how to use ConceptACP with the UnlearnCanvas dataset for SD 1.5.

## Overview

UnlearnCanvas provides:
- **50 artistic styles** (Van_Gogh, Monet, Impressionism, etc.)
- **20 object categories** (Bears, Cats, Dogs, Flowers, etc.)

When erasing a concept:
- **Style erasure**: Test with 20 prompts (all objects in that style)
- **Object erasure**: Test with 50 prompts (that object in all styles)

## Prerequisites

1. Set the path to UnlearnCanvas resources in your environment:
```bash
export UNLEARNCANVAS_PATH=/path/to/UnlearnCanvas_resources
```

Or pass it as a parameter:
```bash
unlearncanvas_resources_path=/path/to/UnlearnCanvas_resources
```

2. Ensure const.py exists at `${UNLEARNCANVAS_PATH}/const.py`

## Complete Workflow

### Method 1: Convert Prompts First (Recommended for Training)

This method converts UnlearnCanvas finetune prompts to ConceptACP format, then uses the standard workflow.

**NEW: Automatically generates "remove-word" prompts for more accurate feature selection!**

#### Step 1: Convert Training Prompts

```bash
# For a specific style
uv run scripts/convert_unlearncanvas_prompts.py \
    --type style \
    --name Van_Gogh

# For a specific object
uv run scripts/convert_unlearncanvas_prompts.py \
    --type object \
    --name Bears
```

This creates/updates **three** files in `data/train_prompts/`:
- `forgot.json` - Original prompts with target concept
- `retain.json` - Other objects/styles (used for training)
- `retain_featureselect.json` - Remove-word prompts (used for feature selection)

**What are remove-word prompts?**
For better feature selection, the target concept is replaced with a generic word:
- Object: "A bear playing violin" → "Something playing violin"
- Style: "Bears in Van_Gogh style" → "Bears"

This isolates features specific to the concept. See `docs/UNLEARNCANVAS_REMOVEWORD.md` for details.

#### Step 2: Feature Selection

**Automatically uses remove-word prompts for better accuracy!**

```bash
# For style
uv run src/conceptacp/utils/feature_select.py \
    model=sd15 \
    unlearn=uc_style_vangogh \
    unlearn.target_name=Van_Gogh

# For object
uv run src/conceptacp/utils/feature_select.py \
    model=sd15 \
    unlearn=uc_object_bears \
    unlearn.target_name=Bears
```

`feature_select.py` automatically:
- Detects UnlearnCanvas task (via `target_type` field)
- Uses `retain_featureselect.json` (remove-word prompts) as retain set
- Compares "A bear playing violin" vs "Something playing violin"
- Finds features specific to Bears, not the context

#### Step 3: Generate Training Data

```bash
# For style
uv run src/conceptacp/utils/make_block_data.py \
    model=sd15 \
    unlearn=uc_style_vangogh

# For object
uv run src/conceptacp/utils/make_block_data.py \
    model=sd15 \
    unlearn=uc_object_bears
```

#### Step 4: Fine-tune

```bash
# For style
uv run src/conceptacp/utils/finetune_block.py \
    model=sd15 \
    unlearn=uc_style_vangogh

# For object
uv run src/conceptacp/utils/finetune_block.py \
    model=sd15 \
    unlearn=uc_object_bears
```

#### Step 5: Inference (Dynamic Test Prompts)

```bash
# For style - generates 20 test prompts (all objects in Van_Gogh style)
uv run src/conceptacp/utils/inference.py \
    model=sd15 \
    unlearn=uc_style_vangogh \
    unlearncanvas_resources_path=$UNLEARNCANVAS_PATH

# For object - generates 50 test prompts (Bears in all styles)
uv run src/conceptacp/utils/inference.py \
    model=sd15 \
    unlearn=uc_object_bears \
    unlearncanvas_resources_path=$UNLEARNCANVAS_PATH
```

### Method 2: Use Any Style/Object On-the-Fly

You can test any style or object without converting prompts first:

```bash
# Test erasing Monet style
uv run src/conceptacp/utils/inference.py \
    model=sd15 \
    unlearn=uc_style_vangogh \
    unlearn.target_name=Monet \
    unlearncanvas_resources_path=$UNLEARNCANVAS_PATH

# Test erasing Cats
uv run src/conceptacp/utils/inference.py \
    model=sd15 \
    unlearn=uc_object_bears \
    unlearn.target_name=Cats \
    unlearncanvas_resources_path=$UNLEARNCANVAS_PATH
```

**Note:** This only works for inference. For training, you must convert prompts first.

## Configuration Files

Two template configs are provided:

### `conf/unlearn/uc_style_vangogh.yaml`
```yaml
unlearn_type: uc_style_vangogh
target_type: style
target_name: Van_Gogh  # Override with unlearn.target_name=<style>
code: down.2.1  # Will be auto-selected by feature_select.py
lambda_neg_c: 0.2
feature_scales: {}  # Populated by feature_select.py
```

### `conf/unlearn/uc_object_bears.yaml`
```yaml
unlearn_type: uc_object_bears
target_type: object
target_name: Bears  # Override with unlearn.target_name=<object>
code: down.2.1  # Will be auto-selected by feature_select.py
lambda_neg_c: 0.2
feature_scales: {}  # Populated by feature_select.py
```

The `code` field is a placeholder - `feature_select.py` will automatically evaluate all available blocks and select the best one for the specific concept.

## Available Styles (50)

Abstractionism, Artist_Sketch, Blossom_Season, Bricks, Byzantine, Cartoon, Cold_Warm, Color_Fantasy, Comic_Etch, Crayon, Cubism, Dadaism, Dapple, Defoliation, Early_Autumn, Expressionism, Fauvism, French, Glowing_Sunset, Gorgeous_Love, Greenfield, Impressionism, Ink_Art, Joy, Liquid_Dreams, Magic_Cube, Meta_Physics, Meteor_Shower, Monet, Mosaic, Neon_Lines, On_Fire, Pastel, Pencil_Drawing, Picasso, Pop_Art, Red_Blue_Ink, Rust, Seed_Images, Sketch, Sponge_Dabbed, Structuralism, Superstring, Surrealism, Ukiyoe, Van_Gogh, Vibrant_Flow, Warm_Love, Warm_Smear, Watercolor, Winter

## Available Objects (20)

Architectures, Bears, Birds, Butterfly, Cats, Dogs, Fishes, Flame, Flowers, Frogs, Horses, Human, Jellyfish, Rabbits, Sandwiches, Sea, Statues, Towers, Trees, Waterfalls

## Test Prompt Format

All test prompts follow the template:
```
An image of {object} in {style} style
```

**Examples:**
- "An image of Bears in Van_Gogh style"
- "An image of Flowers in Impressionism style"
- "An image of Cats in Watercolor style"

## Output Structure

After inference, images are saved to:
```
data/output/
├── forgot/           # Images of the erased concept
│   ├── ori/          # Original model
│   └── {method}/     # Fine-tuned model
└── unrelated/        # Images of retain concepts
    ├── ori/
    └── {method}/
```

For style erasure:
- `forgot/`: Target style with all 20 objects
- `unrelated/`: 3 other styles with all objects

For object erasure:
- `forgot/`: Target object with all 50 styles
- `unrelated/`: 3 other objects with all styles

## Tips

1. **Start with one concept**: Test the workflow with one style or object before batch processing

2. **Adjust lambda_neg_c**: Different concepts may need different erasure strengths
   - Start with 0.2 (default)
   - Increase to 1.0-2.0 for stronger erasure

3. **Monitor feature selection**: Check `data/finetune_config/sd15_uc_<type>_<name>.json` after feature selection

4. **Batch processing**: To erase multiple concepts, use a loop:
   ```bash
   for style in Van_Gogh Monet Impressionism; do
       # Convert prompts
       uv run scripts/convert_unlearncanvas_prompts.py --type style --name $style

       # Run full workflow
       uv run src/conceptacp/utils/feature_select.py model=sd15 unlearn=uc_style_vangogh unlearn.target_name=$style
       uv run src/conceptacp/utils/make_block_data.py model=sd15 unlearn=uc_style_vangogh unlearn.target_name=$style
       uv run src/conceptacp/utils/finetune_block.py model=sd15 unlearn=uc_style_vangogh unlearn.target_name=$style
   done
   ```

## Troubleshooting

**Q: "UnlearnCanvas resources not available"**
A: Set `unlearncanvas_resources_path` parameter or check the path exists

**Q: "target_type and target_name required"**
A: Make sure you're using `unlearn=uc_style_vangogh` or `unlearn=uc_object_bears` config

**Q: "Prompt file not found"**
A: Run the conversion script first: `scripts/convert_unlearncanvas_prompts.py`

**Q: "Feature config not found"**
A: Run feature selection first: `feature_select.py` with the correct unlearn config

## Examples

### Complete Example: Erase Van Gogh Style

```bash
# 1. Convert prompts
uv run scripts/convert_unlearncanvas_prompts.py --type style --name Van_Gogh

# 2. Feature selection
uv run src/conceptacp/utils/feature_select.py \
    model=sd15 \
    unlearn=uc_style_vangogh

# 3. Generate data
uv run src/conceptacp/utils/make_block_data.py \
    model=sd15 \
    unlearn=uc_style_vangogh

# 4. Fine-tune
uv run src/conceptacp/utils/finetune_block.py \
    model=sd15 \
    unlearn=uc_style_vangogh

# 5. Test
uv run src/conceptacp/utils/inference.py \
    model=sd15 \
    unlearn=uc_style_vangogh \
    unlearncanvas_resources_path=$UNLEARNCANVAS_PATH
```

### Complete Example: Erase Bears

```bash
# 1. Convert prompts
uv run scripts/convert_unlearncanvas_prompts.py --type object --name Bears

# 2. Feature selection
uv run src/conceptacp/utils/feature_select.py \
    model=sd15 \
    unlearn=uc_object_bears

# 3. Generate data
uv run src/conceptacp/utils/make_block_data.py \
    model=sd15 \
    unlearn=uc_object_bears

# 4. Fine-tune
uv run src/conceptacp/utils/finetune_block.py \
    model=sd15 \
    unlearn=uc_object_bears

# 5. Test
uv run src/conceptacp/utils/inference.py \
    model=sd15 \
    unlearn=uc_object_bears \
    unlearncanvas_resources_path=$UNLEARNCANVAS_PATH
```
