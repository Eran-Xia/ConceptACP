# Configuration System Redesign

## Overview

The Hydra configuration system has been redesigned for better organization, clarity, and maintainability. The new structure separates concerns into four distinct categories while maintaining backward compatibility with existing scripts.

## New Structure

```
conf/
├── config.yaml              # Main entry point (for reference, not used by scripts)
├── paths.yaml               # Centralized path definitions
│
├── model/                   # Model-specific settings
│   ├── sdxl.yaml           # SDXL configuration
│   └── sd15.yaml           # SD 1.5 configuration
│
├── task/                    # Task-specific hyperparameters
│   ├── make_data.yaml
│   ├── make_block_data.yaml
│   ├── finetune.yaml
│   ├── finetune_block.yaml
│   ├── inference.yaml
│   └── feature_select.yaml
│
├── unlearn/                 # Concept-specific configs
│   ├── nudity.yaml
│   ├── object_church.yaml
│   ├── object_parachute.yaml
│   └── style_vangogh.yaml
│
└── [script_name].yaml       # Per-script composition configs
    ├── make_data.yaml
    ├── make_block_data.yaml
    ├── finetune.yaml
    ├── finetune_block.yaml
    ├── inference.yaml
    └── feature_select.yaml
```

## Key Design Decisions

### 1. Separation of Concerns

Configs are organized by their purpose:
- **Model configs**: Architecture, dimensions, SAE checkpoints
- **Task configs**: Hyperparameters for specific operations
- **Unlearn configs**: Concept-specific features and settings
- **Paths config**: Centralized location for all paths

### 2. Flattened Composition

Per-script configs use the `@_here_` pattern to compose from organized sources while flattening for backward compatibility:

```yaml
defaults:
  - paths
  - model: sdxl
  - task/make_data@_here_
  - unlearn: nudity
  - _self_

# Flatten for backward compatibility
model_path: ${model.model_path}
unlearn_type: ${unlearn.unlearn_type}
# ... etc
```

This allows:
- Clean organization in source files
- No code changes needed (cfg.model_path still works)
- Clear config hierarchy in defaults list

### 3. Override Syntax

**Model selection:**
```bash
model=sd15           # Use SD 1.5 instead of SDXL
model=sdxl          # Default: use SDXL
```

**Concept selection:**
```bash
unlearn=nudity              # Default
unlearn=object_church       # Switch to church erasure
unlearn=style_vangogh       # Switch to style erasure
```

**Model path override:**
```bash
model.model_path=/path/to/model   # Override model location
```

**Task parameter overrides:**
```bash
batch_size=64                     # Change batch size
learning_rate=0.0001              # Change learning rate
num_inference_steps=4             # Change inference steps
```

## Migration from Old System

### What Changed

**Old structure:**
```
conf/
├── data/
│   ├── default.yaml    # All data + model settings mixed
│   └── sd15.yaml       # SD15 overrides
├── finetune.yaml       # Imported data config with @_here_
├── inference.yaml      # Imported data config with @_here_
└── feature_select.yaml
```

**New structure:**
- Separated into model/, task/, unlearn/, and paths.yaml
- Each script has its own composition config
- Clear hierarchy and no duplication

### What Stayed the Same

**No code changes needed:**
- All config access patterns remain: cfg.model_path, cfg.unlearn_type, etc.
- Scripts still use @hydra.main decorators
- Override syntax simplified but old syntax still works

### Command Changes

**Old:**
```bash
uv run ... data=sd15 unlearn_type=nudity
```

**New:**
```bash
uv run ... model=sd15 unlearn_type=nudity
```

The change from `data=sd15` to `model=sd15` is more semantic and clear.

## Benefits

1. **Clear organization**: Easy to find where settings are defined
2. **No duplication**: Each setting defined in one place
3. **Easy model switching**: `model=sd15` vs `model=sdxl`
4. **Easy concept switching**: `unlearn=object_church`
5. **Maintainable**: Adding new models or concepts is straightforward
6. **Backward compatible**: Existing scripts work without modification

## Testing

All configurations have been tested:
- ✓ Config files load successfully
- ✓ Composition works correctly
- ✓ Overrides work as expected
- ✓ Scripts can import and initialize
- ✓ Both SDXL and SD15 models work
- ✓ All unlearn types work

## Examples

**Feature selection for nudity on SDXL:**
```bash
uv run src/conceptacp/utils/feature_select.py unlearn_type=nudity
```

**Block data generation for church on SD15:**
```bash
uv run src/conceptacp/utils/make_block_data.py model=sd15 unlearn=object_church
```

**Training with custom batch size:**
```bash
uv run src/conceptacp/utils/finetune_block.py unlearn=style_vangogh batch_size=64
```

**Inference with local model:**
```bash
uv run src/conceptacp/utils/inference.py model.model_path=/path/to/model
```

## Future Enhancements

Possible improvements for the future:
1. Add more unlearn concepts as YAML configs
2. Create model configs for other diffusion models
3. Add validation schemas for configs
4. Create config generation scripts
5. Add config documentation generation
