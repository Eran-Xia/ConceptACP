# Configuration Refactoring Summary

## What Changed

Unified model path configuration from 3 duplicate definitions to a single source of truth.

## Before (Confusing)

```yaml
# Model path defined in 3 places with inconsistent references
conf/data/default.yaml:    model_path: stabilityai/sdxl-turbo
conf/finetune.yaml:        model_path: stabilityai/sdxl-turbo
conf/inference.yaml:       model_path: stabilityai/sdxl-turbo
```

Problems: Duplicates, `model_id` vs `model_path` confusion, inconsistent references.

## After (Clean)

```yaml
# Model path defined ONLY in conf/data/default.yaml
model_path: stabilityai/sdxl-turbo

# Other configs inherit via defaults:
defaults:
  - data: default  # Inherits model_path
```

Benefits: Single source of truth, consistent access, reliable overrides.

## Access Patterns

**Data scripts** (`make_data.py`, `make_block_data.py`):
- Access: `cfg.model_path`
- Override: `model_path=/path/to/sdxl`

**Training/inference scripts** (`finetune*.py`, `inference.py`):
- Access: `cfg.data.model_path`
- Override: `data.model_path=/path/to/sdxl`

## Migration

```bash
# Old (broken):
uv run src/conceptacp/utils/inference.py model_id=/path/to/sdxl

# New (correct):
uv run src/conceptacp/utils/inference.py data.model_path=/path/to/sdxl
```
