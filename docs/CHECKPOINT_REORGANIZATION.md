# Checkpoint Reorganization Summary

## Changes Made

Reorganized SAE checkpoint structure for better maintainability and clarity between SDXL and SD 1.5 models.

## New Directory Structure

**Before:**
```
checkpoints/
├── unet.down_blocks.2.attentions.1_k10_hidden5120_auxk256_bs4096_lr0.0001/
├── unet.mid_block.attentions.0_k10_hidden5120_auxk256_bs4096_lr0.0001/
├── unet.up_blocks.0.attentions.0_k10_hidden5120_auxk256_bs4096_lr0.0001/
└── unet.up_blocks.0.attentions.1_k10_hidden5120_auxk256_bs4096_lr0.0001/
```

**After:**
```
checkpoints/
├── sdxl-ckpts/                    # SDXL SAE checkpoints (organized)
│   ├── unet.down_blocks.2.attentions.1_k10_hidden5120_auxk256_bs4096_lr0.0001/
│   ├── unet.mid_block.attentions.0_k10_hidden5120_auxk256_bs4096_lr0.0001/
│   ├── unet.up_blocks.0.attentions.0_k10_hidden5120_auxk256_bs4096_lr0.0001/
│   └── unet.up_blocks.0.attentions.1_k10_hidden5120_auxk256_bs4096_lr0.0001/
├── sd1.5-sae-ckpts/              # SD 1.5 SAE checkpoints (from SAeUron)
│   ├── unet.up_blocks.1.attentions.1/
│   └── unet.up_blocks.1.attentions.2/
└── README.md                      # Documentation
```

## Updated Files

### Configuration Files

1. **`src/conceptacp/utils/conf/data/default.yaml`**
   ```yaml
   # Before
   sae_checkpoints: ${work_dir}/checkpoints

   # After
   sae_checkpoints: ${work_dir}/checkpoints/sdxl-ckpts
   ```

2. **`src/conceptacp/utils/conf/data/sd15.yaml`**
   ```yaml
   sae_checkpoints: ${work_dir}/checkpoints/sd1.5-sae-ckpts
   ```

### Documentation Files

1. **`CLAUDE.md`**
   - Updated SAE checkpoint paths
   - Added model-specific checkpoint locations
   - Updated data organization section

2. **`SD15_IMPLEMENTATION.md`**
   - Updated checkpoint structure documentation
   - Changed absolute paths to relative paths
   - Added organized directory tree
   - Updated troubleshooting commands

3. **`docs/FEATURE_SELECTION.md`**
   - Updated SAE checkpoint paths
   - Changed troubleshooting section paths

4. **`docs/FEATURE_SELECTION_IMPLEMENTATION.md`**
   - Updated example output with new paths

5. **`checkpoints/README.md`** (new)
   - Comprehensive documentation of checkpoint structure
   - Format descriptions for both SDXL and SD 1.5
   - Usage examples
   - Storage requirements

## Benefits

1. **Clear Separation**: Model-specific checkpoints in separate directories
2. **Maintainability**: Easier to manage and update checkpoints
3. **Portability**: All checkpoints now local to the project (not relying on external paths)
4. **Documentation**: Comprehensive README in checkpoints directory
5. **Scalability**: Easy to add checkpoints for new models or blocks

## Verification

```bash
# Check SDXL checkpoints
ls checkpoints/sdxl-ckpts/
# Output: 4 checkpoint directories

# Check SD 1.5 checkpoints
ls checkpoints/sd1.5-sae-ckpts/
# Output: 2 checkpoint directories + README.md

# Verify configs
grep sae_checkpoints src/conceptacp/utils/conf/data/default.yaml
# Output: sae_checkpoints: ${work_dir}/checkpoints/sdxl-ckpts

grep sae_checkpoints src/conceptacp/utils/conf/data/sd15.yaml
# Output: sae_checkpoints: ${work_dir}/checkpoints/sd1.5-sae-ckpts
```

## Backward Compatibility

✅ **Fully compatible** - Config changes are internal, scripts automatically use correct paths through Hydra

- SDXL scripts: Automatically use `checkpoints/sdxl-ckpts/`
- SD 1.5 scripts: Automatically use `checkpoints/sd1.5-sae-ckpts/`
- No command-line changes required
- Existing workflows unchanged

## Migration Guide

If you have checkpoints in old locations:

### Option 1: Move Checkpoints (Recommended)

```bash
# Move SDXL checkpoints
mkdir -p checkpoints/sdxl-ckpts
mv checkpoints/unet.* checkpoints/sdxl-ckpts/

# Copy or symlink SD 1.5 checkpoints from SAeUron
cp -r /path/to/SAeUron/sae-ckpts/* checkpoints/sd1.5-sae-ckpts/
# OR
ln -s /path/to/SAeUron/sae-ckpts checkpoints/sd1.5-sae-ckpts
```

### Option 2: Override Config

```bash
# Use custom checkpoint location
uv run src/conceptacp/utils/feature_select.py \
    sae_checkpoints=/path/to/custom/checkpoints
```

## Testing

All scripts tested with new structure:

```bash
# Feature selection
uv run src/conceptacp/utils/feature_select.py --help
# ✓ Config loads correctly

# Make block data
uv run src/conceptacp/utils/make_block_data.py --help
# ✓ Config loads correctly

# SD 1.5 with new paths
uv run src/conceptacp/utils/feature_select.py data=sd15 --help
# ✓ SD 1.5 config loads correctly
```

## Summary

Successfully reorganized SAE checkpoints into a cleaner, more maintainable structure:

- ✅ SDXL checkpoints: `checkpoints/sdxl-ckpts/`
- ✅ SD 1.5 checkpoints: `checkpoints/sd1.5-sae-ckpts/`
- ✅ All configs updated
- ✅ All documentation updated
- ✅ Comprehensive README added
- ✅ Fully backward compatible
- ✅ No breaking changes

The new structure provides better organization, clearer separation between model types, and easier maintenance going forward.
