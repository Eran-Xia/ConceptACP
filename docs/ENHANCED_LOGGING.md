# Enhanced Logging

Both fine-tuning scripts include comprehensive training metrics.

## Metrics Tracked

**Training Time:**
- Total training time
- Average time per epoch

**Memory Usage:**
- CPU memory
- GPU memory (allocated, peak)

**Parameter Changes:**
- Total and changed parameter counts
- Average and maximum change magnitude
- Layer with maximum change

## Output Format

**During training:**
```
Epoch 1/100 | Loss: 0.0123 (forgot: 0.0100, retain: 0.0023) | Time: 45.32s | GPU: 1024MB
```

**After completion:**
```
================================================================================
TRAINING COMPLETE!
================================================================================
Total training time: 20.52 minutes (1231.0s)
Average time per epoch: 12.31s

Memory Usage:
  Peak GPU memory: 1587.2MB
  Final GPU memory: 1536.8MB
  CPU memory: 1024.5MB

Parameter Changes:
  Total parameters: 350,840,320
  Changed parameters: 348,234,567 (99.26%)
  Average parameter change: 1.45e-05
  Maximum parameter change: 4.23e-02
  Layer with max change: transformer_blocks.2.attn2.to_v.weight
================================================================================
```

## Usage

No changes to command syntax - logging is automatic.

## Implementation

Uses `psutil` for CPU memory, `torch.cuda` for GPU stats. Parameter changes computed with 1e-6 threshold for significance.
