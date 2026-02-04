# Unlearning Accuracy Evaluation

This directory contains scripts for evaluating ConceptACP model unlearning accuracy.

## Files

- `evaluate_unlearn_accuracy.py`: Evaluate single model's unlearning accuracy
- `compare_unlearn_accuracy.py`: Compare effects between original model and ConceptACP model

## Dependencies

Required packages (already included in project dependencies):
- timm (for ViT classifier)
- torch
- torchvision
- PIL

## Usage

### 1. Evaluate Single Model

Evaluate ConceptACP model:
```bash
uv run src/conceptacp/utils/evaluate_unlearn_accuracy.py \
    --input_dir data/output/sd15_uc_object_bears \
    --checkpoint_path checkpoints/style50_cls.pth \
    --target_class Bears \
    --model_name ConcepACP
```

Evaluate original model:
```bash
uv run src/conceptacp/utils/evaluate_unlearn_accuracy.py \
    --input_dir data/output/sd15_uc_object_bears \
    --checkpoint_path checkpoints/style50_cls.pth \
    --target_class Bears \
    --model_name ori
```

### 2. Compare Two Models (Recommended)

```bash
uv run src/conceptacp/utils/compare_unlearn_accuracy.py \
    --input_dir data/output/sd15_uc_object_bears \
    --checkpoint_path checkpoints/style50_cls.pth \
    --target_class Bears
```

## Parameters

- `--input_dir`: Base directory containing `forgot/` and `unrelated/` subdirectories
- `--checkpoint_path`: Classifier checkpoint path (style50_cls.pth)
- `--target_class`: Target class to evaluate (default: Bears)
- `--model_name`: Model subdirectory name (ConcepACP or ori)
- `--batch_size`: Batch size (default: 32)
- `--num_workers`: Number of data loading threads (default: 4)

## Required Data Structure

Input directory should have the following structure:
```
data/output/sd15_uc_object_bears/
├── forgot/
│   ├── ConcepACP/          # ConceptACP model generated images
│   │   ├── 0_0.png
│   │   ├── 1_0.png
│   │   └── ...
│   └── ori/                # Original model generated images
│       ├── 0_0.png
│       └── ...
└── unrelated/
    ├── ConcepACP/          # ConceptACP model generated other class images
    │   ├── Architectures/
    │   │   ├── 0_0.png
    │   │   └── ...
    │   ├── Birds/
    │   └── ...
    └── ori/                # Original model generated other class images
        ├── Architectures/
        └── ...
```

## Evaluation Metrics

### 1. Unlearning Accuracy (UA)
- **Definition**: 1 - (proportion classified as target class)
- **Range**: 0-100%
- **Higher is better**: Indicates model successfully "forgot" the target class
- **Example**: UA=95% means only 5% of forgot images are classified as target class

### 2. In-class Retain Accuracy (IRA)
- **Definition**: Classification accuracy for other classes
- **Range**: 0-100%
- **Higher is better**: Indicates model still correctly recognizes other classes
- **Example**: IRA=85% means 85% of non-target class images are correctly classified

## Output Example

```
============================================================
SUMMARY
============================================================
Model: ConcepACP
Target class: Bears
Unlearning Accuracy (UA): 92.16%
In-class Retain Accuracy (IRA): 84.31%

######################################################################
# COMPARISON SUMMARY
######################################################################

Target Class: Bears

Metric                                      Original    ConceptACP         Delta
--------------------------------------------------------------------------------
Unlearning Accuracy (UA)                      23.53%        92.16%       +68.63%
In-class Retain Accuracy (IRA)                87.58%        84.31%        -3.27%
--------------------------------------------------------------------------------

Interpretation:
  - UA measures how well the model has forgotten Bears
    (higher is better: images are NOT classified as Bears)
  - IRA measures how well the model retains other classes (higher is better)

✓ ConceptACP improved unlearning by 68.63% (forgot Bears better)
✓ ConceptACP maintained retention performance (-3.27%)
```

## Classifier Information

The classifier (`style50_cls.pth`) is a ViT-Large based model trained on UnlearnCanvas's 20 classes:
- Architectures, Bears, Birds, Butterfly, Cats, Dogs, Fishes, Flame, Flowers, Frogs
- Horses, Human, Jellyfish, Rabbits, Sandwiches, Sea, Statues, Towers, Trees, Waterfalls

## Troubleshooting

1. **CUDA out of memory**: Reduce `--batch_size`
2. **Images not found**: Check `--input_dir` path and directory structure
3. **Classifier loading failed**: Verify `--checkpoint_path` points to correct .pth file
