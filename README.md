# Concept Acupuncturist: An Efficient and Precise Concept Erasure Method for Diffusion Models Via Sparse Autoencoders

**Warning:** This repository includes model outputs that may be offensive.

## Requirements
- UV installed: https://docs.astral.sh/uv/

## Environment
uv sync

## 1) Feature selection
Run the notebook interactively:
```bash
notebooks/0x00_feature_select.ipynb
```

## 2) Generate the fine-tuning dataset
```bash
uv run src/conceptacp/utils/make_data.py
```

## 3) Fine-tune
```bash
uv run src/conceptacp/utils/finetune.py
```

## 4) Inference (generate test images)
```bash
uv run src/conceptacp/utils/inference.py
```
