# Concept Acupuncturist: An Efficient and Precise Concept Erasure Method for Diffusion Models Via Sparse Autoencoders

**WARNING: This repository contains model outputs that may be offensive in nature.**

## 1. Environment Setup

This project uses [UV] for reproducible environment management.  

## 2. selet_feature
run notebooks/0x00_feature_select.ipynb

## 3. generate fine-tuning dataset
uv run src/conceptacp/utils/make_data.py

## 4. finetune
uv run src/conceptacp/utils/finetune.py

## 5. generate images for test
uv run src/conceptacp/utils/inference.py
