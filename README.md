# ConceptACP

Editting Diffusion model based on SAE

1. selet_feature: notebooks/0x00_feature_select.ipynb

2. get dataset:
uv run src/conceptacp/utils/make_data.py

3. finetune:
uv run src/conceptacp/utils/finetune.py

4. generate images for test
uv run src/conceptacp/utils/inference.py

configs are in src/conceptacp/utils/conf
