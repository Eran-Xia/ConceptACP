---
license: apache-2.0
---
# SAeUron: Interpretable Concept Unlearning in Diffusion Models with Sparse Autoencoders

[Paper on arXiv](https://arxiv.org/abs/2501.18052) |
[GitHub repo](https://github.com/cywinski/SAeUron)

The repository contains Sparse Autoencoders trained in our work for blocks `up.1.1` and `up.1.2`.

After cloning our GitHub repo, you can use them as follows:
```python
from SAE.sae import Sae

device = "cuda"
hookpoint = "unet.up_blocks.1.attentions.1"

sae = Sae.load_from_hub("bcywinski/SAeUron", hookpoint=hookpoint, device=device)
```

## 📚 Bibtex
```bibtex
@article{cywinski2025saeuron,
  title={SAeUron: Interpretable Concept Unlearning in Diffusion Models with Sparse Autoencoders},
  author={Cywi{\'n}ski, Bartosz and Deja, Kamil},
  journal={arXiv preprint arXiv:2501.18052},
  year={2025}
}
```