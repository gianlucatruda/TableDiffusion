# TableDiffusion
[![arxivbadge](https://img.shields.io/badge/arXiv-2308.14784-green)](https://arxiv.org/abs/2308.14784)
[![githubbadge](https://img.shields.io/badge/Github-TableDiffusion-black)](https://github.com/gianlucatruda/TableDiffusion)
[![blogbadge](https://img.shields.io/badge/gianluca.ai-projects-blue)](https://gianluca.ai/table-diffusion)

This is the supporting code for the paper [Generating tabular datasets under differential privacy](https://arxiv.org/abs/2308.14784).

Please check out a quick overview on [my blog](http://gianluca.ai/table-diffusion).

TableDiffusion is a project focused on providing differentially-private generative models for sensitive tabular data. The goal is to enable the synthesis of data that maintains the statistical properties of the original dataset while ensuring the privacy of individuals' information.

The most notable model from this work is `TableDiffusion`, the first differentially-private diffusion model for tabular data. See [tablediffusion/models/table_diffusion.py](tablediffusion/models/table_diffusion.py)

> :warning: **Disclaimer**: This codebase is intended for research purposes only and is not ready for production use. The current implementation may not preserve privacy guarantees due to seed and sampler settings that are not suitable for a production environment.


## Paper explanation on YouTube

[https://youtu.be/2QRrGWoXOb4](https://youtu.be/2QRrGWoXOb4)

[![Paper presentation on YouTube](https://img.youtube.com/vi/2QRrGWoXOb4/0.jpg)](https://www.youtube.com/watch?v=2QRrGWoXOb4)


## Citing this work

Truda, Gianluca. "Generating tabular datasets under differential privacy." arXiv preprint arXiv:2308.14784 (2023).

```
@article{truda2023generating,
  title={Generating tabular datasets under differential privacy},
  author={Truda, Gianluca},
  journal={arXiv preprint arXiv:2308.14784},
  year={2023}
}
```
