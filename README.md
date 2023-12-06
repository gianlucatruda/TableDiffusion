# TableDiffusion

This is the supporting code for the paper [Generating tabular datasets under differential privacy](https://arxiv.org/abs/2308.14784).

TableDiffusion is a project focused on providing differentially-private generative models for sensitive tabular data. The goal is to enable the synthesis of data that maintains the statistical properties of the original dataset while ensuring the privacy of individuals' information.

> :warning: **Disclaimer**: This codebase is intended for research purposes only and is not ready for production use. The current implementation may not preserve privacy guarantees due to seed and sampler settings that are not suitable for a production environment.

## Structure

The project is structured into several high-level modules:
- `config`:
- `metrics`:
- `models`: Contains the differentially-private generative models for tabular data synthesis.
- `utilities`: Provides utility functions and classes for data processing and model evaluation.
- `vis`:
- `compare.py`:
- `evaluate.py`:


## Models

- `DPattentionGAN_Synthesiser`: A generative adversarial network with differential privacy and attention mechanisms to generate synthetic tabular data.
- `DPattentionVAE_Synthesiser`: A variational autoencoder that incorporates differential privacy and attention mechanisms for data synthesis.
- `DPautoGAN_Synthesiser`: An automatic generative adversarial network that ensures differential privacy in the data generation process.
- `TabDM_Synthesiser`: A synthesizer model that uses tabular diffusion models for data generation while maintaining privacy.
- `WGAN_Synthesiser`: A Wasserstein GAN that provides a stable training process for generating synthetic data with privacy considerations.
- `CTGAN_Synthesiser`: A conditional GAN tailored for generating synthetic tabular data that respects the privacy constraints.
- `PATEGAN_Synthesiser`: A GAN model that incorporates the Private Aggregation of Teacher Ensembles (PATE) framework to ensure differential privacy.

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