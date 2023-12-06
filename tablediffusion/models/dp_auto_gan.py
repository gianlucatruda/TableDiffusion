"""
Based on DP-auto-GAN implementation from
https://github.com/DPautoGAN/DPautoGAN
"""

import mlflow
import numpy as np
import torch
from models.architectures import Discriminator, Generator
from opacus import PrivacyEngine
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from utilities import *


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def return_data(self):
        return self.data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        X = self.data[idx, :]
        return torch.as_tensor(X, dtype=torch.float64)


class Autoencoder(nn.Module):
    def __init__(self, example_dim, compression_dim, binary=True, device="cpu"):
        super().__init__()

        self.compression_dim = compression_dim

        self.encoder = nn.Sequential(
            nn.Linear(example_dim, (example_dim + compression_dim) // 2),
            nn.Tanh() if binary else nn.LeakyReLU(0.2),
            nn.Linear((example_dim + compression_dim) // 2, compression_dim),
            nn.Tanh() if binary else nn.LeakyReLU(0.2),
        ).to(device)

        self.decoder = nn.Sequential(
            nn.Linear(compression_dim, (example_dim + compression_dim) // 2),
            nn.Tanh() if binary else nn.LeakyReLU(0.2),
            nn.Linear((example_dim + compression_dim) // 2, example_dim),
            nn.Sigmoid(),
        ).to(device)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_compression_dim(self):
        return self.compression_dim


class DPautoGAN_Synthesiser:
    def __init__(
        self,
        batch_size=64,
        gen_lr=2e-4,
        dis_lr=2e-4,
        ae_lr=0.005,
        latent_dim=100,
        gen_dims=(256, 256),
        dis_dims=(256, 256),
        ae_compress_dim=15,
        epsilon_target=5,
        max_grad_norm=1.0,
        ae_eps_frac=0.3,
        epoch_target=200,
        delta=1e-5,
        mlflow_logging=True,
        cuda=True,
    ):

        # Setting up GPU (if available and specified)
        if cuda:
            assert torch.cuda.is_available()
        self.cuda = cuda
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and cuda else "cpu")

        # Hyperparameters
        self.batch_size = batch_size
        self.gen_lr = gen_lr
        self.dis_lr = dis_lr
        self.ae_lr = ae_lr
        self.latent_dim = latent_dim
        self.gen_dims = gen_dims
        self.dis_dims = dis_dims
        self.ae_compress_dim = ae_compress_dim
        self.max_grad_norm = max_grad_norm
        self.ae_eps_frac = ae_eps_frac
        self.epoch_target = epoch_target

        # Setting privacy budget
        self.epsilon_target = epsilon_target
        self._delta = delta

        # Logging to MLflow
        self.mlflow_logging = mlflow_logging
        if self.mlflow_logging:
            _param_dict = gather_object_params(self, prefix="init.")
            mlflow.log_params(_param_dict)

        # Initialise training variables
        self._elapsed_batches = 0
        self._elapsed_epochs = 0
        self._epsilon = epsilon_target
        self._eps = 0

    def fit(self, train_data, n_epochs=10, epsilon=100):

        self._epsilon = epsilon

        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        self.data_dim = train_data.shape[-1]
        self.data_n = train_data.shape[0]

        if not isinstance(train_data, DataLoader):
            train_data = DataLoader(MyDataset(train_data), batch_size=self.batch_size)

        # Check if training for first time or continuing (don't reinitialise)
        if self._elapsed_batches == 0:

            self.AE = Autoencoder(
                example_dim=self.data_dim,
                compression_dim=self.ae_compress_dim,
                binary=False,
                device=self.device,
            )
            self.decoder = self.AE.decoder

            self.G = Generator(
                self.latent_dim, self.AE.get_compression_dim(), gen_dims=self.gen_dims
            ).to(self.device)
            self.D = Discriminator(self.data_dim, dis_dims=self.dis_dims).to(self.device)

            self._nparams_ae = count_parameters(self.AE)
            self._nparams_g = count_parameters(self.G)
            self._nparams_d = count_parameters(self.D)

            if self.mlflow_logging:
                mlflow.log_params(
                    {
                        "nparams.AE": self._nparams_ae,
                        "nparams.G": self._nparams_g,
                        "nparams.D": self._nparams_d,
                        "nparams.total": self._nparams_ae + self._nparams_g + self._nparams_d,
                    }
                )
                _model_desc = str(self.AE) + "\n" + str(self.G) + "\n" + str(self.D)
                mlflow.log_text(_model_desc, "model_desription.txt")

            self.optim_enc = torch.optim.Adam(
                params=self.AE.get_encoder().parameters(),
                lr=self.ae_lr,
                betas=(0.9, 0.999),
                weight_decay=0,
            )

            self.optim_dec = torch.optim.Adam(
                params=self.AE.get_decoder().parameters(),
                lr=self.ae_lr,
                betas=(0.9, 0.999),
                weight_decay=0,
            )

            self.privacy_engine_AE = PrivacyEngine(accountant="rdp", secure_mode=False)
            (
                self.AE.decoder,
                self.optim_dec,
                train_data,
            ) = self.privacy_engine_AE.make_private_with_epsilon(
                module=self.AE.get_decoder(),
                optimizer=self.optim_dec,
                data_loader=train_data,
                target_epsilon=self.ae_eps_frac * self.epsilon_target,
                target_delta=self._delta,
                epochs=1,
                max_grad_norm=self.max_grad_norm,
                poisson_sampling=False,
            )

            # Log privacy engine and optimiser parameters
            if self.mlflow_logging:
                _param_dict = gather_object_params(
                    self.privacy_engine_AE, prefix="privacy_engine_AE."
                )
                mlflow.log_params(_param_dict)
                _param_dict = gather_object_params(self.optim_dec, prefix="optim_dec.")
                mlflow.log_params(_param_dict)

            self.optim_G = torch.optim.RMSprop(
                params=self.G.parameters(), lr=self.gen_lr, alpha=0.99, weight_decay=0
            )

            self.optim_D = torch.optim.RMSprop(
                params=self.D.parameters(), lr=self.dis_lr, alpha=0.99, weight_decay=0
            )

            self.privacy_engine_D = PrivacyEngine(accountant="rdp", secure_mode=False)
            self.D, self.optim_D, train_data = self.privacy_engine_D.make_private_with_epsilon(
                module=self.D,
                optimizer=self.optim_D,
                data_loader=train_data,
                target_epsilon=self.epsilon_target * (1 / self.ae_eps_frac),
                target_delta=self._delta,
                epochs=n_epochs,
                max_grad_norm=self.max_grad_norm,
                poisson_sampling=False,
            )

            # Log privacy engine and optimiser parameters
            if self.mlflow_logging:
                _param_dict = gather_object_params(
                    self.privacy_engine_D, prefix="privacy_engine_D."
                )
                mlflow.log_params(_param_dict)
                _param_dict = gather_object_params(self.optim_D, prefix="optim_D.")
                mlflow.log_params(_param_dict)

            self.ae_criterion = nn.BCELoss()

            # AE pretraining
            self._elapsed_batches += 1
            for i, X in enumerate(train_data):
                self._eps_ae = self.privacy_engine_AE.get_epsilon(self._delta)
                if self._eps_ae >= self._epsilon:
                    print(f"Privacy budget reached in pre-training ({self._eps_ae})")
                    return self

                self._elapsed_batches += 1
                self.optim_enc.zero_grad()
                self.optim_dec.zero_grad()
                real_X = Variable(X.type(Tensor))
                output = self.AE(real_X)
                loss = self.ae_criterion(output, real_X)
                loss.backward()
                self.optim_enc.step()
                self.optim_dec.step()

                if i % 300 == 0:
                    print(f"AE loss: {loss.item():.4f}")

                if i % 20 == 0 and self.mlflow_logging:
                    mlflow.log_metrics(
                        {
                            "elapsed_batches": self._elapsed_batches,
                            "train_loss.AE": loss.item(),
                            "used_epsilon.AE": self._eps_ae,
                            "used_epsilon.D": 0.0,
                            "used_epsilon.total": self._eps_ae,
                        },
                        step=self._elapsed_batches,
                    )

            self._eps_ae = self.privacy_engine_AE.get_epsilon(self._delta)
            print(f"AE pretrain used {self._eps_ae} epsilon of budget")

        # Ensure that models are in training mode
        self.G.train()
        self.decoder.train()

        # GAN training
        for epoch in range(n_epochs):
            self._elapsed_epochs += 1
            for i, X in enumerate(train_data):
                _eps = self.privacy_engine_D.get_epsilon(self._delta)

                self._eps = _eps + self._eps_ae
                if self._eps >= self._epsilon:
                    print(f"Privacy budget reached in epoch {epoch}")
                    return self

                self._elapsed_batches += 1

                real_X = Variable(X.type(Tensor))

                # Train Discriminator
                self.optim_D.zero_grad()
                z = torch.randn(real_X.size(0), self.latent_dim, device=self.device)
                # This D(G(z)) seems like the wrong approach, but it's what the original author does here: https://github.com/DPautoGAN/DPautoGAN/blob/master/uci/uci.ipynb
                fake = self.decoder(self.G(z)).detach()
                real_validity = self.D(real_X)
                fake_validity = self.D(fake)
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
                d_loss.backward()
                self.optim_D.step()

                # Train Generator
                z = torch.randn(X.size(0), self.latent_dim, device=self.device)
                fake = self.decoder(self.G(z))
                self.optim_G.zero_grad()
                g_loss = -torch.mean(self.D(fake))
                g_loss.backward()
                self.optim_G.step()

                if i % 50 == 0 and self.mlflow_logging:
                    mlflow.log_metrics(
                        {
                            "elapsed_batches": self._elapsed_batches,
                            "elapsed_epochs": self._elapsed_epochs,
                            "train_loss.G": g_loss.item(),
                            "train_loss.D": d_loss.item(),
                            "used_epsilon.D": _eps,
                            "used_epsilon.total": self._eps,
                            "validity.fake": fake_validity.mean().item(),
                            "validity.real": real_validity.mean().item(),
                        },
                        step=self._elapsed_batches,
                    )
                    # Weight and Grad norms for G and D
                    _g_norm_dict = calc_norm_dict(self.G)
                    mlflow.log_metrics(_g_norm_dict, step=self._elapsed_batches)
                    _d_norm_dict = calc_norm_dict(self.D)
                    mlflow.log_metrics(_d_norm_dict, step=self._elapsed_batches)
                    # Diversity of fakes from G and real data
                    mlflow.log_metrics(
                        {
                            "X_fake.norm": fake.norm().item(),
                            "X_real.norm": real_X.norm().item(),
                        },
                        step=self._elapsed_batches,
                    )

            print(
                (
                    f"{epoch}\t[D loss: {d_loss.item()}]\t"
                    f"[G loss: {g_loss.item()}]\t"
                    f"Eps:{self._eps:.3f}"
                )
            )

        return self

    def sample(self, n=None):

        # Set evaluation mode
        self.G.eval()
        self.decoder.eval()
        n = self.batch_size if n is None else n
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        with torch.no_grad():
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (n, self.latent_dim))))

            out = self.decoder(self.G(z))
        self.G.train()
        self.decoder.train()

        return out.detach().cpu()
