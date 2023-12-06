"""
Based on WGAN GP implementation from
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
"""

import mlflow
import numpy as np
import torch
from models.architectures import Discriminator, Generator
from opacus import PrivacyEngine
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


class WGAN_Synthesiser:
    def __init__(
        self,
        batch_size=64,
        gen_lr=2e-4,
        dis_lr=2e-4,
        b1=0.5,
        b2=0.999,
        latent_dim=100,
        gen_dims=(256, 256),
        dis_dims=(256, 256),
        n_critic=5,
        max_grad_norm=1.0,
        epsilon_target=5,
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
        self.b1 = b1
        self.b2 = b2
        self.latent_dim = latent_dim
        self.gen_dims = gen_dims
        self.dis_dims = dis_dims
        self.n_critic = n_critic
        self.max_grad_norm = max_grad_norm
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

    def fit(self, train_data, n_epochs=10, epsilon=100, verbose=True):

        self._epsilon = epsilon
        self.data_dim = train_data.shape[-1]
        self.data_n = train_data.shape[0]

        if not isinstance(train_data, DataLoader):
            train_data = DataLoader(
                MyDataset(train_data), batch_size=self.batch_size, drop_last=True
            )

        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        # Check if training for first time or continuing (don't reinitialise)
        if self._elapsed_batches == 0:

            # Initialize generator and discriminator
            self.G = Generator(self.latent_dim, self.data_dim, gen_dims=self.gen_dims).to(
                self.device
            )
            self.D = Discriminator(self.data_dim, dis_dims=self.dis_dims).to(self.device)

            self._nparams_g = count_parameters(self.G)
            self._nparams_d = count_parameters(self.D)

            if self.mlflow_logging:
                mlflow.log_params(
                    {
                        "nparams.G": self._nparams_g,
                        "nparams.D": self._nparams_d,
                        "nparams.total": self._nparams_g + self._nparams_d,
                    }
                )
                _model_desc = str(self.G) + "\n" + str(self.D)
                mlflow.log_text(_model_desc, "model_desription.txt")

            # Initialise optimisers
            self.optim_G = torch.optim.Adam(
                self.G.parameters(), lr=self.gen_lr, betas=(self.b1, self.b2)
            )
            self.optim_D = torch.optim.Adam(
                self.D.parameters(), lr=self.dis_lr, betas=(self.b1, self.b2)
            )

            self.privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=False)
            self.D, self.optim_D, train_data = self.privacy_engine.make_private_with_epsilon(
                module=self.D,
                optimizer=self.optim_D,
                data_loader=train_data,
                target_epsilon=self.epsilon_target,
                target_delta=self._delta,
                epochs=self.epoch_target,
                max_grad_norm=self.max_grad_norm,
                poisson_sampling=True,
                # alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            )

            # Log privacy engine and optimiser parameters
            if self.mlflow_logging:
                _param_dict = gather_object_params(self.privacy_engine, prefix="privacy_engine_D.")
                mlflow.log_params(_param_dict)
                _param_dict = gather_object_params(self.optim_D, prefix="optim_D.")
                mlflow.log_params(_param_dict)

        # Enforce training mode
        self.G.train()
        self.D.train()

        for epoch in range(n_epochs):
            self._elapsed_epochs += 1
            for i, X in enumerate(train_data):
                self._eps = self.privacy_engine.get_epsilon(self._delta)
                if self._eps >= self._epsilon:
                    print(f"Privacy budget reached in epoch {epoch}")
                    return self

                self._elapsed_batches += 1

                # Configure input
                real_X = Variable(X.type(Tensor))

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optim_D.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (X.shape[0], self.latent_dim))))

                # Generate a batch of samples
                fake_X = self.G(z)

                # Real samples
                real_validity = self.D(real_X)
                # Fake samples
                fake_validity = self.D(fake_X)

                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)

                d_loss.backward()
                self.optim_D.step()

                # -----------------
                #  Train Generator
                # -----------------
                self.optim_G.zero_grad()
                # Train the generator every n_critic steps
                if i % self.n_critic == 0:

                    # Generate a batch of images
                    fake_X = self.G(z)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = self.D(fake_X)
                    g_loss = -torch.mean(fake_validity)

                    g_loss.backward()
                    self.optim_G.step()

                    if i % 300 == 0 and verbose:
                        print(
                            (
                                f"[Epoch {epoch}/{n_epochs}] "
                                f"[Batch {i}/{len(train_data)}] "
                                f"[D loss: {d_loss.item():.4f}] "
                                f"[G loss: {g_loss.item():.4f}]"
                            )
                        )
                        print(f"Epsilon: {self._eps:.3f}")

                    if i % 20 == 0 and self.mlflow_logging:
                        mlflow.log_metrics(
                            {
                                "elapsed_batches": self._elapsed_batches,
                                "elapsed_epochs": self._elapsed_epochs,
                                "train_loss.G": g_loss.item(),
                                "train_loss.D": d_loss.item(),
                                "used_epsilon.D": self._eps,
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
                                "X_fake.norm": fake_X.norm().item(),
                                "X_real.norm": real_X.norm().item(),
                            },
                            step=self._elapsed_batches,
                        )

        return self

    def sample(self, n=None):
        self.G.eval()
        n = self.batch_size if n is None else n
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        with torch.no_grad():
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (n, self.latent_dim))))

        output = self.G(z).detach().cpu()
        self.G.train()

        return output
