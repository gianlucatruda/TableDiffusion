"""
Based on PATE-GAN implementations from
https://github.com/opendp/smartnoise-sdk/blob/f51f7ff9819b5f1fb6764e46d0611c7f85b8f9eb/synth/snsynth/pytorch/nn/pategan.py
"""

import math

import mlflow
import numpy as np
import pandas as pd
import torch
from models.architectures import Discriminator, Generator
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from utilities import *
from utilities.privacy_utils import moments_acc, pate


class PATEGAN_Synthesiser:
    def __init__(
        self,
        batch_size=64,
        gen_lr=2e-4,
        dis_lr=2e-4,
        latent_dim=100,
        gen_dims=(256, 256),
        dis_dims=(256, 256),
        num_teachers=20,
        teacher_iters=5,
        student_iters=5,
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
        self.latent_dim = latent_dim
        self.gen_dims = gen_dims
        self.dis_dims = dis_dims
        self.num_teachers = num_teachers
        self.teacher_iters = teacher_iters
        self.student_iters = student_iters
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

    def fit(self, data, n_epochs=10, epsilon=100, noise_multiplier=1e-3):

        self._epsilon = epsilon

        if isinstance(data, pd.DataFrame):
            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="ignore")
            data = data.to_numpy()
        elif not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array or pandas dataframe")

        self.data_dim = data.shape[-1]

        # Check if training for first time or continuing (don't reinitialise)
        if self._elapsed_batches == 0:

            self._epsilon = epsilon
            self._alphas = torch.tensor([0.0 for _ in range(100)])
            self._l_list = 1 + torch.tensor(range(100))

            self.G = (
                Generator(self.latent_dim, self.data_dim, gen_dims=self.gen_dims)
                .double()
                .to(self.device)
            )
            self.G.apply(weights_init)

            self.D = Discriminator(self.data_dim, dis_dims=self.dis_dims).double().to(self.device)
            self.D.apply(weights_init)

            self.teacher_disc = [
                Discriminator(self.data_dim).double().to(self.device)
                for _ in range(self.num_teachers)
            ]
            for i in range(self.num_teachers):
                self.teacher_disc[i].apply(weights_init)

            self._nparams_g = count_parameters(self.G)
            self._nparams_d = count_parameters(self.D)
            self._nparams_teacher_d = count_parameters(self.teacher_disc[0])

            if self.mlflow_logging:
                mlflow.log_params(
                    {
                        "nparams.G": self._nparams_g,
                        "nparams.D": self._nparams_d,
                        "nparams.teacher_D": self._nparams_teacher_d,
                        "nparams.total": self._nparams_teacher_d * self.num_teachers
                        + self._nparams_g
                        + self._nparams_d,
                    }
                )
                _model_desc = str(self.G) + "\n" + str(self.D) + "\n" + str(self.teacher_disc[0])
                mlflow.log_text(_model_desc, "model_desription.txt")

            self.optimiser_g = optim.Adam(self.G.parameters(), lr=self.gen_lr)
            self.optimiser_s = optim.Adam(self.D.parameters(), lr=self.dis_lr)
            self.optimiser_t = [
                optim.Adam(self.teacher_disc[i].parameters(), lr=1e-4)
                for i in range(self.num_teachers)
            ]

            # Log optimiser parameters
            if self.mlflow_logging:
                _param_dict = gather_object_params(self.optimiser_g, prefix="optim_G.")
                mlflow.log_params(_param_dict)
                _param_dict = gather_object_params(self.optimiser_t[0], prefix="optim_D.")
                mlflow.log_params(_param_dict)

        criterion = nn.BCELoss()

        data_partitions = np.array_split(data, self.num_teachers)
        tensor_partitions = [
            TensorDataset(torch.from_numpy(data.astype("double")).to(self.device))
            for data in data_partitions
        ]

        loader = [
            DataLoader(
                tensor_partitions[teacher_id],
                batch_size=self.batch_size,
                shuffle=True,
            )
            for teacher_id in range(self.num_teachers)
        ]
        for epoch in range(n_epochs):
            if float(self._eps) >= self._epsilon:
                print(f"Privacy budget reached in epoch {epoch} ({self._eps})")
                return self

            self._elapsed_epochs += 1

            # train teacher discriminators
            for _ in range(self.teacher_iters):
                self._elapsed_batches += 1
                teacher_losses_real, teacher_losses_fake = [], []
                for i in range(self.num_teachers):
                    real_data = None
                    for _, _data in enumerate(loader[i], 0):
                        real_data = _data[0].to(self.device)
                        break

                    self.optimiser_t[i].zero_grad()

                    # train with real data
                    label_real = torch.full(
                        (real_data.shape[0],), 1, dtype=torch.float, device=self.device
                    )
                    output = self.teacher_disc[i](real_data)
                    loss_t_real = criterion(output.squeeze(), label_real.double())
                    loss_t_real.backward()
                    teacher_losses_real.append(loss_t_real.item())

                    # train with fake data
                    noise = torch.rand(self.batch_size, self.latent_dim, device=self.device)
                    label_fake = torch.full(
                        (self.batch_size,), 0, dtype=torch.float, device=self.device
                    )
                    fake_data = self.G(noise.double())
                    output = self.teacher_disc[i](fake_data)
                    loss_t_fake = criterion(output.squeeze(), label_fake.double())
                    loss_t_fake.backward()
                    teacher_losses_fake.append(loss_t_fake.item())

                    self.optimiser_t[i].step()

                if self.mlflow_logging:
                    mlflow.log_metrics(
                        {
                            "elapsed_batches": self._elapsed_batches,
                            "elapsed_epochs": self._elapsed_epochs,
                            "train_loss.teachers.real": np.mean(teacher_losses_real),
                            "train_loss.teachers.fake": np.mean(teacher_losses_fake),
                        },
                        step=self._elapsed_batches,
                    )

            # train student discriminator
            for _ in range(self.student_iters):
                self._elapsed_batches += 1
                noise = torch.rand(self.batch_size, self.latent_dim, device=self.device)
                fake_data = self.G(noise.double())
                predictions, votes = pate(fake_data, self.teacher_disc, noise_multiplier)
                output = self.D(fake_data.detach())

                # update moments accountant
                self._alphas = self._alphas + moments_acc(
                    self.num_teachers, votes, noise_multiplier, self._l_list
                )

                loss_s = criterion(output.squeeze(), predictions.to(self.device).squeeze())
                self.optimiser_s.zero_grad()
                loss_s.backward()
                self.optimiser_s.step()

                if self.mlflow_logging:
                    mlflow.log_metrics(
                        {"elapsed_batches": self._elapsed_batches, "train_loss.D": loss_s.item()},
                        step=self._elapsed_batches,
                    )
                    # Weight and Grad norms
                    _d_norm_dict = calc_norm_dict(self.D)
                    mlflow.log_metrics(_d_norm_dict, step=self._elapsed_batches)

            self._eps = (
                min((self._alphas - math.log(self._delta)) / self._l_list).detach()
            ).item()

            if self.mlflow_logging:
                mlflow.log_metrics({"used_epsilon.total": self._eps}, step=self._elapsed_batches)

            if float(self._eps) >= self._epsilon:
                print(f"Privacy budget reached before training G in epoch {epoch} ({self._eps})")
                return self

            # train generator
            self.G.train()
            label_g = torch.full((self.batch_size,), 1, dtype=torch.float, device=self.device)
            noise = torch.rand(self.batch_size, self.latent_dim, device=self.device)
            fake_data = self.G(noise.double())
            fake_validity = self.D(fake_data)
            loss_g = criterion(fake_validity.squeeze(), label_g.double())
            self.optimiser_g.zero_grad()
            loss_g.backward()
            self.optimiser_g.step()

            if self.mlflow_logging:
                mlflow.log_metrics(
                    {
                        "elapsed_batches": self._elapsed_batches,
                        "train_loss.G": loss_g.item(),
                        "used_epsilon.total": self._eps,
                        "validity.fake": fake_validity.mean().item(),
                    },
                    step=self._elapsed_batches,
                )
                # Weight and Grad norms
                _g_norm_dict = calc_norm_dict(self.G)
                mlflow.log_metrics(_g_norm_dict, step=self._elapsed_batches)
                # Diversity of fakes from G and real data
                mlflow.log_metrics(
                    {
                        "X_fake.norm": fake_data.norm().item(),
                        "X_real.norm": real_data.norm().item(),
                    },
                    step=self._elapsed_batches,
                )

        return self

    def sample(self, n=None):

        self.G.eval()
        n = self.batch_size if n is None else n
        steps = n // self.batch_size + 1
        data = []

        for _ in range(steps):
            noise = torch.randn(self.batch_size, self.latent_dim, device=self.device)
            noise = noise.view(-1, self.latent_dim)

            fake_data = self.G(noise.double())
            data.append(fake_data.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]
        self.G.train()

        return data
