"""
Building on the state-of-the-art SAINT architecture
(https://github.com/somepago/saint), we implemented an end-to-end
attention model for synthesising tabular data under differential privacy.

https://arxiv.org/abs/2308.14784

@article{truda2023generating,
  title={Generating tabular datasets under differential privacy},
  author={Truda, Gianluca},
  journal={arXiv preprint arXiv:2308.14784},
  year={2023}
}
"""

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from models.architectures import Decoder, Encoder
from models.saint_ae import SAINT_AE
from opacus import PrivacyEngine
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader
from utilities import *
from utilities.saint_utils import DataSetCatCon, embed_data_mask, mdn_loss


class DPattentionVAE_Synthesiser:
    def __init__(
        self,
        batch_size=512,
        embedding_size=64,
        b1=0.5,
        b2=0.999,
        latent_dim=100,
        dec_dims=(256, 256),
        enc_dims=(256, 256),
        n_critic=5,
        lr_trans=0.0005,
        lr_mlps=0.01,
        lr_vae=1e-3,
        transformer_depth=1,
        attention_heads=3,
        mdn=True,
        mdn_heads=7,
        loss_weight_cat=1.0,
        loss_weight_cont=1.0,
        loss_weight_kld=1.0,
        max_grad_norm=1.0,
        epsilon_target=5,
        epoch_target=5,
        pretrain_epochs=0,
        delta=1e-5,
        mlflow_logging=True,
        cuda=True,
        **kwargs,
    ):
        # Setting up GPU (if available and specified)
        if cuda:
            assert torch.cuda.is_available()
        self.cuda = cuda
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and cuda else "cpu")

        # Hyperparameters
        self.batch_size = batch_size
        self.lr_trans = lr_trans
        self.lr_mlps = lr_mlps
        self.embedding_size = embedding_size
        self.lr_vae = lr_vae
        self.b1, self.b2 = b1, b2
        self.latent_dim = latent_dim
        self.dec_dims = dec_dims
        self.enc_dims = enc_dims
        self.n_critic = n_critic
        self.transformer_depth = transformer_depth
        self.attention_heads = attention_heads
        self.mdn = mdn
        self.mdn_heads = mdn_heads
        self.lw_cat = loss_weight_cat
        self.lw_cont = loss_weight_cont
        self.lw_reg = loss_weight_kld
        self.max_grad_norm = max_grad_norm
        self.epoch_target = epoch_target
        self.pretrain_epochs = pretrain_epochs

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

    def fit(
        self, train_data, discrete_columns=(), n_epochs=7, epsilon=100, verbose=True, **kwargs
    ):

        self._epsilon = epsilon
        self.data_dim = train_data.shape[-1]
        self.data_n = train_data.shape[0]

        if verbose:
            print(f"Device is {self.device}.")

        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        self.dset_shape = train_data.shape
        self.colnames = train_data.columns

        self.disc_cols = list(discrete_columns)
        self.numeric_cols = [c for c in self.colnames if c not in self.disc_cols]
        self.cat_idxs = [i for i, c in enumerate(self.colnames) if c in self.disc_cols]

        # The cardinality of each categorical feature
        cat_dims = [train_data.iloc[:, i].nunique() for i in self.cat_idxs]

        self.con_idxs = [i for i, _ in enumerate(self.colnames) if i not in self.cat_idxs]

        # Encode categoricals
        self.cat_label_encs = []
        for i in self.cat_idxs:
            l_enc = LabelEncoder()
            train_data.iloc[:, i] = l_enc.fit_transform(train_data.values[:, i])
            # store labels for inverse transform
            self.cat_label_encs.append(l_enc)

        # TODO do I even need this still?
        train_mean = train_data.values[:, self.con_idxs].mean()
        train_std = train_data.values[:, self.con_idxs].std()
        continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32)

        # Appending 1 for CLS token, this is later used to generate embeddings.
        cat_dims = np.append(np.array([1]), np.array(cat_dims)).astype(int)

        # Check if training for first time or continuing (don't reinitialise)
        if self._elapsed_batches == 0:

            # Initialise transformer-based Autoencoder
            self.AE = SAINT_AE(
                categories=tuple(cat_dims),
                num_continuous=len(self.con_idxs),
                dim=self.embedding_size,
                transformer_depth=self.transformer_depth,
                attention_heads=self.attention_heads,
                mdn=self.mdn,
                mdn_heads=self.mdn_heads,
            ).to(self.device)

            # Initialise generator and discriminator
            self.enc = Encoder(self.embedding_size, self.enc_dims, self.latent_dim).to(self.device)
            self.dec = Decoder(self.latent_dim, self.dec_dims, self.embedding_size).to(self.device)

            # Count and log model parameters
            self._nparams_ae = count_parameters(self.AE)
            self._nparams_trans = count_parameters(self.AE.transformer, verbose=False)
            self._nparams_mlp1 = count_parameters(self.AE.mlp1, verbose=False)
            self._nparams_mlp2 = count_parameters(self.AE.mlp2, verbose=False)
            self._nparams_enc = count_parameters(self.enc)
            self._nparams_dec = count_parameters(self.dec)
            self.total_params = self._nparams_dec + self._nparams_enc + self._nparams_ae

            if self.mlflow_logging:
                mlflow.log_params(
                    {
                        "nparams.AE": self._nparams_ae,
                        "nparams.trans": self._nparams_trans,
                        "nparams.mlp1": self._nparams_mlp1,
                        "nparams.mlp2": self._nparams_mlp2,
                        "nparams.enc": self._nparams_enc,
                        "nparams.dec": self._nparams_dec,
                        "nparams.total": self.total_params,
                    }
                )
                _model_desc = str(self.AE) + "\n" + str(self.dec) + "\n" + str(self.enc)
                mlflow.log_text(_model_desc, "model_desription.txt")

            # Make the dataset and dataloader
            train_ds = DataSetCatCon(
                train_data,
                self.cat_idxs,
                # continuous_mean_std, # Normalisation
            )
            trainloader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
            self.trainloader = trainloader  # Needed later for pseudo-sampling

            # Initialise optimisers
            self.optim_trans = optim.AdamW(self.AE.transformer.parameters(), lr=self.lr_trans)
            self.optim_mlp1 = optim.Adam(self.AE.mlp1.parameters(), lr=self.lr_mlps)
            self.optim_mlp2 = optim.Adam(self.AE.mlp2.parameters(), lr=self.lr_mlps)
            self.optim_VAE = torch.optim.Adam(
                list(self.enc.parameters()) + list(self.dec.parameters()),
                lr=self.lr_vae,
            )

            # Initialise and attach privacy engines
            self.privacy_engine_1 = PrivacyEngine(accountant="rdp", secure_mode=False)
            (
                self.AE.mlp1,
                self.optim_mlp1,
                trainloader,
            ) = self.privacy_engine_1.make_private_with_epsilon(
                module=self.AE.mlp1,
                optimizer=self.optim_mlp1,
                data_loader=trainloader,
                target_epsilon=self.epsilon_target / 2,
                target_delta=self._delta,
                epochs=self.epoch_target,
                max_grad_norm=self.max_grad_norm,
                poisson_sampling=True,
            )
            self.privacy_engine_2 = PrivacyEngine(accountant="rdp", secure_mode=False)
            (
                self.AE.mlp2,
                self.optim_mlp2,
                trainloader,
            ) = self.privacy_engine_2.make_private_with_epsilon(
                module=self.AE.mlp2,
                optimizer=self.optim_mlp2,
                data_loader=trainloader,
                target_epsilon=self.epsilon_target / 2,
                target_delta=self._delta,
                epochs=self.epoch_target - self.pretrain_epochs,
                max_grad_norm=self.max_grad_norm,
                poisson_sampling=True,
            )

            # Log privacy engine and optimiser parameters
            if self.mlflow_logging:
                _param_dict = gather_object_params(
                    self.privacy_engine_1, prefix="privacy_engine_1."
                )
                mlflow.log_params(_param_dict)
                _param_dict = gather_object_params(self.optim_mlp1, prefix="optim_mlp1.")
                mlflow.log_params(_param_dict)
                _param_dict = gather_object_params(
                    self.privacy_engine_2, prefix="privacy_engine_2."
                )
                mlflow.log_params(_param_dict)
                _param_dict = gather_object_params(self.optim_mlp2, prefix="optim_mlp2.")
                mlflow.log_params(_param_dict)

        # Loss criteria
        criterion1 = nn.CrossEntropyLoss()
        criterion2 = nn.MSELoss()

        # Enforce training mode
        self.AE.train()
        self.dec.train()
        self.enc.train()

        if verbose:
            print("Training...")

        for epoch in range(n_epochs):
            self._elapsed_epochs += 1
            for i, data in enumerate(trainloader, 0):
                self._eps1 = self.privacy_engine_1.get_epsilon(self._delta)
                self._eps2 = self.privacy_engine_2.get_epsilon(self._delta)

                # TODO additive?
                self._eps = self._eps2 + self._eps1
                if self._eps >= self._epsilon:
                    print(f"Privacy budget reached in epoch {epoch}")
                    return self

                loss = 0
                self._elapsed_batches += 1

                # Zero gradients for all optimisers
                self.optim_trans.zero_grad()
                self.optim_mlp1.zero_grad()
                self.optim_mlp2.zero_grad()
                self.optim_VAE.zero_grad()

                """
                `x_categ` is the the categorical data (batch_size x cat_cols+1)
                `x_cont` has continuous data (batch_size x cont_cols)
                `cat_mask` is an array of ones same shape as x_categ and an additional column(corresponding to CLS token) set to 0s (batch_size, cat_cols+1)
                `con_mask` is an array of ones same shape as x_cont (batch_size, cont_cols)
                """
                x_categ, x_cont, cat_mask, con_mask = (
                    data[0].to(self.device),
                    data[1].to(self.device),
                    data[2].to(self.device),
                    data[3].to(self.device),
                )
                _, x_categ_enc_2, x_cont_enc_2 = embed_data_mask(
                    x_categ, x_cont, cat_mask, con_mask, self.AE
                )

                # Trim CLS token off categorical targets
                x_categ = x_categ[:, 1:]

                # Forward pass
                x_prime = self.AE.encode(x_categ_enc_2, x_cont_enc_2)

                if epoch < self.pretrain_epochs:
                    x_prime_fake = x_prime
                else:
                    mu, std, logvar = self.enc(x_prime)
                    x_prime_fake, _ = self.dec(torch.randn_like(std) * std + mu)
                cat_outs, con_outs = self.AE.decode(x_prime_fake)

                l2 = 0
                if len(con_outs) > 0:
                    if self.mdn:
                        # MDN loss
                        l2 = 0
                        for _i, con in enumerate(con_outs):
                            pi, mu, sigma = con
                            l_mdn = mdn_loss(x_cont[:, _i], pi, mu, sigma)
                            l2 += l_mdn
                    else:
                        con_outs = torch.cat(con_outs, dim=1)
                        # Sqrt makes into RMSE loss
                        l2 = torch.sqrt(criterion2(con_outs, x_cont))
                l1 = 0
                n_cat = x_categ.shape[-1]
                for j in range(n_cat):
                    x_probs = torch.full(cat_outs[j].shape, 0.01).to(self.device)
                    for _i in range(cat_outs[j].shape[0]):
                        x_probs[_i, x_categ[_i, j]] = 0.99
                    l1 += criterion1(cat_outs[j], x_probs)

                # Reconstruction loss
                loss += self.lw_cat * l1 + self.lw_cont * l2

                if epoch >= self.pretrain_epochs:
                    # Add KLD for regularisation loss
                    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
                    loss += self.lw_reg * KLD
                else:
                    KLD = torch.Tensor([0])

                # Early stop on collapsed loss
                # if loss.isnan():
                #     raise RuntimeError("Loss is NaN")
                # if loss.isinf():
                #     raise RuntimeError("Loss is inf")

                loss.backward()

                self.optim_mlp1.step()
                self.optim_mlp2.step()
                if epoch < self.pretrain_epochs:
                    self.optim_trans.step()
                else:
                    self.optim_VAE.step()

                if i % 10 == 0 and self.mlflow_logging:
                    mlflow.log_metrics(
                        {
                            "elapsed_batches": self._elapsed_batches,
                            "elapsed_epochs": self._elapsed_epochs,
                            "loss.l1": l1.item(),
                            "loss.l2": l2.item(),
                            "loss.KLD": KLD.item(),
                            "loss.Total": loss.item(),
                            "used_epsilon.mlp1": self._eps1,
                            "used_epsilon.mlp2": self._eps2,
                            "used_epsilon.total": self._eps,
                            # "best_alpha1": self._best_alpha1,
                            # "best_alpha2": self._best_alpha2,
                        },
                        step=self._elapsed_batches,
                    )
                    # Weight and Grad norms
                    mlflow.log_metrics(calc_norm_dict(self.AE), step=self._elapsed_batches)
                    mlflow.log_metrics(calc_norm_dict(self.enc), step=self._elapsed_batches)
                    mlflow.log_metrics(calc_norm_dict(self.dec), step=self._elapsed_batches)

            if verbose:
                print(f"Epoch: {epoch}, Total Loss: {loss}")

        if self.mlflow_logging:
            mlflow.log_metrics(
                {
                    "elapsed_batches": self._elapsed_batches,
                    "elapsed_epochs": self._elapsed_epochs,
                    "loss.l1": l1.item(),
                    "loss.l2": l2.item(),
                    "loss.KLD": KLD.item(),
                    "loss.Total": loss.item(),
                    "used_epsilon.mlp1": self._eps1,
                    "used_epsilon.mlp2": self._eps2,
                    "used_epsilon.total": self._eps,
                    # "best_alpha1": self._best_alpha1,
                    # "best_alpha2": self._best_alpha2,
                },
                step=self._elapsed_batches,
            )

        if verbose:
            print("END OF AE TRAINING!")

        return self

    def sample(self, n=None):

        # Set evaluation mode
        self.AE.eval()
        self.dec.eval()
        self.enc.eval()

        n = self.batch_size if n is None else n
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        cat_samples, con_samples = [[] for i in self.cat_idxs], [[] for i in self.con_idxs]

        with torch.no_grad():
            # Sample noise z
            _m = torch.zeros(self.batch_size, self.latent_dim)
            _s = _m + 1
            z = torch.normal(mean=_m, std=_s).to(self.device)

            # Pass z to Decoder, then inverse trasformation (MLPs)
            x_hat, sigmas = self.dec(z)
            cat_tx, con_tx = self.AE.inv_transform(x_hat)

            # Set output
            for j, d in enumerate(cat_tx):
                cat_samples[j] = list(d.cpu().flatten().numpy())
            for j, d in enumerate(con_tx):
                con_samples[j] = list(d.cpu().flatten().numpy())

        # convert labels back to strings
        cat_samples = [
            self.cat_label_encs[i].inverse_transform(c) for i, c in enumerate(cat_samples)
        ]

        # Rebuild columns in correct order
        cols = []
        for i in range(self.dset_shape[1]):
            if i in self.cat_idxs:
                col = cat_samples[0]
                cat_samples = cat_samples[1:]
            else:
                col = con_samples[0]
                con_samples = con_samples[1:]
            cols.append(col)

        # Convert to dataframe
        df = pd.DataFrame(cols).T.sample(n, replace=True)
        df.columns = self.colnames
        # Try cast non-discrete columns to numeric types
        df[self.numeric_cols] = df[self.numeric_cols].apply(pd.to_numeric, errors="ignore")
        # Apply rounding
        df[self.numeric_cols] = df[self.numeric_cols].round(1)

        # Re-enable training mode
        self.AE.train()
        self.dec.train()
        self.enc.train()

        return df
