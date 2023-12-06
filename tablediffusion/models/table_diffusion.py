"""
Code for the `TableDiffusion` model:
The first differentially-private diffusion model for tabular datasets.

https://arxiv.org/abs/2308.14784

@article{truda2023generating,
  title={Generating tabular datasets under differential privacy},
  author={Truda, Gianluca},
  journal={arXiv preprint arXiv:2308.14784},
  year={2023}
}
"""

import warnings

import mlflow
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from models.architectures import Generator
from opacus import PrivacyEngine
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utilities import *

# Ignore opacus hook warnings
warnings.filterwarnings(
    "ignore",
    message="Using a non-full backward hook when the forward contains multiple autograd Nodes",
)

# Function to compute the cosine noise schedule
def get_beta(t, T):
    return (1 - np.cos((np.pi * t) / T)) / 2 + 0.1

class MixedTypeGenerator(Generator):
    def __init__(
        self,
        embedding_dim,
        data_dim,
        gen_dims=(256, 256),
        predict_noise=True,
        categorical_start_idx=None,
        cat_counts=None,
    ):
        # Initialise parent (Generator) with the parameters
        super().__init__(embedding_dim, data_dim, gen_dims)
        self.categorical_start_idx = categorical_start_idx
        self.cat_counts = cat_counts
        self.predict_noise = predict_noise

    def forward(self, x):
        data = self.seq(x)

        if self.predict_noise:
            # Just predicting gaussian noise
            return data

        # Split into numerical and categorical outputs
        numerical_outputs = data[:, : self.categorical_start_idx]
        categorical_outputs = data[:, self.categorical_start_idx :]
        _idx = 0
        # Softmax over each category
        for k, v in self.cat_counts.items():
            categorical_outputs[:, _idx : _idx + v] = torch.softmax(
                categorical_outputs[:, _idx : _idx + v], dim=-1
            )
            _idx += v
        return torch.cat((numerical_outputs, categorical_outputs), dim=-1)


class TableDiffusion_Synthesiser:
    def __init__(
        self,
        batch_size=1024,
        lr=0.005,
        b1=0.5,
        b2=0.999,
        dims=(128, 128),
        diffusion_steps=5,
        predict_noise=True,
        max_grad_norm=1.0,
        epsilon_target=1.0,
        epoch_target=5,
        delta=1e-5,
        sample_img_interval=None,
        mlflow_logging=True,
        cuda=True,
    ):
        from datetime import datetime
        self._now = datetime.now().strftime("%m%d%H%M%S")
        # Setting up GPU (if available and specified)
        if cuda:
            assert torch.cuda.is_available()
        self.cuda = cuda
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and cuda else "cpu")

        # Hyperparameters
        self.batch_size = batch_size
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.dims = dims
        self.diffusion_steps = diffusion_steps
        self.pred_noise = predict_noise
        self.max_grad_norm = max_grad_norm
        self.epoch_target = epoch_target
        self.sample_img_interval = sample_img_interval

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

    def fit(self, df, n_epochs=10, epsilon=100, discrete_columns=[], verbose=True):

        self._epsilon = epsilon
        self.data_dim = df.shape[1]
        self.data_n = df.shape[0]
        self.disc_columns = discrete_columns

        Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        self.q_transformers = {}
        self.encoders = {}
        self.category_counts = {}

        # Preprocessing
        self._original_types = df.dtypes
        self._original_columns = df.columns
        df_encoded = df.select_dtypes(include="number").copy()  # numerical features
        df_encoded_cat = pd.DataFrame()  # categorical features
        for col in df.columns:
            if col in self.disc_columns:
                self.encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                transformed = self.encoders[col].fit_transform(df[col].values.reshape(-1, 1))
                transformed_df = pd.DataFrame(
                    transformed, columns=[f"{col}_{i}" for i in range(transformed.shape[1])]
                )
                df_encoded_cat = pd.concat([df_encoded_cat, transformed_df], axis=1)
                # Log the number of categories for each discrete column
                self.category_counts[col] = transformed_df.shape[1]
            else:
                self.q_transformers[col] = QuantileTransformer()
                df_encoded[col] = self.q_transformers[col].fit_transform(
                    df[col].values.reshape(-1, 1)
                )
        df_encoded = pd.concat([df_encoded, df_encoded_cat], axis=1)

        categorical_start_idx = transformed_df.shape[1] + 1
        self.total_categories = sum(self.category_counts.values())
        self.encoded_columns = df_encoded.columns  # store the column names of the encoded data
        self.data_dim = df_encoded.shape[1]  # store the dimensionality of the encoded data
        self.data_n = df_encoded.shape[0]  # store the total number of data points

        # Convert df to tensor and wrap in DataLoader
        train_data = DataLoader(
            torch.from_numpy(df_encoded.values.astype(np.float32)).to(self.device),
            batch_size=self.batch_size,
            drop_last=False,
        )

        # Create MLP model
        self.model = MixedTypeGenerator(
            df_encoded.shape[1],
            self.data_dim,
            self.dims,
            self.pred_noise,
            categorical_start_idx,
            self.category_counts,
        ).to(self.device)
        if verbose:
            print(self.model)
        self._nparams = count_parameters(self.model)

        if self.mlflow_logging:
            mlflow.log_params(
                {
                    "nparams.total": self._nparams,
                }
            )

        # Initialise optimiser (and scheduler)
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=(self.b1, self.b2),
        )

        self.privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=False)
        self.model, self.optim, train_data = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optim,
            data_loader=train_data,
            target_epsilon=self.epsilon_target,
            target_delta=self._delta,
            epochs=self.epoch_target,
            max_grad_norm=self.max_grad_norm,
            poisson_sampling=True,
        )

        # Log privacy engine and optimiser parameters
        if self.mlflow_logging:
            _param_dict = gather_object_params(self.privacy_engine, prefix="privacy_engine.")
            mlflow.log_params(_param_dict)
            _param_dict = gather_object_params(self.optim, prefix="optim.")
            mlflow.log_params(_param_dict)

        # Define loss functions
        mse_loss = nn.MSELoss()
        kl_loss = nn.KLDivLoss(reduction="batchmean")

        # Enforce training mode
        self.model.train()

        # Training loop
        for epoch in range(n_epochs):
            self._elapsed_epochs += 1
            for i, X in enumerate(train_data):
                # Check if loss is NaN and early stop
                if i > 2 and loss.isnan():
                    print("Loss is NaN. Early stopping.")
                    return self
                if self.sample_img_interval is not None and i % self.sample_img_interval == 0:
                    fig, axs = plt.subplots(self.diffusion_steps, 5, figsize=(4*self.diffusion_steps, 4*5))

                self._elapsed_batches += 1

                # Configure input
                real_X = Variable(X.type(Tensor))
                agg_loss = torch.Tensor([0]).to(self.device)

                # Diffusion process with cosine noise schedule
                for t in range(self.diffusion_steps):
                    self._eps = self.privacy_engine.get_epsilon(self._delta)
                    if self._eps >= self.epsilon_target:
                        print(f"Privacy budget reached in epoch {epoch} (batch {i}, {t=}).")
                        return self
                    beta_t = get_beta(t, self.diffusion_steps)
                    noise = torch.randn_like(real_X).to(self.device) * np.sqrt(beta_t)
                    noised_data = real_X + noise
                    if self.sample_img_interval is not None and i % self.sample_img_interval == 0:
                        print(f"Epoch {epoch} (batch {i}, {t=}), {np.sqrt(beta_t)=}")

                    if self.pred_noise:
                        # Use the model as a diffusion noise predictor
                        predicted_noise = self.model(noised_data)

                        # Calculate loss between predicted and actualy noise using MSE
                        numeric_loss = mse_loss(predicted_noise, noise)
                        categorical_loss = torch.tensor(0.0)
                        loss = numeric_loss

                    else:
                        # Use the model as a mixed-type denoiser
                        denoised_data = self.model(noised_data)

                        # Calculate numeric loss using MSE
                        numeric_loss = mse_loss(
                            denoised_data[:, :categorical_start_idx],
                            real_X[:, :categorical_start_idx],
                        )

                        # Convert categoricals to log-space (to avoid underflow issue) and calculate KL loss for each original feature
                        _idx = categorical_start_idx
                        categorical_losses = []
                        for _col, _cat_len in self.category_counts.items():
                            categorical_losses.append(
                                kl_loss(
                                    torch.log(denoised_data[:, _idx : _idx + _cat_len]),
                                    real_X[:, _idx : _idx + _cat_len],
                                )
                            )
                            _idx += _cat_len

                        # Average categorical losses over total number of categories across all categorical features
                        categorical_loss = (
                            sum(categorical_losses) / self.total_categories
                            if categorical_losses
                            else 0
                        )

                        loss = numeric_loss + categorical_loss

                    if self.sample_img_interval is not None and i % self.sample_img_interval == 0:
                        with torch.no_grad():
                            ax = axs[t]
                            ax[0].imshow(X.clone().detach().cpu().numpy()); ax[0].set_title("X")
                            ax[1].imshow(noise.clone().detach().cpu().numpy()); ax[1].set_title(f"noise_{t}")
                            ax[2].imshow(noised_data.clone().detach().cpu().numpy()); ax[2].set_title(f"noised_data_{t}")
                            if self.pred_noise:
                                ax[3].imshow(predicted_noise.clone().detach().cpu().numpy()); ax[3].set_title(f"predicted_noise_{t}")
                                denoised_data = noised_data - predicted_noise*np.sqrt(beta_t)
                            ax[4].imshow(denoised_data.clone().detach().cpu().numpy()); ax[4].set_title(f"denoised_data_{t}")

                    # Add losses from each diffusion step
                    agg_loss += loss

                # Average loss over diffusion steps
                loss = agg_loss / self.diffusion_steps
                print(f"Batches: {self._elapsed_batches}, {agg_loss=}")

                # Backward propagation and optimization step
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()


                if self.sample_img_interval is not None and i % self.sample_img_interval == 0:
                    plt.savefig(f"../results/diffusion_figs/{self._now}_forward_T{self.diffusion_steps}_B{self._elapsed_batches}.png")
                    sample = self.sample(n=X.shape[0], post_process=False)
                    plt.cla(); plt.clf()
                    plt.imshow(sample)
                    plt.savefig(f"../results/diffusion_figs/{self._now}_sample_T{self.diffusion_steps}_B{self._elapsed_batches}.png")

                if i % 20 == 0:
                    if verbose:
                        print(
                            f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(train_data)}] numerical loss: {numeric_loss.item():.6f}, categorical loss: {categorical_loss.item():.6f}, epsilon: {self._eps:.6f}"
                        )
                    if self.mlflow_logging:
                        mlflow.log_metrics(
                            {
                                "elapsed_batches": self._elapsed_batches,
                                "elapsed_epochs": self._elapsed_epochs,
                                "train_loss.numerical": numeric_loss.item(),
                                "train_loss.categorical": categorical_loss.item(),
                                "train_loss.total": loss.item(),
                                "used_epsilon.total": self._eps,
                            },
                            step=self._elapsed_batches,
                        )

                        # Log weight and grad norms
                        _norm_dict = calc_norm_dict(self.model)
                        mlflow.log_metrics(_norm_dict, step=self._elapsed_batches)

        return self

    def sample(self, n=None, post_process=True):
        self.model.eval()
        n = self.batch_size if n is None else n
        # Generate noise samples
        samples = torch.randn((n, self.data_dim)).to(self.device)
        fig, axs = plt.subplots(self.diffusion_steps, 4, figsize=(4*self.diffusion_steps, 4*4))

        # Generate synthetic data by runnin reverse diffusion process
        with torch.no_grad():
            for t in range(self.diffusion_steps -1, -1, -1):
                beta_t = get_beta(t, self.diffusion_steps)
                noise_scale = np.sqrt(beta_t)
                print(f"Sampling {t=}, {np.sqrt(beta_t)=}")
                ax = axs[self.diffusion_steps - t - 1]
                ax[2].imshow(samples.clone().detach().cpu().numpy()); ax[2].set_title(f"samples_{t}")

                if self.pred_noise:
                    # Repeatedly predict and subtract noise
                    pred_noise = self.model(samples)
                    predicted_noise = pred_noise * noise_scale
                    ax[0].imshow(pred_noise.clone().detach().cpu().numpy()); ax[0].set_title(f"pred_noise_{t}")
                    ax[1].imshow(predicted_noise.clone().detach().cpu().numpy()); ax[1].set_title(f"predicted_noise_{t}")

                    samples = samples - predicted_noise
                else:
                    # Repeatedly denoise
                    samples = self.model(samples)
                ax[3].imshow(samples.clone().detach().cpu().numpy()); ax[3].set_title(f"samples_{t-1}")

        if self.sample_img_interval is not None:
            plt.savefig(f"../results/diffusion_figs/{self._now}_reverse_T{self.diffusion_steps}_B{self._elapsed_batches}.png")

        synthetic_data = samples.detach().cpu().numpy()
        self.model.train()

        if not post_process:
            return synthetic_data

        # Postprocessing: apply inverse transformations
        df_synthetic = pd.DataFrame(synthetic_data, columns=self.encoded_columns)
        for col in self.encoders:
            transformed_cols = [c for c in df_synthetic.columns if c.startswith(f"{col}_")]
            if transformed_cols:
                encoded_data = df_synthetic[transformed_cols].values
                df_synthetic[col] = self.encoders[col].inverse_transform(encoded_data).ravel()
                df_synthetic = df_synthetic.drop(columns=transformed_cols)

        for col in self.q_transformers:
            df_synthetic[col] = self.q_transformers[col].inverse_transform(
                df_synthetic[col].values.reshape(-1, 1)
            )

        # Cast to the original datatypes for dataframe compatibility
        df_synthetic = df_synthetic.astype(self._original_types)
        # Order the columns as they were in the original dataframe
        df_synthetic = df_synthetic[self._original_columns]

        return df_synthetic
