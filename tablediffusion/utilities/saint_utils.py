"""
Supporting code for SAINT-based models.
Adapted from https://github.com/somepago/saint
"""

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn
from torch.distributions.normal import Normal
from torch.utils.data import Dataset


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=16, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)


class RowColTransformer(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim,
        nfeats,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout,
        style="col",
    ):
        super().__init__()
        self.embeds = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])
        self.mask_embed = nn.Embedding(nfeats, dim)
        self.style = style
        for _ in range(depth):
            if self.style == "colrow":
                self.layers.append(
                    nn.ModuleList(
                        [
                            PreNorm(
                                dim,
                                Residual(
                                    Attention(
                                        dim, heads=heads, dim_head=dim_head, dropout=attn_dropout
                                    )
                                ),
                            ),
                            PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout))),
                            PreNorm(
                                dim * nfeats,
                                Residual(
                                    Attention(
                                        dim * nfeats,
                                        heads=heads,
                                        dim_head=64,
                                        dropout=attn_dropout,
                                    )
                                ),
                            ),
                            PreNorm(
                                dim * nfeats,
                                Residual(FeedForward(dim * nfeats, dropout=ff_dropout)),
                            ),
                        ]
                    )
                )
            else:
                self.layers.append(
                    nn.ModuleList(
                        [
                            PreNorm(
                                dim * nfeats,
                                Residual(
                                    Attention(
                                        dim * nfeats,
                                        heads=heads,
                                        dim_head=64,
                                        dropout=attn_dropout,
                                    )
                                ),
                            ),
                            PreNorm(
                                dim * nfeats,
                                Residual(FeedForward(dim * nfeats, dropout=ff_dropout)),
                            ),
                        ]
                    )
                )

    def forward(self, x, x_cont=None, mask=None):
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        _, n, _ = x.shape
        if self.style == "colrow":
            for attn1, ff1, attn2, ff2 in self.layers:
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, "b n d -> 1 b (n d)")
                x = attn2(x)
                x = ff2(x)
                x = rearrange(x, "1 b (n d) -> b n d", n=n)
        else:
            for attn1, ff1 in self.layers:
                x = rearrange(x, "b n d -> 1 b (n d)")
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, "1 b (n d) -> b n d", n=n)
        return x


class Transformer(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Residual(
                                Attention(
                                    dim, heads=heads, dim_head=dim_head, dropout=attn_dropout
                                )
                            ),
                        ),
                        PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout))),
                    ]
                )
            )

    def forward(self, x, x_cont=None):
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class MLP(nn.Module):
    def __init__(self, dims, act=None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue
            if act is not None:
                layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class simple_MLP(nn.Module):
    def __init__(self, dims, act=None):
        super(simple_MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]), nn.ReLU(), nn.Linear(dims[1], dims[2])
        )
        if act is not None:
            self.layers.add_module(act())

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


class sep_MLP(nn.Module):
    def __init__(self, dim, len_feats, categories, mdn_heads=0):
        super(sep_MLP, self).__init__()
        self.len_feats = len_feats
        self.mdn = mdn_heads > 0
        self.mdn_heads = mdn_heads
        self.layers = nn.ModuleList([])
        if self.mdn:
            self.mdns = nn.ModuleList([])
        for i in range(len_feats):
            if self.mdn:
                self.layers.append(simple_MLP([dim, 5 * dim, 16]))
                self.mdns.append(MixtureDensityNetwork(16, categories[i], self.mdn_heads))
            else:
                self.layers.append(simple_MLP([dim, 5 * dim, categories[i]]))

    def forward(self, x):
        y_pred = []
        for i in range(self.len_feats):
            pred = self.layers[i](x)
            if self.mdn:
                pred = self.mdns[i](pred)
            y_pred.append(pred)
        return y_pred


class MixtureDensityNetwork(nn.Module):
    """
    Mixture density network (Bishop, 1994).
    Adapted from https://github.com/pdogr/pytorch-MDN

    Parameters
    ----------
    dim_in; dimensionality of the input
    dim_out: int; dimensionality of the output
    num_latent: int; number of components in the mixture model

    Output
    ----------
    (pi,mu,sigma)
    pi (batch_size x num_latent) is prior
    mu (batch_size x dim_out x num_latent) is mean of each Gaussian
    sigma (batch_size x dim_out x num_latent) is standard deviation of each Gaussian
    """

    def __init__(self, dim_in, dim_out, num_latent):
        super(MixtureDensityNetwork, self).__init__()
        self.dim_in = dim_in
        self.num_latent = num_latent
        self.dim_out = dim_out
        self.pi_h = nn.Linear(dim_in, num_latent)
        self.mu_h = nn.Linear(dim_in, dim_out * num_latent)
        self.sigma_h = nn.Linear(dim_in, dim_out * num_latent)

    def forward(self, x):
        x = torch.tanh(x)

        pi = self.pi_h(x)
        pi = F.softmax(pi, dim=-1)

        mu = self.mu_h(x)
        mu = mu.view(-1, self.dim_out, self.num_latent)

        sigma = torch.exp(self.sigma_h(x))
        sigma = sigma.view(-1, self.dim_out, self.num_latent)

        return pi, mu, sigma


def mdn_loss(y, pi, mu, sigma):
    """Mixture Density Net loss
    Adapted from https://github.com/DuaneNielsen/mixturedensity
    """
    _y = y if len(y.shape) == 2 else y.view(-1, 1)

    mixture = Normal(mu, sigma)
    log_prob = mixture.log_prob(_y)
    weighted_log_prob = log_prob + torch.log(pi)
    log_prob_loss = -torch.sum(weighted_log_prob, dim=1)
    return torch.mean(log_prob_loss)


class DataSetCatCon(Dataset):
    def __init__(self, X, cat_cols, continuous_mean_std=None):

        # Don't modify the original dataset
        _X = X.copy()

        cat_cols = list(cat_cols)
        X_mask = np.ones_like(X.values, dtype=int)
        _X = _X.values
        self.length = int(_X.shape[0])
        con_cols = list(set(np.arange(_X.shape[1])) - set(cat_cols))
        self.X1 = _X[:, cat_cols].copy().astype(np.int64)  # categorical columns
        self.X2 = _X[:, con_cols].copy().astype(np.float32)  # numerical columns
        self.X1_mask = X_mask[:, cat_cols].copy().astype(np.int64)  # categorical columns
        self.X2_mask = X_mask[:, con_cols].copy().astype(np.int64)  # numerical columns

        null_dim = _X[:, 0]
        self.cls = np.zeros_like(null_dim, dtype=int)
        self.cls = self.cls[..., np.newaxis]
        self.cls_mask = np.ones_like(null_dim, dtype=int)
        self.cls_mask = self.cls_mask[..., np.newaxis]

        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        # print(self.cls.shape, self.cls_mask.shape, self.X1.shape, self.X2.shape)
        return (
            np.concatenate((self.cls[idx], self.X1[idx])),
            self.X2[idx],
            np.concatenate((self.cls_mask[idx], self.X1_mask[idx])),
            self.X2_mask[idx],
        )


def embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model):
    # TODO understand what this does and how
    device = x_cont.device
    # How does this offsetting work? Why?
    x_categ = x_categ + model.categories_offset.type_as(x_categ)
    x_categ_enc = model.embeds(x_categ)
    n1, n2 = x_cont.shape
    _, n3 = x_categ.shape
    if model.cont_embeddings == "MLP":
        # Embeds each continuous feature into a 32-dim latent space
        x_cont_enc = torch.empty(n1, n2, model.dim)
        for i in range(model.num_continuous):
            x_cont_enc[:, i, :] = model.simple_MLP[i](x_cont[:, i])
    else:
        raise Exception("This case should not work!")

    x_cont_enc = x_cont_enc.to(device)

    return x_categ, x_categ_enc, x_cont_enc


def parse_mixed_dataset(*args, **kwargs):
    return None
