"""
Adapted from https://github.com/somepago/saint
"""

from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch import nn
from utilities.saint_utils import (RowColTransformer, Transformer,
                                   embed_data_mask, mdn_loss, sep_MLP,
                                   simple_MLP)


class SAINT_AE(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim_head=16,
        mlp_hidden_mults=(4, 2),
        num_special_tokens=0,
        attn_dropout=0.0,
        ff_dropout=0.0,
        cont_embeddings="MLP",
        dim=32,
        transformer_depth=1,
        attention_heads=4,
        attentiontype="colrow",
        final_mlp_style="sep",
        mdn=False,
        mdn_heads=3,
    ):
        super().__init__()
        self.mdn = mdn
        self.mdn_heads = mdn_heads if self.mdn else 0
        assert all(map(lambda n: n > 0, categories)), "number of each category must be positive"

        # categories related calculations
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]

        self.register_buffer("categories_offset", categories_offset)

        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype
        self.final_mlp_style = final_mlp_style

        if self.cont_embeddings == "MLP":
            self.simple_MLP = nn.ModuleList(
                [simple_MLP([1, 100, self.dim]) for _ in range(self.num_continuous)]
            )
            input_size = (dim * self.num_categories) + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        elif self.cont_embeddings == "pos_singleMLP":
            self.simple_MLP = nn.ModuleList([simple_MLP([1, 100, self.dim]) for _ in range(1)])
            input_size = (dim * self.num_categories) + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        else:
            print("Continous features are not passed through attention")
            input_size = (dim * self.num_categories) + num_continuous
            nfeats = self.num_categories

        # transformer
        if attentiontype == "col":
            self.transformer = Transformer(
                num_tokens=self.total_tokens,
                dim=dim,
                depth=transformer_depth,
                heads=attention_heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
            )
        elif attentiontype in ["row", "colrow"]:
            self.transformer = RowColTransformer(
                num_tokens=self.total_tokens,
                dim=dim,
                nfeats=nfeats,
                depth=transformer_depth,
                heads=attention_heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                style=attentiontype,
            )

        # l = input_size // 8
        # hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))

        self.embeds = nn.Embedding(self.total_tokens, self.dim)  # .to(self.device)

        # cat_mask_offset = F.pad(
        #     torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value=0
        # )
        # cat_mask_offset = cat_mask_offset.cumsum(dim=-1)[:-1]

        # con_mask_offset = F.pad(
        #     torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value=0
        # )
        # con_mask_offset = con_mask_offset.cumsum(dim=-1)[:-1]

        # self.register_buffer("cat_mask_offset", cat_mask_offset)
        # self.register_buffer("con_mask_offset", con_mask_offset)

        # self.mask_embeds_cat = nn.Embedding(self.num_categories * 2, self.dim)
        # self.mask_embeds_cont = nn.Embedding(self.num_continuous * 2, self.dim)
        # self.single_mask = nn.Embedding(2, self.dim)
        # self.pos_encodings = nn.Embedding(self.num_categories + self.num_continuous, self.dim)

        self.mlp1 = sep_MLP(dim, self.num_categories - 1, categories[1:])
        self.mlp2 = sep_MLP(
            dim,
            self.num_continuous,
            np.ones(self.num_continuous).astype(int),
            mdn_heads=self.mdn_heads,
        )

    def encode(self, x_categ, x_cont):

        x = self.transformer(x_categ, x_cont)

        # Only pass CLS token forward
        x = x[:, 0, :]
        # x = torch.tanh(x)

        return x

    def decode(self, x):

        cat_outs = self.mlp1(x)
        con_outs = self.mlp2(x)

        return cat_outs, con_outs

    def forward(self, x_categ, x_cont):

        x = self.encode(x_categ, x_cont)
        cat_outs, con_outs = self.decode(x)

        return cat_outs, con_outs

    def inv_transform(self, x):

        cat_outs, con_outs = self.decode(x)

        cats, cons = [], []
        with torch.no_grad():
            for cat in cat_outs:
                cat = cat.detach()
                cats.append(torch.argmax(nn.functional.softmax(cat, dim=1), dim=1))
            for con in con_outs:
                if self.mdn:
                    pi, mu, sigma = con[0].detach(), con[1].detach(), con[2].detach()
                    mixture = torch.normal(mu, sigma)
                    k = torch.multinomial(pi, 1, replacement=True).squeeze()
                    cons.append(mixture.view(pi.shape)[range(k.size(0)), k])
                else:
                    con = con.detach()
                    # naive rounding to nearest .1
                    cons.append(torch.round(con * 10) / 10)
        return cats, cons


def embed_and_mask(data: Tuple, model: SAINT_AE, device="cpu", keep_cls=False):

    # Parse the batch tuple into components
    x_categ, x_cont, cat_mask, con_mask = (
        data[0].to(device),
        data[1].to(device),
        data[2].to(device),
        data[3].to(device),
    )
    """
    `x_categ` is the the categorical data (batch_size x cat_cols+1)
    `x_cont` has continuous data (batch_size x cont_cols)
    `cat_mask` is an array of ones same shape as x_categ and an additional column(corresponding to CLS token) set to 0s (batch_size, cat_cols+1)
    `con_mask` is an array of ones same shape as x_cont (batch_size, cont_cols)
    """

    # Prepare categorical and continunous embeddings
    _, x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model)

    if not keep_cls:
        # Trim CLS token off categorical targets
        x_categ = x_categ[:, 1:]

    return x_categ, x_cont, x_categ_enc_2, x_cont_enc_2


def train_on_batch(
    data: Tuple,
    model: SAINT_AE,
    criterion_cat=nn.CrossEntropyLoss(),
    criterion_cont=nn.MSELoss(),
    mdn=False,
    device="cpu",
):
    """Train SAINT Autoencoder on a batch, returning losses."""

    # Apply first embedding and masking
    x_categ, x_cont, x_categ_enc_2, x_cont_enc_2 = embed_and_mask(data, model=model, device=device)

    # Forward pass (embed and decode)
    cat_outs, con_outs = model(x_categ_enc_2, x_cont_enc_2)

    # Reconstruction loss (l_cat for Categorical l_cont for Continuous)
    l_cat, l_cont = 0, 0

    # Number of categories
    n_cat = x_categ.shape[-1]

    # Categorical reconstruction loss
    for j in range(n_cat):
        # TODO make this faster?
        x_probs = torch.full(cat_outs[j].shape, 0.01).to(device)
        for _i in range(cat_outs[j].shape[0]):
            x_probs[_i, x_categ[_i, j]] = 0.99
        l_cat += criterion_cat(cat_outs[j], x_probs)

    # Continuous reconstruction loss
    if len(con_outs) > 0:
        if mdn:
            # MDN loss
            for _i, con in enumerate(con_outs):
                pi, mu, sigma = con
                l_mdn = mdn_loss(x_cont[:, _i], pi, mu, sigma)
                l_cont += l_mdn
        else:
            con_outs = torch.cat(con_outs, dim=1)
            # Sqrt makes into RMSE loss
            l_cont = torch.sqrt(criterion_cont(con_outs, x_cont))

    return l_cat, l_cont


def parse_mixed_dataset(train_data, discrete_columns):

    data_dim = train_data.shape[-1]
    data_n = train_data.shape[0]
    dset_shape = train_data.shape
    colnames = train_data.columns
    disc_cols = list(discrete_columns)
    cat_idxs = [i for i, c in enumerate(colnames) if c in disc_cols]

    # The cardinality of each categorical feature
    cat_dims = [train_data.iloc[:, i].nunique() for i in cat_idxs]
    # Appending 1 for CLS token, this is later used to generate embeddings.
    cat_dims = np.append(np.array([1]), np.array(cat_dims)).astype(int)

    con_idxs = [i for i, _ in enumerate(colnames) if i not in cat_idxs]

    # Encode categoricals
    cat_label_encs = []
    for i in cat_idxs:
        l_enc = LabelEncoder()
        train_data.iloc[:, i] = l_enc.fit_transform(train_data.values[:, i])
        # store labels for inverse transform
        cat_label_encs.append(l_enc)

    # TODO I don't think this is needed
    # train_mean = train_data.values[:, con_idxs].mean()
    # train_std = train_data.values[:, con_idxs].std()
    # continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32)

    return (
        data_dim,
        data_n,
        dset_shape,
        colnames,
        disc_cols,
        cat_idxs,
        cat_dims,
        con_idxs,
        cat_label_encs,
    )


def pseudo_sample(model, dataloader, cat_idxs, con_idxs, n, device=None):

    model.eval()
    saved_latent = None
    with torch.no_grad():
        cat_samples, con_samples = [[] for _ in cat_idxs], [[] for _ in con_idxs]
        for i, data in enumerate(dataloader, 0):
            if i * dataloader.batch_size >= n:
                break

            # Forward pass the data through the SAINT encoder
            x_categ, x_cont, x_categ_enc_2, x_cont_enc_2 = embed_and_mask(
                data, model, device=device, keep_cls=False
            )

            # Forward pass the latents through the decoder MLPS
            x_hat = model.encode(x_categ_enc_2, x_cont_enc_2)
            if saved_latent is None:
                saved_latent = x_hat.detach().cpu()
            cat_tx, con_tx = model.inv_transform(x_hat)

            # Append to running output
            for j, d in enumerate(cat_tx):
                cat_samples[j].extend(list(d.cpu().flatten().numpy()))
            for j, d in enumerate(con_tx):
                con_samples[j].extend(list(d.cpu().flatten().numpy()))

    return cat_samples, con_samples, saved_latent


def samples_to_df(cat_samples, con_samples, cat_idxs, colnames, cat_label_encs, datatypes=None):

    n_feats = len(cat_samples) + len(con_samples)
    assert n_feats == len(colnames)

    # convert labels back to strings
    cat_samples = [cat_label_encs[i].inverse_transform(c) for i, c in enumerate(cat_samples)]

    cols = []
    for i in range(n_feats):
        if i in cat_idxs:
            col = cat_samples[0]
            cat_samples = cat_samples[1:]
        else:
            col = con_samples[0]
            con_samples = con_samples[1:]
        cols.append(col)

    df = pd.DataFrame(cols).T
    df.columns = colnames
    df = df.apply(pd.to_numeric, errors="ignore")

    if datatypes is not None:
        # Non-parametric post-processing
        for c, kind in datatypes:
            # If non-neg then clip values to zero
            if "positive" in kind:
                df[c] = df[c].apply(lambda x: max(0.0, x))
            # If integer, then round to units
            if "int" in kind:
                df[c] = df[c].round().astype("int64")
            # If float then round to 3dp
            if "float" in kind:
                df[c] = df[c].round(3).astype("float64")

    return df
