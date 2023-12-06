"""
Shared network classes for Generator, Discriminator, etc.
"""

import torch
from torch import nn


class Discriminator(nn.Module):
    """Based on the CTGAN implementation at
    https://github.com/sdv-dev/CTGAN/blob/master/ctgan/synthesizers/ctgan.py
    """

    def __init__(self, input_dim, dis_dims=(256, 256), pack=1):
        super().__init__()
        dim = input_dim * pack
        self.pack = pack
        self.packdim = dim
        seq = []
        for item in list(dis_dims):
            seq += [nn.Linear(dim, item), nn.LeakyReLU(0.2), nn.Dropout(0.5)]
            dim = item

        seq += [nn.Linear(dim, 1)]
        seq += [nn.Sigmoid()]
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        # assert input.size()[0] % self.pack == 0
        return self.seq(x.view(-1, self.packdim))


class Residual(nn.Module):
    """Residual layer"""

    def __init__(self, i, o):
        super().__init__()
        self.fc = nn.Linear(i, o)
        # self.bn = nn.BatchNorm1d(o)
        self.bn = nn.GroupNorm(1, o)  # Use privacy-safe groupnorm over batchnorm
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, x], dim=1)


class Generator(nn.Module):
    """Based on the CTGAN implementation at
    https://github.com/sdv-dev/CTGAN/blob/master/ctgan/synthesizers/ctgan.py
    """

    def __init__(self, embedding_dim, data_dim, gen_dims=(256, 256)):
        super().__init__()
        dim = embedding_dim
        seq = []
        for item in list(gen_dims):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(nn.Linear(dim, data_dim))
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)


class Encoder(nn.Module):
    """Encoder for the TVAESynthesizer.

    Args:
        data_dim (int):
            Dimensions of the data.
        compress_dims (tuple or list of ints):
            Size of each hidden layer.
        embedding_dim (int):
            Size of the output vector.
    """

    def __init__(self, data_dim, compress_dims, embedding_dim):
        super().__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [nn.Linear(dim, item), nn.ReLU()]
            dim = item

        self.seq = nn.Sequential(*seq)
        self.fc1 = nn.Linear(dim, embedding_dim)
        self.fc2 = nn.Linear(dim, embedding_dim)

    def forward(self, input_):
        """Encode the passed `input_`."""
        feature = self.seq(input_)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(nn.Module):
    """Decoder for the TVAESynthesizer.

    Args:
        embedding_dim (int):
            Size of the input vector.
        decompress_dims (tuple or list of ints):
            Size of each hidden layer.
        data_dim (int):
            Dimensions of the data.
    """

    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [nn.Linear(dim, item), nn.ReLU()]
            dim = item

        seq.append(nn.Linear(dim, data_dim))
        self.seq = nn.Sequential(*seq)
        self.sigma = nn.Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input_):
        """Decode the passed `input_`."""
        return self.seq(input_), self.sigma
