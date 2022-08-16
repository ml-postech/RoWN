import torch
import geoopt
from torch import nn
from torch.nn import functional as F

from ..utils import LogLayer


class EncoderLayer(nn.Module):
    def __init__(self, args, feature_dim) -> None:
        super().__init__()

        self.latent_dim = args.latent_dim
        self.feature_dim = feature_dim

        self.manifold = geoopt.manifolds.Lorentz()
        self.variational = nn.Linear(
            self.feature_dim,
            self.latent_dim + self.latent_dim ** 2
        )

    def forward(self, feature):
        feature = self.variational(feature)
        mu, covar = torch.split(
            feature,
            [self.latent_dim, self.latent_dim ** 2],
            dim=-1
        )

        mu = F.pad(mu, (1, 0))
        mu = self.manifold.expmap0(mu)

        covar_size = covar.size()[:-1]
        covar = covar.view(
            *covar_size, 
            self.latent_dim, 
            self.latent_dim
        )
        covar = covar.matmul(covar.transpose(-1, -2))
        covar = covar + 1e-9 * torch.eye(
            self.latent_dim,
            device=covar.device
        )[None, ...]

        return mu, covar


DecoderLayer = LogLayer


class EmbeddingLayer(nn.Module):
    def __init__(self, args, n_words):
        super().__init__()

        self.args = args
        self.latent_dim = args.latent_dim
        self.n_words = n_words
        self.initial_sigma = args.initial_sigma
        self.manifold = geoopt.manifolds.Lorentz()

        mean_initialize = torch.empty([self.n_words, self.latent_dim])
        nn.init.normal_(mean_initialize, std=args.initial_sigma)
        self.mean = nn.Embedding.from_pretrained(mean_initialize, freeze=False)

        covar_initialize = torch.stack(
             [torch.eye(self.latent_dim) for _ in range(self.n_words)]
        ).view(self.n_words, -1)
        covar_initialize = covar_initialize * torch.randn(covar_initialize.size()) * self.initial_sigma
        self.covar = nn.Embedding.from_pretrained(covar_initialize, freeze=False)

    def forward(self, x):
        mean = self.mean(x)
        mean = F.pad(mean, (1, 0))
        mean = self.manifold.expmap0(mean)

        covar = self.covar(x)
        covar_size = covar.size()[:-1]
        covar = covar.view(
            *covar_size, 
            self.latent_dim, 
            self.latent_dim
        )
        covar = covar.matmul(covar.transpose(-1, -2))
        covar = covar + 1e-9 * torch.eye(
            self.latent_dim,
            device=covar.device
        )[None, ...]

        return mean, covar

