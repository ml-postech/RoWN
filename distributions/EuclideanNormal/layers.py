import torch
from torch import nn
from torch.nn import functional as F


class EncoderLayer(nn.Module):
    def __init__(self, args, feature_dim) -> None:
        super().__init__()

        self.latent_dim = args.latent_dim
        self.feature_dim = feature_dim

        self.variational = nn.Linear(
            self.feature_dim,
            2 * self.latent_dim
        )

    def forward(self, feature):
        feature = self.variational(feature)
        mean, covar = torch.split(
            feature,
            [self.latent_dim, self.latent_dim],
            dim=-1
        )
        covar = F.softplus(covar)

        return mean, covar


class DecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, z):
        return z


class EmbeddingLayer(nn.Module):
    def __init__(self, args, n_words):
        super().__init__()

        self.args = args
        self.latent_dim = args.latent_dim
        self.n_words = n_words
        self.initial_sigma = args.initial_sigma

        mean_initialize = torch.empty([self.n_words, self.latent_dim])
        nn.init.normal_(mean_initialize, std=args.initial_sigma)
        self.mean = nn.Embedding.from_pretrained(mean_initialize, freeze=False)
        
        covar_initialize = torch.empty([self.n_words, self.latent_dim])
        nn.init.normal_(covar_initialize, std=args.initial_sigma)
        self.covar = nn.Embedding.from_pretrained(covar_initialize, freeze=False)

    def forward(self, x):
        mean = self.mean(x)
        covar = F.softplus(self.covar(x))

        return mean, covar

