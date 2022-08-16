import torch
import geoopt
from torch import nn
from torch.nn import functional as F


class ExpLayer(nn.Module):
    def __init__(self, args, feature_dim) -> None:
        super().__init__()

        self.latent_dim = args.latent_dim
        self.feature_dim = feature_dim

        self.manifold = geoopt.manifolds.Lorentz()
        self.variational = nn.Linear(
            self.feature_dim,
            2 * self.latent_dim
        )

    def forward(self, feature):
        feature = self.variational(feature)
        mu, covar = torch.split(
            feature,
            [self.latent_dim, self.latent_dim],
            dim=-1
        )

        mu = F.pad(mu, (1, 0))
        mu = self.manifold.expmap0(mu)
        covar = F.softplus(covar)

        return mu, covar


class LogLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.manifold = geoopt.manifolds.Lorentz()

    def forward(self, z):
        z = self.manifold.logmap0(z)
        return z[..., 1:]

