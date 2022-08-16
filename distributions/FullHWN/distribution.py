import torch
from torch.distributions import MultivariateNormal

from ..hwn import HWN


class Distribution(HWN):
    def __init__(self, mean, covar) -> None:
        base = MultivariateNormal(
            torch.zeros(
                mean.size(), 
                device=covar.device
            )[..., 1:],
            covar
        )

        super().__init__(mean, base)

    def log_prob(self, z):
        u = self.manifold.logmap(self.mean, z)
        v = self.manifold.transp(self.mean, self.origin, u)
        log_prob_v = self.base.log_prob(v[:, :, 1:])

        r = self.manifold.norm(u)
        log_det = (self.latent_dim - 1) * (torch.sinh(r).log() - r.log())

        log_prob_z = log_prob_v - log_det
        return log_prob_z

