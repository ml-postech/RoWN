import torch
import geoopt
from torch.nn import functional as F

class HWN():
    def __init__(self, mean, base) -> None:
        self.mean = mean
        self.base = base
        self.manifold = geoopt.manifolds.Lorentz()

        self.origin = self.manifold.origin(
            self.mean.size(),
            device=self.mean.device
        )
        self.latent_dim = self.mean.size(-1) - 1

    def log_prob(self, z):
        u = self.manifold.logmap(self.mean, z)
        v = self.manifold.transp(self.mean, self.origin, u)
        log_prob_v = self.base.log_prob(v[:, :, 1:]).sum(-1)

        r = self.manifold.norm(u)
        log_det = (self.latent_dim - 1) * (torch.sinh(r).log() - r.log())

        log_prob_z = log_prob_v - log_det
        return log_prob_z

    def rsample(self, N):
        v = self.base.rsample([N])
        v = F.pad(v, (1, 0))

        u = self.manifold.transp0(self.mean, v)
        z = self.manifold.expmap(self.mean, u)

        return z

    def sample(self, N):
        with torch.no_grad():
            return self.rsample(N)

