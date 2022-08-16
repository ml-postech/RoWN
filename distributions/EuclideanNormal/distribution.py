import torch
from torch.distributions import Normal

def kl_dist(mu0, std0, mu1, std1):
    k = mu0.size(-1)
    logvar0, logvar1 = 2 * std0.log(), 2 * std1.log()

    dist = logvar1 - logvar0 + (((mu0 - mu1).pow(2) + 1e-9).log() - logvar1).exp() + (logvar0 - logvar1).exp()
    dist = dist.sum(dim=-1) - k
    return dist * 0.5

class Distribution():
    def __init__(self, mean, covar) -> None:
        self.mean = mean
        self.covar = covar

        self.base = Normal(self.mean, self.covar)

    def log_prob(self, z):
        return self.base.log_prob(z).sum(dim=-1)

    def rsample(self, N):
        return self.base.rsample([N])

    def sample(self, N):
        with torch.no_grad():
            return self.rsample(N)

