import torch
import geoopt
from .distribution import Distribution


def get_prior(args):
    m = geoopt.manifolds.Lorentz()

    mean = m.origin([1, args.latent_dim + 1], device=args.device)
    covar = torch.ones(
        1, 
        args.latent_dim, 
        device=args.device
    )

    prior = Distribution(mean, covar)
    return prior

