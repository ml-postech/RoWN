import torch

from .distribution import Distribution


def get_prior(args):
    mean = torch.zeros(
        [1, args.latent_dim], 
        device=args.device
    )
    covar = torch.ones(
        [1, args.latent_dim], 
        device=args.device
    )

    prior = Distribution(mean, covar)
    return prior

