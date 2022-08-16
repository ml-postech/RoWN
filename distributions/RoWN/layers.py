import torch
import geoopt
from torch import nn
from torch.nn import functional as F

from ..utils import ExpLayer, LogLayer

EncoderLayer = ExpLayer
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
        
        covar_initialize = torch.empty([self.n_words, self.latent_dim])
        nn.init.normal_(covar_initialize, std=args.initial_sigma)
        self.covar = nn.Embedding.from_pretrained(covar_initialize, freeze=False)

    def forward(self, x):
        mean = self.mean(x)
        mean = F.pad(mean, (1, 0))
        mean = self.manifold.expmap0(mean)

        covar = F.softplus(self.covar(x))

        return mean, covar

