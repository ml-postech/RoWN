import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, 
        prior, 
        dist, 
        encoder, 
        encoder_layer, 
        decoder, 
        decoder_layer, 
        loss_type
    ):
        super().__init__()

        self.prior = prior
        self.dist = dist
        self.encoder = encoder
        self.encoder_layer = encoder_layer
        self.decoder = decoder
        self.decoder_layer = decoder_layer
        self.loss_type = loss_type

    def forward(self, x, n_samples=1, beta=1.):
        mean, covar = self.encoder_layer(self.encoder(x))
        variational = self.dist(mean, covar)
        
        z = variational.rsample(n_samples)
        log_prob_base = variational.log_prob(z)
        log_prob_target = self.prior.log_prob(z)
        kl_loss = (log_prob_base - log_prob_target).mean(dim=0)

        x_generated = self.generate(z)
        if self.loss_type == 'BCE':
            recon_loss = F.binary_cross_entropy(
                x_generated, 
                x.unsqueeze(0).expand(x_generated.size()), 
                reduction='none'
            )
        else:
            recon_loss = F.gaussian_nll_loss(
                x_generated, 
                x.unsqueeze(0).expand(x_generated.size()), 
                torch.ones(x_generated.size(), device=x.device) * 0.01, 
                reduction='none'
            )

        while len(recon_loss.size()) > 2:
            recon_loss = recon_loss.sum(-1)
        recon_loss = recon_loss.mean(dim=0)

        total_loss_sum = recon_loss + beta * kl_loss
        loss = total_loss_sum.mean()

        recon_loss = recon_loss.sum()
        kl_loss = kl_loss.sum()
        elbo = -(recon_loss + kl_loss)

        return loss, elbo, z, mean, recon_loss, kl_loss

    def generate(self, z):
        return self.decoder(self.decoder_layer(z))

