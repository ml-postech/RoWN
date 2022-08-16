from torch import nn


def add_model_args(parser):
    pass


class Encoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.args = args
        self.output_dim = 10 * 10 * 64
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(),  # (80, 80)
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU(),  # (40, 40)
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),  # (40, 40)
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),  # (20, 20)
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),  # (20, 20)
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(),  # (10, 10)
            nn.Flatten(),
        )

    def forward(self, x):
        feature = self.encoder(x)
        return feature


class Decoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.args = args
        self.latent_dim = args.latent_dim

        self.decoder1 = nn.Sequential(
            nn.Linear(self.latent_dim, 10 * 10 * 64),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # (16, 40, 40)
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),   # (1, 80, 80)
            nn.Sigmoid()
        )

    def forward(self, z):
        fixed_shapes = z.size()[:-1]
        z = z.view(-1, self.latent_dim)
        z = self.decoder1(z)
        z = z.view(-1, 64, 10, 10)
        x = self.decoder2(z)
        x = x.view(*fixed_shapes, 1, 80, 80)
        return x

