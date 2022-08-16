from torch import nn
from .utils import stack_linear_layers


def add_model_args(parser):
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_hids', type=int, default=256)


class Encoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.args = args
        self.depth = args.depth
        self.n_layers = args.n_layers
        self.n_hids = args.n_hids

        self.encoder = nn.Sequential(
            nn.Linear(2 ** self.depth - 1, self.n_hids),
            nn.ReLU(),
            *stack_linear_layers(self.n_hids, self.n_layers),
            # nn.Linear(self.n_hids, 2 * self.latent_dim)
        )

        self.output_dim = self.n_hids

    def forward(self, x):
        feature = self.encoder(x)
        return feature


class Decoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.args = args
        self.depth = args.depth
        self.n_layers = args.n_layers
        self.n_hids = args.n_hids
        self.latent_dim = args.latent_dim

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.n_hids),
            nn.ReLU(),
            *stack_linear_layers(self.n_hids, self.n_layers),
            nn.Linear(self.n_hids, 2 ** self.depth - 1),
            nn.Tanh()
        )

    def forward(self, z):
        fixed_shapes = z.size()[:-1]
        z = z.view(-1, self.latent_dim)
        x = self.decoder(z)
        x = x.view(*fixed_shapes, -1)

        return x

