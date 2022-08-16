import torch
from torch.distributions import Normal

from ..hwn import HWN


class Distribution(HWN):
    def __init__(self, mean, covar) -> None:
        base = Normal(
            torch.zeros(
                covar.size(), 
                device=covar.device
            ),
            covar
        )

        super().__init__(mean, base)

