import numpy as np
from pathlib import Path
from torch.utils import data

from .utils import synthetic_binary_tree, noisy_sythetic_binary_tree


def add_dataset_args(parser):
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--depth', type=int, default=4)


class Dataset(data.Dataset):
    def __init__(self, args, is_train=True) -> None:
        super().__init__()

        self.args = args
        self.is_train = is_train
        self.features = None
        self.depth = args.depth

        if self.is_train:
            if args.data_dir is None:
                self.data, _ = noisy_sythetic_binary_tree(args.depth)
            else:
                data_dir = Path(args.data_dir)
                self.data = np.load(data_dir / f'depth_{self.depth}.npy')
        else:
            self.data = synthetic_binary_tree(self.depth)

        self.data = (self.data - 0.5) * 2

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        return x

