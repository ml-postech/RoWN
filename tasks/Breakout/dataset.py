import numpy as np
from pathlib import Path
from torch.utils import data
from torchvision.transforms import ToTensor


def add_dataset_args(parser):
    parser.add_argument('--data_dir', type=str)


class Dataset(data.Dataset):
    def __init__(self, args, is_train=True) -> None:
        super().__init__()

        self.args = args
        self.is_train = is_train
        self.transform = ToTensor()
        self.data_dir = Path(args.data_dir)
    
        prefix = 'train_' if self.is_train else 'test_'
        raw_data = np.load(
            self.data_dir / f'{prefix}data.npy'
        )
        self.data = np.transpose(
            raw_data, 
            (0, 2, 3, 1)
        ).astype(np.float64)

        self.features = np.load(
            self.data_dir / f'{prefix}labels.npy'
        )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        x = self.transform(x)
        return x

