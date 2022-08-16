import numpy as np
from pathlib import Path
from torch.utils import data

from .utils import slurp


class Dataset(data.Dataset):
    def __init__(self, args):
        self.args = args
        file_name = Path(args.data_dir).expanduser() / f'{args.data_type}_closure.tsv'
        indices, objects = slurp(file_name.as_posix(), symmetrize=False)
        
        self.relations = indices[:, :2]
        self.words = objects
        self.n_negatives = args.n_negatives
        self.n_words = len(self.words)

    def __len__(self):
        return len(self.relations)

    def __getitem__(self, i):
        return np.r_[
            self.relations[i],
            np.random.randint(self.n_words, size=self.n_negatives)
        ]

