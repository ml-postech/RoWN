import numpy as np
from math import pi
from torch import nn
from tqdm import tqdm


def stack_linear_layers(n_hids, n_layers):
    return [nn.Sequential(
                nn.Linear(n_hids, n_hids),
                nn.ReLU()
            ) for _ in range(n_layers)]


def hasone(node_index, dim_index):
    bin_i, bin_j = np.binary_repr(node_index), np.binary_repr(dim_index)
    length = len(bin_j)
    return (bin_i[:length] == bin_j) * 1


def synthetic_binary_tree(depth):
    n = 2 ** depth - 1
    x = np.fromfunction(
        lambda i, j: np.vectorize(hasone)(i + 1, j + 1), 
        (n, n), 
        dtype=np.int64
    ).astype(np.float64)

    return x


def noisy_sythetic_binary_tree(depth, n_samples=100):
    original_data = synthetic_binary_tree(depth)
    data = np.empty((0, original_data.shape[-1]))
    features = []
    for idx in tqdm(range(original_data.shape[0])):
        x = original_data[idx]
        idxs = (x == 1).nonzero()[0][1:]
        for _ in range(n_samples):
            x_ = x.copy()
            if len(idxs) > 0:
                theta = np.random.random(len(idxs)) * 0.5 * pi / 2
                eps_x = np.cos(theta)
                eps_y = np.sin(theta)

                x_[idxs] = eps_x
                idxs_ = idxs + (idxs % 2 - 0.5) * 2
                idxs_ = idxs_.astype(np.int64)
                x_[idxs_] = eps_y
                features.append(theta[-1])
            else:
                features.append(0.)
            data = np.concatenate(
                (data, x_[None, ...]),
                axis=0
            )

    features = np.array(features)
    return data, features


if __name__ == "__main__":
    for depth in range(4, 9):
        data, feature = noisy_sythetic_binary_tree(depth)
        np.save(f'/data_seoul/shhj1998/dataset/NSBT/depth_{depth}.npy', data)
        np.save(f'/data_seoul/shhj1998/dataset/NSBT/feature_{depth}.npy', feature)
