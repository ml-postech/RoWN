import numpy as np
from tqdm import tqdm
from sklearn import metrics

from .utils import create_adjacency, calculate_energy


def evaluation(
    args, 
    model, 
    dataset,
    dist_fn
):
    ranks = []
    ap_scores = []

    adjacency = create_adjacency(dataset.relations)

    iterator = tqdm(adjacency.items())
    batch_size = dataset.n_words // 10
    for i, (source, targets) in enumerate(iterator):
        if i % 1000 != 0:
            continue
        input_ = np.c_[
            source * np.ones(dataset.n_words).astype(np.int64),
            np.arange(dataset.n_words)
        ]
        _energies = calculate_energy(
            model,
            input_, 
            args.test_samples, 
            batch_size,
            dist_fn
        ).detach().cpu().numpy()
        
        _energies[source] = 1e+12
        _labels = np.zeros(dataset.n_words)
        _energies_masked = _energies.copy()
        _ranks = []
        for o in targets:
            _energies_masked[o] = np.Inf
            _labels[o] = 1
        ap_scores.append(metrics.average_precision_score(_labels, -_energies))
        for o in targets:
            ene = _energies_masked.copy()
            ene[o] = _energies[o]
            r = np.argsort(ene)
            _ranks.append(np.where(r == o)[0][0] + 1)
        ranks += _ranks

    return np.mean(ranks), np.mean(ap_scores)

