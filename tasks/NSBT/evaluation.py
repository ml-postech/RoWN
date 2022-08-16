import wandb
import geoopt
import numpy as np


def evaluation(
    args, 
    vae, 
    dataset, 
    means
):    
    x = dataset.data
    means = means.detach()
    x_recon = vae.generate(means).detach().cpu().numpy()
    
    N = x.shape[0]
    d_true = np.zeros((N, N))
    d_pred = np.zeros((N, N))
    m = geoopt.manifolds.Euclidean(1) if args.dist == 'EuclideanNormal' else geoopt.manifolds.Lorentz()

    test_error = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j] * x_recon[i, j] <= 0:
                test_error += 1
    
    n_data = 2 ** args.depth - 1
    test_error /= n_data * n_data

    x = x / 2 + 0.5
    for i in range(N):
        for j in range(i):
            d_true[i, j] = (x[i].astype(np.int32) ^ x[j].astype(np.int32)).sum()
            d_pred[i, j] = m.dist(means[i], means[j])

    mask = np.fromfunction(lambda i, j: i > j, shape=d_true.shape)
    corr_distance = np.corrcoef(d_pred[mask], d_true[mask])[0, 1]

    depths = x.sum(axis=-1)
    if args.dist != 'EuclideanNormal':
        norm = means[..., 0].cpu().numpy()
        norm = np.sqrt((norm - 1) / (norm + 1))
    else:
        norm = means.cpu().numpy()
        norm = np.sqrt((norm ** 2).sum(axis=-1))

    corr_depth = np.corrcoef(norm, depths)[0, 1]

    print(f'===========> Test error: {test_error}')
    print(f'===========> Correlation with hamming distance: {corr_distance} | with depth: {corr_depth}')
    wandb.log({
        'test_error': test_error,
        'test_corr_distance': corr_distance,
        'test_corr_depth': corr_depth
    })

