import wandb
import numpy as np


def evaluation(
    args, 
    vae, 
    dataset, 
    means
):
    means = means.detach().cpu().numpy()
    if args.dist != 'EuclideanNormal':
        norm = means[..., 0]
        norm = np.sqrt((norm - 1) / (norm + 1))
    else:
        norm = np.sqrt((means ** 2).sum(axis=-1))
   
    features = dataset.features
    metric = np.corrcoef(norm, features)[0, 1]
    print(f'===========> Correlation with cumulative rewards: {metric}')
    wandb.log({
        'test_corr_reward': metric
    })

