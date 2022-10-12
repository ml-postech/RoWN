# A Rotated Hyperbolic Wrapped Normal Distribution for Hierarchical Representation Learning
This repository is the official implementation of ["A Rotated Hyperbolic Wrapped Normal Distribution for Hierarchical Representation Learning"](https://arxiv.org/abs/2205.13371) accepted by NeurIPS 2022.

## Abstract
We present a rotated hyperbolic wrapped normal distribution (RoWN), a simple yet effective alteration of a hyperbolic wrapped normal distribution (HWN). The HWN expands the domain of probabilistic modeling from Euclidean to hyperbolic space, where a tree can be embedded with arbitrary low distortion in theory. In this work, we analyze the geometric properties of the \emph{diagonal} HWN, a standard choice of distribution in probabilistic modeling. The analysis shows that the distribution is inappropriate to represent the data points at the same hierarchy level through their angular distance with the same norm in the Poincar\'e disk model. We then empirically verify the presence of limitations of HWN, and show how RoWN, the proposed distribution, can alleviate the limitations on various hierarchical datasets, including noisy synthetic binary tree, WordNet, and Atari 2600 Breakout.

## Usages
You can reproduce the experiments from our paper using the following command:
```
> python train_vae.py --task NSBT --dist RoWN --depth=7 --device=cuda:0 --eval_interval=1001 --exp_name=nsbt --lr=0.0001 --n_epochs=1000 --n_layers=1 --test_samples=500 --train_batch_size=128 --seed 1
> python train_vae.py --task Breakout --dist RoWN --data_dir=<data_dir> --device=cuda:0 --eval_interval=201 --exp_name=breakout --latent_dim=20 --lr=0.0001 --n_epochs=200 --test_batch_size=64 --test_samples=100 --train_batch_size=100 --train_samples=1 --seed 1
> python train_embedding.py --dist=RoWN --data_dir data/ --device=cuda:0 --latent_dim=20 --seed=1
```

### Dataset
For noisy synthetic binary tree, we can generate the dataset using the following command:
```
> cd tasks/NSBT; python utils.py
```

For Atari 2600 Breakout, the images can be download from [here](https://www.dropbox.com/s/hyq44euztzz23o8/breakout_states_v2.h5?dl=0).

For WordNet, we can download the dataset using the following command:
```
> mkdir data; cd tasks/WordNet; python utils.py
```

### Distributions
For the distributions, we implemented:
- `EuclideanNormal`: Gaussian distribution defined in Euclidean space.
- `IsotropicHWN`: Hyperbolic wrapped normal distribution with isotropic covariance.
- `DiagonalHWN`: Hyperbolic wrapped normal distribution with diagonal covariance.
- `FullHWN`: Hyperbolic wrapped normal distribution with full covariance.
- `RoWN`: Hyperbolic wrapped normal distribution with rotated covariance.

## Cite
Please cite our paper if you use the model or this code in your own work:
```
@article{cho2022rotated,
  title={A Rotated Hyperbolic Wrapped Normal Distribution for Hierarchical Representation Learning},
  author={Cho, Seunghyuk and Lee, Juyong and Park, Jaesik and Kim, Dongwoo},
  journal={arXiv preprint arXiv:2205.13371},
  year={2022}
}
```
