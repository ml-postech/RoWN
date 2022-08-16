# Code from https://github.com/pfnet-research/hyperbolic_wrapped_distribution/blob/master/lib/models/embedding.py


import copy
import wandb
import torch
import argparse
import importlib
import numpy as np
from math import ceil
from torch.optim import Adagrad
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tasks.WordNet import Dataset, evaluation


class LRScheduler():
    def __init__(self, optimizer, lr, c, n_burnin_steps):
        self.optimizer = optimizer
        self.lr = lr
        self.n_burnin_steps = n_burnin_steps
        self.c = c
        self.n_steps = 0

    def step_and_update_lr(self):
        self._update_learning_rate()
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _update_learning_rate(self):
        self.n_steps += 1
        if self.n_steps <= self.n_burnin_steps:
            lr = self.lr / self.c
        else:
            lr = self.lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--data_type', type=str, default='noun')
    parser.add_argument('--n_negatives', type=int, default=1)
    parser.add_argument('--latent_dim', type=int)
    parser.add_argument('--batch_size', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=0.6)
    parser.add_argument('--n_epochs', type=int, default=10000)
    parser.add_argument('--dist', type=str, choices=['EuclideanNormal', 'IsotropicHWN', 'DiagonalHWN', 'RoWN', 'FullHWN'])
    parser.add_argument('--initial_sigma', type=float, default=0.01)
    parser.add_argument('--bound', type=float, default=37)
    parser.add_argument('--train_samples', type=int, default=1)
    parser.add_argument('--test_samples', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--c', type=float, default=40)
    parser.add_argument('--burnin_epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.set_default_tensor_type(torch.DoubleTensor)

    dataset = Dataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    dist_module = importlib.import_module(f'distributions.{args.dist}')
    model = getattr(dist_module, 'EmbeddingLayer')(args, dataset.n_words).to(args.device)
    dist_fn = getattr(dist_module, 'Distribution')

    optimizer = Adagrad(model.parameters(), lr=args.lr)
    n_batches = int(ceil(len(dataset) / args.batch_size))
    n_burnin_steps = args.burnin_epochs * n_batches
    lr_scheduler = LRScheduler(optimizer, args.lr, args.c, n_burnin_steps)

    best_model = copy.deepcopy(model)
    best_score = None 

    wandb.init(project='RoWN')
    wandb.run.name = 'wordnet'
    wandb.config.update(args)
    for epoch in range(1, args.n_epochs + 1):
        total_loss, total_kl_target, total_kl_negative = 0., 0., 0.
        total_diff = 0.
        n_batches = 0
        model.train()

        for x in loader:
            for param in model.parameters():
                param.grad = None
            x = x.cuda()
            mean, covar = model(x)
            dist_anchor = dist_fn(mean[:, 0, :], covar[:, 0, :])
            dist_target = dist_fn(mean[:, 1, :], covar[:, 1, :])
            dist_negative = dist_fn(mean[:, 2, :], covar[:, 2, :])

            z = dist_anchor.rsample(args.train_samples)
            log_prob_anchor = dist_anchor.log_prob(z)
            log_prob_target = dist_target.log_prob(z)
            log_prob_negative = dist_negative.log_prob(z)
            kl_target = (log_prob_anchor - log_prob_target).mean(dim=0)
            kl_negative = (log_prob_anchor - log_prob_negative).mean(dim=0)
            
            loss = F.relu(args.bound + kl_target - kl_negative).mean()
            loss.backward()
            lr_scheduler.step_and_update_lr()
           
            total_loss += loss.item() * kl_target.size(0)
            total_kl_target += kl_target.sum().item()
            total_kl_negative += kl_negative.sum().item()
            total_diff += (kl_target - kl_negative).sum().item()
            n_batches += kl_target.size(0)

        if best_score is None or best_score > total_loss:
            best_score = total_loss
            best_model = copy.deepcopy(model)

        print(f"Epoch {epoch:8d} | Total loss: {total_loss / n_batches:.3f} | KL Target: {total_kl_target / n_batches:.3f} | KL Negative: {total_kl_negative / n_batches:.3f}")
        wandb.log({
            'epoch': epoch,
            'train_loss': total_loss / n_batches,
            'train_kl_target': total_kl_target / n_batches,
            'train_kl_negative': total_kl_negative / n_batches
        })
        
        if epoch % args.eval_interval == 0 or epoch == args.n_epochs:
            best_model.eval()
            rank, ap = evaluation(args, best_model, dataset, dist_fn)
            print(f"===========> Mean rank: {rank} | MAP: {ap}")
            wandb.log({
                'epoch': epoch,
                'rank': rank,
                'map': ap
            })

