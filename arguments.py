import argparse


tasks = ['Breakout', 'NSBT']
distributions = ['EuclideanNormal', 'IsotropicHWN', 'DiagonalHWN', 'FullHWN', 'RoWN']


def get_initial_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--task', type=str, choices=tasks)
    parser.add_argument('--dist', type=str, choices=distributions)
    return parser


def add_train_args(parser):
    group = parser.add_argument_group('train')
    group.add_argument('--task', type=str, choices=tasks)
    group.add_argument('--dist', type=str, choices=distributions)
    group.add_argument('--seed', type=int, default=7777)
    group.add_argument('--latent_dim', type=int, default=2)
    group.add_argument('--beta', type=float, default=1.)
    group.add_argument('--n_epochs', type=int, default=10)
    group.add_argument('--train_batch_size', type=int, default=32)
    group.add_argument('--test_batch_size', type=int, default=32)
    group.add_argument('--lr', type=float, default=1e-5)
    group.add_argument('--device', type=str, default='cuda:0')
    group.add_argument('--eval_interval', type=int, default=10)
    group.add_argument('--log_interval', type=int, default=10)
    group.add_argument('--log_dir', type=str, default='logs/')
    group.add_argument('--train_samples', type=int, default=1)
    group.add_argument('--test_samples', type=int, default=500)
    group.add_argument('--exp_name', type=str, default='dummy')

