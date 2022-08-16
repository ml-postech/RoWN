import copy
import wandb
import datetime
import importlib
from pathlib import Path

import torch
import argparse
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader

from vae import VAE
from arguments import add_train_args, get_initial_parser


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.xavier_normal_(m.weight)


def train(epoch, args, train_loader, vae, optimizer):
    n_data = 0
    train_elbo, train_recon, train_kl = 0., 0., 0.

    for x in train_loader:
        for param in vae.parameters():
            param.grad = None
        x = x.to(args.device)
        loss, elbo, _, _, recon_loss, kl_loss = vae(
            x, 
            args.train_samples, 
            args.beta
        )

        loss.backward()
        optimizer.step()
       
        n_data += x.size(0)
        train_elbo += elbo.item()
        train_recon += recon_loss.item()
        train_kl += kl_loss.item()

    if epoch % args.log_interval == 0 or epoch == args.n_epochs:
        train_elbo /= n_data
        train_recon /= n_data
        train_kl /= n_data
        print(f'Epoch: {epoch:6d} | ELBO: {train_elbo:.2f} | Recon Loss: {train_recon:.2f} | KL: {train_kl:.3f}')
        wandb.log({
            'epoch': epoch,
            'train_elbo': train_elbo,
            'train_recon': train_recon,
            'train_kl': train_kl
        })

    return train_elbo


def eval(epoch, args, test_loader, vae, root_dir, test_data, eval_fn):
    log_dir = root_dir / str(epoch)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        vae.eval()

        n_data = 0
        means = torch.empty((
            0, 
            (args.latent_dim + 1 if args.dist != 'EuclideanNormal' else args.latent_dim)
        ), device=args.device)
        total_elbo, total_recon, total_kl = 0., 0., 0.
        for x in test_loader:
            x = x.to(args.device)
            _, elbo, _, means_, recon_loss, kl_loss = vae(x, args.test_samples)

            n_data += x.size(0)
            total_elbo += elbo.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            means = torch.concat((means, means_))

        total_elbo /= n_data
        total_recon /= n_data
        total_kl /= n_data
        print(f'===========> Test ELBO: {total_elbo:.2f} | Test Recon: {total_recon:.2f} | Test KL: {total_kl:.2f}')
        wandb.log({
            'test_elbo': total_elbo,
            'test_recon': total_recon,
            'test_kl': total_kl
        })

        eval_fn(args, vae, test_data, means)
        torch.save(vae.state_dict(), root_dir / 'model.pt')


if __name__ == "__main__":
    init_parser = get_initial_parser()
    task_name = init_parser.parse_known_args()[0].task
    task_module = importlib.import_module(f'tasks.{task_name}')
    dist_name = init_parser.parse_known_args()[0].dist
    dist_module = importlib.import_module(f'distributions.{dist_name}')

    parser = argparse.ArgumentParser()
    add_train_args(parser)
    getattr(task_module, 'add_task_args')(parser)
    args = parser.parse_args()

    if args.task == 'NSBT':
        args.latent_dim = args.depth
        args.n_hids = 8 * (2 ** args.depth)
        # args.train_batch_size = 2 ** args.depth - 1

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.set_default_tensor_type(torch.DoubleTensor)

    runId = datetime.datetime.now().isoformat().replace(':', '_')
    root_dir = Path(args.log_dir) / runId

    train_data = getattr(task_module, 'Dataset')(args, is_train=True)
    train_loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)
    test_data = getattr(task_module, 'Dataset')(args, is_train=False)
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)
    eval_fn = getattr(task_module, 'evaluation')

    variational_fn = getattr(dist_module, 'Distribution')
    prior = getattr(dist_module, 'get_prior')(args)

    encoder = getattr(task_module, 'Encoder')(args)
    encoder_layer = getattr(dist_module, 'EncoderLayer')(args, encoder.output_dim)
    decoder = getattr(task_module, 'Decoder')(args)
    decoder_layer = getattr(dist_module, 'DecoderLayer')()

    recon_loss_type = getattr(task_module, 'recon_loss_type')
    vae = VAE(
        prior, 
        variational_fn, 
        encoder, 
        encoder_layer, 
        decoder, 
        decoder_layer, 
        recon_loss_type
    )
    # vae.apply(init_weights)
    vae = vae.to(args.device)

    optimizer = Adam(
        list(encoder.parameters()) + list(decoder.parameters()), 
        lr=args.lr
    )

    wandb.init(project='RoWN')
    wandb.run.name = args.exp_name
    wandb.config.update(args)
    print(root_dir)

    best_model = copy.deepcopy(vae)
    best_elbo = -1e9

    for epoch in range(1, args.n_epochs + 1):
        vae.train()
        train_elbo = train(epoch, args, train_loader, vae, optimizer)
        if best_elbo < train_elbo:
            best_elbo = train_elbo
            best_model = copy.deepcopy(vae)

        if epoch % args.eval_interval == 0 or epoch == args.n_epochs:
            best_model.eval()
            eval(epoch, args, test_loader, best_model, root_dir, test_data, eval_fn)

