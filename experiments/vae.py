"""Variational Autoencoder with a subset latent space"""

import math
import pathlib
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from lightning import seed_everything
from lightning.fabric.loggers import CSVLogger
from torch.distributions import Categorical
from torch.optim import Adam
from tqdm import tqdm

from datasets import get_dataloaders, infinite_iter
from models import get_decoder, get_encoder, get_sfe, get_subset_layer


class VAE(nn.Module):

    def __init__(self, n, m, num_samples, subset_layer, encoder, decoder):
        super().__init__()
        self.n = n
        self.m = m
        self.num_samples = num_samples
        self.subset_layer = subset_layer
        self.encoder = encoder
        self.decoder = decoder
        self.extra = None

    def forward(self, x):
        if self.n > 1:
            logits = self.encoder(x)
            reshaped_logits = logits.reshape(-1, self.n, self.m)
            z = []
            extra = []
            for i in range(self.n):
                sample, e = self.subset_layer(reshaped_logits[:, i].unsqueeze(-1))
                z.append(sample)
                if e is not None:
                    extra.append(e)
            z = torch.stack(z, dim=-1)
            z = z.view(self.num_samples, -1, self.n * self.m)
            if len(extra) > 0:
                self.extra = torch.stack(extra, dim=-1).sum(-1)
        else:
            logits = self.encoder(x)
            z, self.extra = self.subset_layer(logits.unsqueeze(-1))

        z = z.squeeze(-1)
        x = self.decoder(z)
        return x, z, logits


def loss_fn(x, x_hat, logits, criterion, n, m):
    reconstruction = criterion(x_hat, x.unsqueeze(0).expand(x_hat.shape)).sum(
        (-1, -2, -3)
    )
    logits = logits.view(logits.shape[0], n, m)
    prior = torch.zeros(logits.shape[0], n, device=logits.device)
    for i in range(n):
        prior[:, i] += math.log(m) - Categorical(logits=logits[:, i]).entropy()
    prior = prior.mean()
    return reconstruction + prior


def random_latent(n, m, k, device, single=True):
    z = torch.zeros(n, m, device=device, requires_grad=False)
    z[:, :k] = 1
    for i in range(n):
        z[i] = z[i][torch.randperm(m)]
    z = z.permute(1, 0)
    z = z.flatten()
    z = z.expand(1, 1, -1)
    return z


def main(args):
    seed_everything(args.seed)
    device = torch.device(args.device)
    log_dir = (
        pathlib.Path(args.log_dir)
        / args.task
        / args.dataset
        / (args.name or args.sampler)
        / f"seed_{args.seed}"
    )
    logger = CSVLogger(log_dir, name=None, version="")

    data_module, train_loader, val_loader, test_loader = get_dataloaders(args)
    shape = data_module.size()

    n = m = args.d
    encoder = get_encoder(args.dataset, n * m)
    decoder = get_decoder(args.dataset, n * m, activation="sigmoid")
    subset_layer = get_subset_layer(args.k, args)
    model = VAE(n, m, args.num_samples, subset_layer, encoder, decoder).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss(reduction="none")
    sfe = get_sfe(args)

    training_steps = args.epochs * len(train_loader) if args.epochs else args.steps
    step_bar = tqdm(range(training_steps + 1), desc="Steps", unit="step")
    inf_loader = infinite_iter(train_loader)
    for global_step in step_bar:
        if global_step % 1000 == 0 or global_step == training_steps:
            model.eval()
            with torch.no_grad():
                val_bar = tqdm(
                    val_loader,
                    desc="Validation",
                    leave=False,
                )
                for x, _ in val_bar:
                    x = x.to(device)
                    x_hat, z, logits = model(x)
                    x_hat = x_hat.view(x_hat.shape[0], x_hat.shape[1], *shape)

                    losses = loss_fn(x, x_hat, logits, criterion, n, m)
                    loss = losses.mean()

                    logger.log_metrics(
                        {"global_step": global_step, "val_loss": loss.item()}
                    )

            if not args.no_plot:
                cmap = data_module.cmap()
                _, axs = plt.subplots(3, 5, figsize=(10, 6))
                for i in range(5):
                    axs[0, i].imshow(
                        x[i].view(shape).permute(1, 2, 0).cpu().numpy(), cmap=cmap
                    )
                    axs[1, i].imshow(z[0][i].view(n, m).cpu().numpy(), cmap="binary_r")
                    axs[2, i].imshow(
                        x_hat[0][i].view(shape).permute(1, 2, 0).cpu().numpy(),
                        cmap=cmap,
                    )
                    for ax in axs[:, i]:
                        ax.axis("off")
                plt.tight_layout()
                plt.savefig(
                    log_dir / f"reconstruction_{global_step}.png", bbox_inches="tight"
                )
                plt.close()

                _, axs = plt.subplots(2, 5, figsize=(10, 4))
                for i in range(5):
                    z = random_latent(n, m, args.k, device)
                    axs[0, i].imshow(z[0][0].view(n, m).cpu().numpy(), cmap="binary_r")
                    with torch.no_grad():
                        x_hat = model.decoder(z)
                    axs[1, i].imshow(
                        x_hat[0][0].view(shape).permute(1, 2, 0).cpu().numpy(),
                        cmap=cmap,
                    )
                    for ax in axs[:, i]:
                        ax.axis("off")
                plt.tight_layout()
                plt.savefig(
                    log_dir / f"generated_{global_step}.png", bbox_inches="tight"
                )
                plt.close()

            if global_step == training_steps:
                break  # Final validation after training

        model.train()
        x, y = next(inf_loader)
        x = x.to(device)

        st = time.time()
        x_hat, z, logits = model(x)
        logits.retain_grad()
        x_hat = x_hat.view(x_hat.shape[0], x_hat.shape[1], *shape)
        losses = loss_fn(x, x_hat, logits, criterion, n, m)
        loss = sfe(losses, model.extra)
        torch.cuda.synchronize(device)
        et = time.time()
        forward_time = et - st

        optimizer.zero_grad()
        st = time.time()
        loss.backward()
        torch.cuda.synchronize(device)
        et = time.time()
        backward_time = et - st

        # Compute the gradient norm
        logits_gard_mean = torch.norm(logits.grad, p=2, dim=-1).mean()
        logits_grad_var = torch.norm(logits.grad, p=2, dim=-1).var()

        optimizer.step()

        logger.log_metrics(
            {
                "global_step": global_step,
                "train_loss": loss.item(),
                "forward_time": forward_time,
                "backward_time": backward_time,
                "train_grad_norm": logits_gard_mean.item(),
                "train_grad_var": logits_grad_var.item(),
            }
        )

    model.eval()
    with torch.no_grad():
        test_losses = []

        test_bar = tqdm(
            test_loader,
            desc="Testing",
        )
        for x, y in test_bar:
            x = x.to(device)
            y = y.to(device)
            x_hat, z, logits = model(x)

            x_hat = x_hat.view(x_hat.shape[0], x_hat.shape[1], *shape)
            losses = loss_fn(x, x_hat, logits, criterion, n, m)
            loss = losses.mean()

            test_losses.append(loss.item())

        test_loss = sum(test_losses) / len(test_losses)
        logger.log_metrics({"test_loss": test_loss})

    logger.finalize("success")
