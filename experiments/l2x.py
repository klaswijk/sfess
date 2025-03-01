"""Feature selection and Learning to Explain (L2X)"""

import pathlib

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import seed_everything
from lightning.fabric.loggers import CSVLogger
from torch.optim import Adam
from torchmetrics.functional import accuracy
from tqdm import tqdm

from datasets import get_dataloaders, infinite_iter
from models import get_encoder, get_sfe, get_subset_layer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class L2X(nn.Module):

    def __init__(
        self,
        input_size,
        num_samples,
        k,
        subset_layer,
        cond_network,
        classifier,
        instancewise=False,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.k = k
        self.subset_layer = subset_layer
        if instancewise:
            self.cond_network = cond_network
        else:
            self.logits = nn.Parameter(torch.randn(input_size))
        self.classifier = classifier
        self.instancewise = instancewise
        self.extra = None

    def sample_mask(self, logits, batch):
        if logits.dim() == 1:
            logits = logits.expand(batch, -1)
        if self.training:
            mask, self.extra = self.subset_layer(logits.unsqueeze(-1))
            mask = mask.squeeze(-1)
            return mask
        else:
            indices = torch.topk(logits, self.k, dim=-1)[1]
            mask = F.one_hot(indices, logits.size(-1)).sum(1)
            mask = mask.float()
            mask = mask.expand(self.num_samples, -1, -1)
            return mask

    def forward(self, x):
        x = x.flatten(2).mean(1)
        if self.instancewise:
            logits = self.cond_network(x)
            mask = self.sample_mask(logits, x.size(0))
        else:
            # batch = 1 -> same mask for all samples. This is a lot faster
            mask = self.sample_mask(self.logits, 1)
        # x = torch.einsum("sbd,bd->sbd", mask, x)
        x = mask * x
        x = x.flatten(0, 1)
        x = self.classifier(x)
        x = x.view(self.num_samples, -1, x.size(-1))
        return x, mask


def main(args):
    seed_everything(args.seed)
    plt.switch_backend("agg")  # No display
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

    instancewise = args.instancewise
    subset_layer = get_subset_layer(args.k, args)
    y_shape = 28**2 if args.reconstruction else data_module.num_classes()
    if args.reconstruction:
        classifier = nn.Sequential(
            nn.Linear(28**2, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 28**2),
            nn.Sigmoid(),
        )
    else:
        classifier = get_encoder(args.dataset, y_shape, activation="softmax")
    cond_network = nn.Sequential(
        nn.Linear(28**2, 100), nn.ReLU(), nn.Linear(100, 28**2)
    )
    model = L2X(
        shape[1] * shape[2],
        args.num_samples,
        args.k,
        subset_layer,
        cond_network,
        classifier,
        instancewise,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = (
        nn.CrossEntropyLoss(reduction="none")
        if not args.reconstruction
        else nn.BCELoss(reduction="none")
    )
    sfe = get_sfe(args)

    training_steps = args.epochs * len(train_loader) if args.epochs else args.steps
    step_bar = tqdm(range(training_steps + 1), desc="Steps", unit="step")
    inf_loader = infinite_iter(train_loader)
    for global_step in step_bar:
        if global_step % 1000 == 0 or global_step == training_steps:
            model.eval()
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc="Validation", leave=False)
                for x, y in val_bar:
                    x = x.to(device)
                    if args.reconstruction:
                        y = x
                        y = (
                            y.reshape(1, -1, y_shape)
                            .expand(args.num_samples, -1, -1)
                            .reshape(-1, y_shape)
                        )
                    else:
                        y = y.to(device)
                        y = y.reshape(1, -1).expand(args.num_samples, -1).reshape(-1)
                    y_pred, mask = model(x)
                    y_pred = y_pred.reshape(-1, y_shape)

                    losses = criterion(y_pred, y).view(args.num_samples, -1)
                    loss = losses.mean()

                    acc = accuracy(y_pred, y, "multiclass", num_classes=10).item()
                    logger.log_metrics(
                        {
                            "global_step": global_step,
                            "val_loss": loss.item(),
                            "val_acc": acc,
                        }
                    )

            if not args.no_plot:
                if instancewise:
                    _, axs = plt.subplots(1, 5, figsize=(20, 4))
                    for i in range(5):
                        x = x.mean(1, keepdim=True)
                        axs[i].axis("off")
                        axs[i].imshow(
                            x[i].reshape(*shape[1:]).detach().cpu().numpy(), cmap="gray"
                        )
                        axs[i].imshow(
                            mask[0, i].reshape(*shape[1:]).detach().cpu().numpy(),
                            alpha=0.5,
                            cmap="Reds",
                        )
                    plt.tight_layout()
                    plt.savefig(log_dir / f"input_{global_step}.png")
                    plt.close()
                else:
                    plt.imshow(model.logits.reshape(*shape[1:]).detach().cpu().numpy())
                    plt.axis("off")
                    plt.title(f"Logits at epoch {global_step}")
                    plt.savefig(log_dir / f"epoch_{global_step}.png")
                    plt.close()

                    plt.imshow(
                        model.sample_mask(model.logits, 1)[0]
                        .reshape(*shape[1:])
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    plt.axis("off")
                    plt.title(f"Mask at epoch {global_step}")
                    plt.savefig(log_dir / f"mask_{global_step}.png")
                    plt.close()
                if args.reconstruction:
                    _, axs = plt.subplots(2, 10, figsize=(20, 4))

                    from matplotlib.colors import ListedColormap

                    cmap = ListedColormap([(0, 0, 0, 0), (1, 0, 0, 0.5)])

                    for i in range(10):

                        axs[0, i].imshow(
                            x[i].reshape(*shape[1:]).detach().cpu().numpy(), cmap="gray"
                        )
                        axs[0, i].imshow(
                            mask[0, i if instancewise else 0]
                            .reshape(*shape[1:])
                            .detach()
                            .cpu()
                            .numpy(),
                            cmap=cmap,
                        )
                        axs[0, i].axis("off")
                        axs[1, i].imshow(
                            y_pred[i].reshape(*shape[1:]).detach().cpu().numpy(),
                            cmap="gray",
                        )
                        axs[1, i].axis("off")
                    plt.tight_layout()
                    plt.savefig(log_dir / f"reconstruction_{global_step}.png")
                    plt.close()

        if global_step == training_steps:
            break  # Final validation after training

        model.train()
        x, y = next(inf_loader)

        x = x.to(device)
        if args.reconstruction:
            y = x
            y = (
                y.reshape(1, -1, y_shape)
                .expand(args.num_samples, -1, -1)
                .reshape(-1, y_shape)
            )
        else:
            y = y.to(device)
            y = y.reshape(1, -1).expand(args.num_samples, -1).reshape(-1)
        y_pred, _ = model(x)

        y_pred = y_pred.reshape(-1, y_shape)
        losses = criterion(y_pred, y).view(args.num_samples, -1)

        loss = sfe(losses, model.extra)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = accuracy(y_pred, y, "multiclass", num_classes=10).item()
        logger.log_metrics(
            {
                "global_step": global_step,
                "train_loss": loss.item(),
                "train_acc": acc,
            }
        )

    model.eval()
    with torch.no_grad():
        test_losses = []
        test_accuracies = []

        test_bar = tqdm(test_loader, desc="Testing")
        for x, y in test_bar:
            x = x.to(device)
            if args.reconstruction:
                y = x
                y = (
                    y.reshape(1, -1, y_shape)
                    .expand(args.num_samples, -1, -1)
                    .reshape(-1, y_shape)
                )
            else:
                y = y.to(device)
                y = y.reshape(1, -1).expand(args.num_samples, -1).reshape(-1)
            y_pred, _ = model(x)
            y_pred = y_pred.reshape(-1, y_shape)
            losses = criterion(y_pred, y).view(args.num_samples, -1)
            loss = losses.mean()

            test_losses.append(loss.item())
            test_accuracies.append(
                accuracy(y_pred, y, "multiclass", num_classes=10).item()
            )

        logger.log_metrics({"test_loss": sum(test_losses) / len(test_losses)})

    logger.finalize("success")
