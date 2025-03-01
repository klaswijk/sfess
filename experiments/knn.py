"""Stochastic k-nearst neighbors (kNN)"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from lightning import seed_everything
from lightning.fabric.loggers import CSVLogger
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import get_dataloaders, infinite_iter
from models import get_encoder, get_sfe, get_subset_layer


def neighbor_dist(query, neighbors):
    return torch.cdist(query.unsqueeze(0), neighbors).squeeze(0)


def dknn_loss(knn, query_label, neighbor_labels, k):
    query_label = F.one_hot(query_label, 10)
    neighbor_labels = F.one_hot(neighbor_labels, 10)
    correct = (query_label.unsqueeze(1) * neighbor_labels.unsqueeze(0)).sum(-1)
    correct_in_top_k = (correct.unsqueeze(0) * knn).sum(-1)
    loss = -1 / k * correct_in_top_k
    return loss


def new_predict(query, neighbors, neighbor_labels, k):
    dist = neighbor_dist(query, neighbors)
    nearest = torch.topk(dist, k, largest=False).indices
    pred = torch.mode(neighbor_labels[nearest], dim=1).values.squeeze(1)
    return pred


class SubsetsDKNN(torch.nn.Module):
    def __init__(self, subset_layer):
        super(SubsetsDKNN, self).__init__()
        self.subset = subset_layer
        self.extra = None

    def forward(self, query, neighbors):
        scores = -neighbor_dist(query, neighbors)
        top_k, self.extra = self.subset(scores.unsqueeze(-1))
        top_k = top_k.squeeze(-1)
        return top_k


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
    train_loader_neighbor = DataLoader(
        train_loader.dataset,
        args.num_candidates,
        shuffle=True,
        num_workers=args.num_workers,
    )

    model = get_encoder(args.dataset, args.d).to(device)
    optimizer = Adam(model.parameters(), args.learning_rate)
    dknn_layer = SubsetsDKNN(get_subset_layer(args.k, args)).to(device)
    sfe = get_sfe(args)

    training_steps = args.epochs * len(train_loader) if args.epochs else args.steps
    step_bar = tqdm(range(training_steps + 1), desc="Steps", unit="step")
    inf_loader = infinite_iter(train_loader)
    neighbor_loader = infinite_iter(train_loader_neighbor)
    for global_step in step_bar:
        if global_step % 1000 == 0 or global_step == training_steps:
            model.eval()
            dknn_layer.eval()
            with torch.no_grad():
                xs = []
                embeddings = []
                labels = []
                val_embed_bar = tqdm(
                    train_loader_neighbor,
                    desc="Validation (embed train)",
                    leave=False,
                )
                for neighbor_x, neighbor_y in val_embed_bar:
                    neighbor_x = neighbor_x.to(device)
                    neighbor_y = neighbor_y.to(device)
                    embeddings.append(model(neighbor_x).unsqueeze(0))
                    labels.append(neighbor_y.unsqueeze(0))
                    xs.append(neighbor_x.unsqueeze(0))
                neighbors_x = torch.cat(xs, 1).reshape(-1, *data_module.size())
                neighbors_e = torch.cat(embeddings, 1).reshape(-1, args.d)
                labels = torch.cat(labels, 1).reshape(-1, 1)

                results = []
                val_bar = tqdm(
                    val_loader,
                    desc="Validation (query validation)",
                    leave=False,
                )
                query_es = []
                query_xs = []
                query_labels = []
                for queries in val_bar:
                    query_x, query_y = queries
                    query_x = query_x.to(device)
                    query_y = query_y.to(device)
                    query_e = model(query_x)  # batch_size x embedding_size
                    query_pred = new_predict(query_e, neighbors_e, labels, args.test_k)

                    results.append(query_pred.eq(query_y).float().mean().item())
                    query_es.append(query_e.unsqueeze(0))
                    query_xs.append(query_x.unsqueeze(0))
                    query_labels.append(query_y.unsqueeze(0))

                query_es = torch.cat(query_es, 1).reshape(-1, args.d)
                query_xs = torch.cat(query_xs, 1).reshape(-1, *data_module.size())
                query_labels = torch.cat(query_labels, 1).reshape(-1, 1)
                total_acc = np.mean(results)

                logger.log_metrics({"global_step": global_step, "accuracy": total_acc})

            if not args.no_plot:
                if args.d == 2 and global_step % 5000 == 0:
                    x = query_es[:, 0].cpu()
                    y = query_es[:, 1].cpu()
                    _x_min, _x_max = x.min(), x.max()
                    _y_min, _y_max = y.min(), y.max()

                    center = (x.mean(), y.mean())
                    longest_from_center = 1.05 * max(
                        abs(_x_min - center[0]),
                        abs(_x_max - center[0]),
                        abs(_y_min - center[1]),
                        abs(_y_max - center[1]),
                    )
                    x_min = center[0] - longest_from_center
                    x_max = center[0] + longest_from_center
                    y_min = center[1] - longest_from_center
                    y_max = center[1] + longest_from_center

                    fig, ax = plt.subplots(dpi=300, figsize=(8, 8))
                    ax.set_aspect("equal")
                    ax.set_axis_off()
                    plt.scatter(
                        query_es[:, 0].cpu(),
                        query_es[:, 1].cpu(),
                        c=query_labels.cpu(),
                        cmap="tab10",
                        s=1,
                    )
                    ax.autoscale()
                    ax.axis("square")
                    plt.xlim(x_min, x_max)
                    plt.ylim(y_min, y_max)
                    plt.tight_layout()
                    plt.savefig(
                        log_dir / f"embeddings_scatter_{global_step}.png",
                        bbox_inches="tight",
                    )
                    plt.close()

                    fig, ax = plt.subplots(dpi=300, figsize=(12, 12))
                    ax.set_aspect("equal")
                    ax.set_axis_off()
                    n_plots = None  # Set to None to plot all

                    ax.scatter(x[:n_plots], y[:n_plots])
                    for x0, y0, image in zip(
                        x[:n_plots], y[:n_plots], query_xs[:n_plots]
                    ):
                        if image.shape[0] == 1:
                            im = OffsetImage(
                                image.squeeze(0).cpu(),
                                cmap=data_module.cmap(),
                                zoom=0.3,
                            )
                        else:
                            im = OffsetImage(image.permute(1, 2, 0).cpu(), zoom=0.3)
                        ab = AnnotationBbox(im, (x0, y0), frameon=False)
                        ax.add_artist(ab)
                    ax.autoscale()
                    ax.axis("square")
                    plt.savefig(
                        log_dir / f"embeddings_no_decision_{global_step}.png",
                        bbox_inches="tight",
                    )

                    # Add decision boundaries
                    if True:

                        xx, yy = np.meshgrid(
                            np.linspace(x_min, x_max, 1000),
                            np.linspace(y_min, y_max, 1000),
                        )

                        # Create grid
                        grid = []
                        for i in range(xx.shape[0]):
                            for j in range(yy.shape[1]):
                                grid.append([xx[i, j], yy[i, j]])
                        grid = torch.tensor(grid).float().to(device)

                        grid_preds = []
                        grid_loader = DataLoader(grid, batch_size=128)
                        grid_bar = tqdm(
                            grid_loader,
                            desc="Plotting decision bound",
                            unit="batch",
                            leave=False,
                        )
                        for point in grid_bar:
                            point = point.to(device)
                            grid_pred = new_predict(
                                point, neighbors_e, labels, args.test_k
                            )
                            grid_preds.append(grid_pred)
                        grid_pred = torch.cat(grid_preds)

                        grid_pred = grid_pred.reshape(xx.shape)
                        plt.imshow(
                            grid_pred.cpu(),
                            extent=(x_min, x_max, y_min, y_max),
                            origin="lower",
                            alpha=0.5,
                            cmap="tab10",
                            interpolation="none",
                        )

                    plt.xlim(x_min, x_max)
                    plt.ylim(y_min, y_max)
                    plt.tight_layout()
                    plt.savefig(
                        log_dir / f"embeddings_{global_step}.png", bbox_inches="tight"
                    )
                    plt.close()
                    del fig, ax

        if global_step == training_steps:
            break  # Final validation after training

        model.train()
        dknn_layer.train()

        query_x, query_y = next(inf_loader)
        cand_x, cand_y = next(neighbor_loader)

        cand_x = cand_x.to(device)
        cand_y = cand_y.to(device)
        query_x = query_x.to(device)
        query_y = query_y.to(device)

        neighbor_e = model(cand_x).reshape(-1, args.d)
        query_e = model(query_x).reshape(-1, args.d)

        knn = dknn_layer(query_e, neighbor_e)
        losses = dknn_loss(knn, query_y, cand_y, args.k)
        loss = sfe(losses, dknn_layer.extra)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.log_metrics({"global_step": global_step, "train_loss": loss.item()})
        step_bar.set_postfix({"train_loss": loss.item(), "accuracy": total_acc})

    model.eval()
    dknn_layer.eval()
    with torch.no_grad():
        xs = []
        embeddings = []
        labels = []
        test_embed_bar = tqdm(
            train_loader_neighbor,
            desc="Testing (embed train)",
            leave=False,
        )
        for neighbor_x, neighbor_y in test_embed_bar:
            neighbor_x = neighbor_x.to(device)
            neighbor_y = neighbor_y.to(device)
            embeddings.append(model(neighbor_x).unsqueeze(0))
            labels.append(neighbor_y.unsqueeze(0))
            xs.append(neighbor_x.unsqueeze(0))
        neighbors_e = torch.cat(embeddings, 1).reshape(-1, args.d)
        labels = torch.cat(labels, 1).reshape(-1, 1)

        results = []
        test_bar = tqdm(
            test_loader,
            desc="Testing (query test)",
            leave=False,
        )
        query_es = []
        query_xs = []
        query_labels = []
        for queries in test_bar:
            query_x, query_y = queries
            query_x = query_x.to(device)
            query_y = query_y.to(device)
            query_e = model(query_x)  # batch_size x embedding_size
            query_pred = new_predict(query_e, neighbors_e, labels, args.test_k)

            results.append(query_pred.eq(query_y).float().mean().item())
            query_es.append(query_e.unsqueeze(0))
            query_xs.append(query_x.unsqueeze(0))
            query_labels.append(query_y.unsqueeze(0))

        query_es = torch.cat(query_es, 1).reshape(-1, args.d)
        query_xs = torch.cat(query_xs, 1).reshape(-1, *data_module.size())
        query_labels = torch.cat(query_labels, 1).reshape(-1, 1)
        total_acc = np.mean(results)

        logger.log_metrics({"global_step": global_step, "test_accuracy": total_acc})

    logger.finalize("success")
