from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from samplers.get_sampler import SamplerArgs, get_sampler
from samplers.sfess.sfess import score_function_estimator


def get_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "softmax":
        return nn.Softmax(dim=-1)
    elif activation is None or activation == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation: {activation}")


class MNISTEncoder(nn.Sequential):

    def __init__(self, dim, activation=None):
        super().__init__(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, dim),
            get_activation(activation),
        )


class MNISTDecoder(nn.Sequential):

    def __init__(self, dim, activation=None):
        super().__init__(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Unflatten(2, (1, 28, 28)),
            get_activation(activation),
        )


class CIFAR10Encoder(nn.Sequential):

    def __init__(self, dim, activation=None):
        super().__init__(
            nn.Flatten(),
            nn.Linear(3072, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, dim),
            get_activation(activation),
        )


class CIFAR10Decoder(nn.Sequential):

    def __init__(self, dim, activation=None):
        super().__init__(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 3072),
            nn.Unflatten(2, (3, 32, 32)),
            get_activation(activation),
        )


class CIFAR10ConvEncoder(nn.Sequential):

    def __init__(self, dim, activation=None):
        super().__init__(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, dim),
            get_activation(activation),
        )


class CIFAR10ConvDecoder(nn.Module):

    def __init__(self, dim, activation=None):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * 4 * 4),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, 2, 1, output_padding=1),
        )
        self.activation = get_activation(activation)

    def forward(self, x):
        shape = x.size()
        x = self.linear(x)
        x = x.view(shape[0] * shape[1], -1)
        x = self.conv(x)
        x = x.view(shape[0], shape[1], 3, 32, 32)
        x = self.activation(x)
        return x


def get_encoder(dataset, dim, activation=None, conv=False):
    if dataset in ("mnist", "fashion"):
        return MNISTEncoder(dim, activation)
    elif dataset == "cifar10":
        cifar_class = CIFAR10ConvEncoder if conv else CIFAR10Encoder
        return cifar_class(dim, activation)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_decoder(dataset, dim, activation=None, conv=False):
    if dataset in ("mnist", "fashion"):
        return MNISTDecoder(dim, activation)
    elif dataset == "cifar10":
        cifar_class = CIFAR10ConvDecoder if conv else CIFAR10Decoder
        return cifar_class(dim, activation)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


class SubsetLayer(nn.Module):

    def __init__(self, subset_layer, k, num_samples):
        super(SubsetLayer, self).__init__()
        self.subset = subset_layer
        self.k = k
        self.num_samples = num_samples

    def forward(self, logits):
        if self.training:
            res = self.subset(logits)
            return res
        else:
            indices = torch.topk(logits.squeeze(-1), self.k, dim=1)[1]
            khot = F.one_hot(indices, logits.size(1)).sum(1).float()
            khot = khot.unsqueeze(0).unsqueeze(-1).expand(self.num_samples, -1, -1, -1)
            return khot, None


def get_subset_layer(k, args):
    name = {
        "sfess": "sfess",
        "sfess-vr": "sfess",
        "gumbel": "gumbel",
        "st-gumbel": "gumbel",
        "simple": "simple",
        "imle": "imle",
    }[args.sampler]
    sampler_args = SamplerArgs(
        name=name,
        sample_k=k,
        n_samples=args.num_samples,
        noise_scale=args.noise_scale,
        beta=args.beta,
        tau=args.tau,
        hard=args.sampler != "gumbel",
    )
    sampler = get_sampler(sampler_args, device=args.device)
    subset_layer = SubsetLayer(sampler, k, args.num_samples)
    return subset_layer


def get_sfe(args):
    estimator = {
        "sfess": "reinforce",
        "sfess-vr": "control-variate",
        "gumbel": None,
        "st-gumbel": None,
        "simple": None,
        "imle": None,
    }[args.sampler]
    return partial(score_function_estimator, estimator=estimator)
