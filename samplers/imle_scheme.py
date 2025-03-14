import torch
import torch.nn as nn
from samplers.deterministic_scheme import select_from_candidates

from samplers.imle_pkg.noise import GumbelDistribution
from samplers.imle_pkg.target import TargetDistribution
from samplers.imle_pkg.wrapper import imle


LARGE_NUMBER = 1.0e10


class IMLESampler(nn.Module):
    def __init__(self, sample_k, device, n_samples, noise_scale, beta):
        super(IMLESampler, self).__init__()
        self.k = sample_k

        @imle(
            target_distribution=TargetDistribution(alpha=1.0, beta=beta),
            noise_distribution=GumbelDistribution(0.0, noise_scale, device),
            nb_samples=n_samples,
            input_noise_temperature=1.0,
            target_noise_temperature=1.0,
        )
        def imle_train_scheme(logits: torch.Tensor):
            return self.sample(logits)

        self.train_forward = imle_train_scheme

        @imle(
            target_distribution=None,
            noise_distribution=(
                GumbelDistribution(0.0, noise_scale, device) if n_samples > 1 else None
            ),
            nb_samples=n_samples,
            input_noise_temperature=1.0,
            target_noise_temperature=1.0,
        )
        def imle_val_scheme(logits: torch.Tensor):
            return self.sample(logits)

        self.val_forward = imle_val_scheme

        self.nnodes_list = None  # for potential usage

    @torch.no_grad()
    def sample(self, logits: torch.Tensor):
        local_logits = logits.detach()
        mask = select_from_candidates(local_logits, self.k)
        return mask, None

    def forward(self, logits: torch.Tensor, train=True):
        if train:
            return self.train_forward(logits)
        else:
            return self.val_forward(logits)

    @torch.no_grad()
    def validation(self, scores):
        return self.forward(scores, False)
