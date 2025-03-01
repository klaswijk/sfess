import cmath
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli


@torch.no_grad
def sample_subset(w, k, n_samples):
    shape = (n_samples,) + w.shape
    uniform = torch.rand(shape, device=w.device)
    z = -torch.log(-torch.log(uniform))
    w = w + z
    indices = w.topk(k, dim=-1).indices
    samples = torch.zeros_like(w)
    samples.scatter_(2, indices, 1)
    return samples


class SFESSSampler(nn.Module):

    def __init__(self, k, device, n_samples=1, cps=False):
        super(SFESSSampler, self).__init__()
        self.k = torch.tensor(k)
        self.device = device
        assert n_samples > 0
        self.n_samples = n_samples
        self.f = None
        self.cps = cps

    def forward(self, scores):
        batch, choices, _ = scores.shape
        flat_scores = scores.permute((0, 2, 1)).reshape(batch, choices)
        samples = sample_subset(flat_scores, self.k, self.n_samples)
        independent = (
            Bernoulli(logits=flat_scores.unsqueeze(0)).log_prob(samples.float()).sum(-1)
        )
        condition = log_prob_fourier(flat_scores, self.k)
        log_p = independent - condition
        samples = samples.reshape(self.n_samples, batch, 1, choices).permute(
            (0, 1, 3, 2)
        )
        return samples, log_p


def log_prob_fourier(logits, k):
    """
    "Closed-Form Expression for the Poisson-Binomial Probability Density Function"
    Fernandez and Williams (2010)
    """
    _, n = logits.size()
    probs = torch.sigmoid(logits).unsqueeze(-1).to(torch.complex64)

    i = torch.arange(n + 1, device=logits.device, dtype=logits.dtype).view(1, 1, -1)
    c = cmath.exp(2j * torch.pi / (n + 1))
    prod = torch.prod(probs * c**i + (1 - probs), dim=1)

    prob = torch.fft.fft(prod, norm="forward").real
    prob[:, k].clamp_(
        min=torch.finfo(prob.dtype).eps, max=1 - torch.finfo(prob.dtype).eps
    )
    result = prob[:, k].log()
    return result


def log_prob(logits, value, condition=None):
    k = int(value.sum())
    if k == value.size(0):
        return torch.tensor(0.0)
    independent = Bernoulli(logits=logits.unsqueeze(0)).log_prob(value.float()).sum()
    if condition is None:
        condition = log_prob_fourier(logits, k)
    return independent - condition


def score_function_estimator(loss_samples, log_p_samples, estimator):
    num_samples = loss_samples.size(0)
    if estimator == "reinforce":
        reinforce_loss = log_p_samples * loss_samples.detach()
        loss = loss_samples + reinforce_loss - reinforce_loss.detach()
        loss = loss.mean(0)
    elif estimator == "control-variate":
        i = torch.arange(num_samples, dtype=torch.long, device=loss_samples.device)
        baseline = loss_samples[~i].mean(0).detach()
        reinforce_loss = log_p_samples * (loss_samples - baseline).detach()
        loss = loss_samples + reinforce_loss - reinforce_loss.detach()
        loss = loss.mean(0)
    elif estimator is None:
        loss = loss_samples.mean(0)
    else:
        raise ValueError(f"Unknown estimator: {estimator}")
    return loss.mean()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler = SFESSSampler(2, device, n_samples=10)
    scores = torch.rand(64, 4, 1, device=device)
    samples, log_p = sampler(scores)
    print(samples.shape, log_p.shape)
