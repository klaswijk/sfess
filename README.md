## SFESS: Score Function Estimators for $k$-Subset Sampling<br><sub>Official PyTorch implementation of the ICLR 2025 paper</sub>

**SFESS: Score Function Estimators for $k$-Subset Sampling**<br>
Klas Wijk, Ricardo Vinuesa & Hossein Azizpour<br>
https://openreview.net/forum?id=q87GUkdQBm

## Overview
Implementations of gradient estimators for subset distributions:
- Gumbel softmax top-$k$ (GS) (Xie and Ermon 2019) [[arXiv]](https://arxiv.org/abs/1901.10517)
- Straight-through Gumbel softmax top-$k$ (STGS) (Xie and Ermon 2019) [[arXiv]](https://arxiv.org/abs/1901.10517)
- Implicit maximum likelihood estimation (I-MLE) (Niepert et al. 2021) [[arXiv]](https://arxiv.org/abs/2106.01798)
- SIMPLE (Ahmed et al. 2023) [[arXiv]](https://arxiv.org/abs/2210.01941)
- Score function estimators for $k$-subset sampling (SFESS) [[this paper]](https://openreview.net/forum?id=q87GUkdQBm)

Multiple experiments:
- Feature selection
- Learning to explain (L2X)
- Subset VAE
- Stochastic k-nearest neighbors

<p align="center">
    <img src="images/experiments.svg" width="66%" alt="Overview of experiments.">
</p>

## Requirements
```
numpy
matplotlib
seaborn
pytorch
torchvision
lightning
torchmetrics
```

## Usage
To see the list of parameters for an experiment, run:
```sh
python main.py [task] --help
```
where `[task]` is one of `{l2x,vae,knn}`.

The toy experiment is found in `/notebooks`.

## Acknowledgements
This implementation extends code from:
- https://github.com/UCLA-StarAI/SIMPLE (toy experiment)
- https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/sampling/subsets.html (stochastic $k$-nearest neighbors)
- https://github.com/chendiqian/PR-MPNN and correspondence with Andrei-Marian Manolache, Ahmed Kareem, and Mathias Niepert (samplers)