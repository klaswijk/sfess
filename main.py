import argparse

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Gradient estimation for k-subset sampling",
        usage="python main.py [task] [options]",
        description="Run experiments for gradient estimation with k-subset sampling",
    )
    task_parsers = parser.add_subparsers(dest="task", required=True)

    # Learning to explain (L2X)
    l2x_parser = task_parsers.add_parser("l2x")
    l2x_parser.add_argument("--k", type=int, default=50)
    l2x_parser.add_argument(
        "--instancewise",
        action="store_true",
        help="Do instancewise selection instead of global.",
    )
    l2x_parser.add_argument("--reconstruction", action="store_true")

    # Variational autoencoder with a k-subset latent space
    vae_parser = task_parsers.add_parser("vae")
    vae_parser.add_argument("--d", type=int, default=10)
    vae_parser.add_argument("--k", type=int, default=5)

    # Stochastic k-nearest neighbors
    knn_parser = task_parsers.add_parser("knn")
    knn_parser.add_argument("--d", type=int, default=2)
    knn_parser.add_argument("--k", type=int, default=10)
    knn_parser.add_argument("--test-k", type=int, default=10)
    knn_parser.add_argument("--num_candidates", type=int, default=128)

    # Common arguments
    for subparser in [l2x_parser, vae_parser, knn_parser]:
        subparser.add_argument(
            "--dataset",
            type=str,
            choices=["mnist", "fashion", "cifar10"],
            required=True,
        )
        sampler_group = subparser.add_argument_group("Sampler")
        sampler_group.add_argument(
            "--sampler",
            type=str,
            choices=["simple", "gumbel", "st-gumbel", "sfess", "sfess-vr", "imle"],
            required=True,
        )
        sampler_group.add_argument(
            "--num_samples",
            type=int,
            default=1,
            help="Number of samples for variance reduction",
        )
        sampler_group.add_argument(
            "--noise_scale",
            type=float,
            default=1.0,
            help="Noise scale parameter for I-MLE",
        )
        sampler_group.add_argument(
            "--beta", type=float, default=1.0, help="Beta parameter for I-MLE"
        )
        sampler_group.add_argument(
            "--tau",
            type=float,
            default=0.5,
            help="Temperature parameter for Gumbel-Softmax",
        )
        training_group = subparser.add_argument_group("Training")
        training_group.add_argument("--epochs", type=int, default=None)
        training_group.add_argument("--steps", type=int, default=50_000)
        training_group.add_argument("--batch_size", type=int, default=128)
        training_group.add_argument("--learning_rate", type=float, default=1e-4)
        misc_group = subparser.add_argument_group("Misc")
        misc_group.add_argument("--seed", type=int, default=0)
        misc_group.add_argument("--log_dir", type=str, default="logs")
        misc_group.add_argument(
            "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
        )
        misc_group.add_argument("--num_workers", type=int, default=4)
        misc_group.add_argument("--name", type=str, default=None)
        misc_group.add_argument("--data_path", type=str, default="~/datasets")
        misc_group.add_argument("--no_plot", action="store_true")

    args = parser.parse_args()
    for key, value in vars(args).items():
        print(f"{key}\t: {value}".expandtabs(15))
    print()

    if args.task == "l2x":
        from experiments.l2x import main
    elif args.task == "vae":
        from experiments.vae import main
    elif args.task == "knn":
        from experiments.knn import main

    main(args)
