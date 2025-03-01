from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, KMNIST, MNIST, FashionMNIST
from torchvision.transforms import ToTensor


class TorchvisionDataModule(LightningDataModule, ABC):
    def __init__(
        self,
        dataset_class,
        data_dir="~/datasets",
        batch_size=32,
        num_workers=0,
        seed=None,
        split_lengths=(0.8, 0.2),
    ):
        super().__init__()
        self.dataset_class = dataset_class
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_lengths = split_lengths
        if seed is not None:
            self.generator = torch.Generator().manual_seed(seed)
        else:
            self.generator = None

    def prepare_data(self):
        self.dataset_class(self.data_dir, train=True, download=True)
        self.dataset_class(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full = self.dataset_class(
                self.data_dir, train=True, transform=self.transform()
            )
            self.train, self.val = random_split(
                full, self.split_lengths, self.generator
            )
        if stage == "test" or stage is None:
            self.test = self.dataset_class(
                self.data_dir, train=False, transform=self.transform()
            )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            generator=self.generator,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    @abstractmethod
    def size(self):
        raise NotImplementedError

    @abstractmethod
    def num_classes(self):
        raise NotImplementedError

    def transform(self):
        return ToTensor()

    def cmap(self):
        return None


class MNISTDataModule(TorchvisionDataModule):
    def __init__(self, data_dir="~/datasets", batch_size=32, num_workers=0, seed=None):
        super().__init__(MNIST, data_dir, batch_size, num_workers, seed, (50000, 10000))

    def size(self):
        return torch.Size((1, 28, 28))

    def num_classes(self):
        return 10

    def reconstruction_loss(self):
        return nn.BCELoss

    def cmap(self):
        return "gray"


class FashionMNISTDataModule(TorchvisionDataModule):
    def __init__(self, data_dir="~/datasets", batch_size=32, num_workers=0, seed=None):
        super().__init__(
            FashionMNIST, data_dir, batch_size, num_workers, seed, (50000, 10000)
        )

    def size(self):
        return torch.Size((1, 28, 28))

    def num_classes(self):
        return 10

    def reconstruction_loss(self):
        return nn.BCELoss

    def cmap(self):
        return "gray_r"


class KMNISTDataModule(TorchvisionDataModule):
    def __init__(self, data_dir="~/datasets", batch_size=32, num_workers=0, seed=None):
        super().__init__(
            KMNIST, data_dir, batch_size, num_workers, seed, (50000, 10000)
        )

    def size(self):
        return torch.Size((1, 28, 28))

    def num_classes(self):
        return 10

    def reconstruction_loss(self):
        return nn.BCELoss

    def cmap(self):
        return "gray"


class CIFAR10DataModule(TorchvisionDataModule):
    def __init__(self, data_dir="~/datasets", batch_size=32, num_workers=0, seed=None):
        super().__init__(
            CIFAR10, data_dir, batch_size, num_workers, seed, (45000, 5000)
        )

    def size(self):
        return torch.Size((3, 32, 32))

    def num_classes(self):
        return 10

    def reconstruction_loss(self):
        return nn.MSELoss


def get_dataloaders(args):
    data_module = {
        "mnist": MNISTDataModule,
        "fashion": FashionMNISTDataModule,
        "cifar10": CIFAR10DataModule,
    }[args.dataset](
        data_dir=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data_module.prepare_data()
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    return data_module, train_loader, val_loader, test_loader


def infinite_iter(dataloader):
    while True:
        for batch in dataloader:
            yield batch


if __name__ == "__main__":

    for datamodule_class in (
        MNISTDataModule,
        FashionMNISTDataModule,
        CIFAR10DataModule,
    ):
        print("\n\n", datamodule_class)
        datamodule = datamodule_class()
        datamodule.prepare_data()
        datamodule.setup()

        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        test_loader = datamodule.test_dataloader()

        print("Train batches:", len(train_loader))
        print("Validation batches:", len(val_loader))
        print("Test batches:", len(test_loader))
        print("Shape:", datamodule.size())
