import wandb

import types
from typing import Callable, Optional, TypeVar
import torch
from torch_geometric.data.lightning_datamodule import LightningDataset
from torch_geometric.transforms import BaseTransform
from torch.utils.data import random_split, Dataset, Subset
import wandb

T = TypeVar("T")


class WithTransform(Dataset[T]):
    def __init__(self, dataset: Dataset[T], transform: Callable[[T], T]) -> None:
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index) -> T:
        return self.transform(self.dataset[index])

    def __len__(self) -> int:
        return len(self.dataset)


def make_data_module(
    dataset,
    batch_size: int,
    num_workers: int,
    train_size: float = 1.0,
    val_size: Optional[float] = None,
    test_size: Optional[float] = None,
    seed: int = 0,
    log_seed: bool = False,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
) -> LightningDataset:
    if log_seed:
        wandb.config.data_seed = seed
    split = dict()
    split["train_dataset"] = int(train_size * len(dataset))
    if val_size is not None:
        split["val_dataset"] = int(val_size * len(dataset))
    if test_size is not None:
        split["test_dataset"] = int(test_size * len(dataset))
    split["train_dataset"] += len(dataset) - sum(split.values())

    kwargs = dict(
        zip(
            split.keys(),
            random_split(
                dataset,
                list(split.values()),
                generator=torch.Generator().manual_seed(seed),
            ),
        )
    )
    if train_transform is not None:
        kwargs["train_dataset"] = WithTransform(
            kwargs["train_dataset"], train_transform
        )
    if val_transform is not None:
        assert (
            "val_dataset" in kwargs
        ), "Cannot set val transform when val_size is unspecified."
        kwargs["val_dataset"] = WithTransform(kwargs["val_dataset"], val_transform)

    kwargs["batch_size"] = batch_size
    kwargs["num_workers"] = num_workers

    return LightningDataset(**kwargs)
