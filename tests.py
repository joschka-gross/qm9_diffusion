import torch
import math
import config

from model import EDM
from util import rmsd, kabsch_alignment
from transform import PerturbNodePositions
from torch_geometric.data import Data, Batch


def test_dummy():

    model = EDM()
    perturb = PerturbNodePositions(model.alpha_bar)
    data1 = Data(
        z=torch.tensor([1, 1, 1, 1, 6]).long(),
        pos=torch.tensor(
            [[1, 1, 0], [1, 2, 0], [2, 1, 0], [-1, 2, 0], [0, -1, 0]]
        ).float(),
    )

    data2 = data1.clone()
    data2.pos = data2.pos + 42

    data1 = perturb(data1)
    data2 = perturb(data2)

    batch = Batch.from_data_list([data1, data2])

    loss = model.training_step(batch)


def test_kabsch():
    theta = 30
    rotation = torch.tensor(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    P = torch.randn(4, 2)
    Q = P @ rotation.t()

    P_ = kabsch_alignment(P, Q)
    assert torch.allclose(P_, Q)


def test_sample():
    model = EDM()
    perturb = PerturbNodePositions(model.alpha_bar, t=config.T)

    data1 = Data(
        z=torch.tensor([1, 1, 1, 1, 6]).long(),
        pos=torch.tensor(
            [[1, 1, 0], [1, 2, 0], [2, 1, 0], [-1, 2, 0], [0, -1, 0]]
        ).float(),
    )
    data2 = data1.clone()
    data1 = perturb(data1)
    data2 = perturb(data2)
    batch = Batch.from_data_list([data1, data2])

    model.validation_step(batch)


if __name__ == "__main__":
    test_sample()
    test_kabsch()
    test_dummy()
