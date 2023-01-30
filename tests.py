import torch

from model import EDM
from torch_geometric.data import Data, Batch


def test_dummy():

    data1 = Data(
        z=torch.tensor([1, 1, 1, 1, 6]).long(),
        pos=torch.tensor(
            [[1, 1, 0], [1, 2, 0], [2, 1, 0], [-1, 2, 0], [0, -1, 0]]
        ).float(),
    )

    data2 = data1.clone()
    data2.pos = data2.pos + 42

    batch = Batch.from_data_list([data1, data2])

    model = EDM()
    loss = model.training_step(batch)


if __name__ == "__main__":
    test_dummy()
