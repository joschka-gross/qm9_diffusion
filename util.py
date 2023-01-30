import torch
from torch import Tensor, LongTensor
from torch_scatter import scatter

def nodes_per_graph(batch: LongTensor):
    n = torch.ones_like(batch)
    return scatter(n, batch, 0, reduce="sum")

def centered_pos(pos: Tensor, batch: LongTensor) -> Tensor:
    mean = scatter(pos, batch, 0, reduce="mean")
    return pos - mean[batch]
