from typing import Optional, Tuple
import torch
from torch import Tensor, LongTensor
from torch_scatter import scatter


def nodes_per_graph(index: LongTensor):
    n = torch.ones_like(index)
    return scatter(n, index, 0, reduce="sum")


def expand_to_pos_shape(
    batch_level_attribute: Tensor, index: Tensor, num_spatial_dims: int = 3
) -> Tensor:
    return batch_level_attribute[index].unsqueeze(-1).expand(-1, num_spatial_dims)


def centered_pos(pos: Tensor, batch: Optional[LongTensor] = None) -> Tensor:
    if batch is None:
        batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
    mean = scatter(pos, batch, 0, reduce="mean")
    return pos - mean[batch]


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def rmsd(x: torch.Tensor, y: torch.Tensor) -> Tensor:
    return (x - y).pow(2).sum(dim=1).mean().sqrt()


@torch.no_grad()
def kabsch_alignment(P: torch.Tensor, Q: torch.Tensor) -> Tuple[Tensor, Tensor]:
    assert P.size() == Q.size()
    p_mean = P.mean(0)
    q_mean = Q.mean(0)

    P_c = P - p_mean
    Q_c = Q - q_mean

    H = P_c.t() @ Q_c
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.mH
    R = V @ U.t()
    t = q_mean - (R @ p_mean).t()

    return P @ R.t() + t
