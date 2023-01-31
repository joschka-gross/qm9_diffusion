from typing import Callable
import config

import torch
from torch import Tensor
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

from util import centered_pos, cosine_beta_schedule, kabsch_alignment


class PerturbNodePositions(BaseTransform):
    def __init__(self, alpha_bar: Tensor) -> None:
        self.T = config.T
        self.alpha_bar = alpha_bar
        super().__init__()

    def __call__(self, data: Data) -> Data:
        t = torch.randint(0, self.T, (1,))
        eps = torch.randn_like(data.pos)
        pos = data.pos - torch.mean(data.pos, 0, keepdim=True)
        alpha_bar_t = torch.full_like(pos, fill_value=self.alpha_bar[t].item())
        perturbed_pos = alpha_bar_t.sqrt() * pos + (1 - alpha_bar_t).sqrt() * eps

        pos_aligned = kabsch_alignment(pos, perturbed_pos)
        eps = (perturbed_pos - alpha_bar_t.sqrt() * pos_aligned) / (
            1 - alpha_bar_t
        ).sqrt()

        data.perturbed_pos = perturbed_pos
        data.t = t.float()
        data.eps = eps

        return data
