import torch
from torch import Tensor, LongTensor
import torch.nn as nn
from torch_cluster import radius_graph
from torch_scatter import scatter
from torch.nn.functional import silu
import math

import pytorch_lightning as pl

from util import centered_pos, nodes_per_graph

MAX_D = 5.0
HIDDEN_CHANNELS = 32
TIME_DIM = HIDDEN_CHANNELS * 4
REDUCE = "sum"
NUM_LAYERS = 3
T = 100
BATCH_SIZE = 32
LEARNING_RATE = 3e-4

config = {name.lower(): value for name, value in vars().items() if name.upper() == name}


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


class WithLinearProjection(nn.Module):
    def __init__(self, dim, fn) -> None:
        super().__init__()
        self.w = nn.Linear(dim, dim, bias=False)
        self.fn = fn

    def forward(self, x):
        return self.w(self.fn(x))


class ExpnormRBFEmbedding(nn.Module):
    def __init__(self, size: int, d_cut: float, trainable: bool = True) -> None:
        super().__init__()
        self.size = size
        self.d_cut = nn.parameter.Parameter(
            torch.tensor([d_cut], dtype=torch.float32), requires_grad=False
        )

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    # taken from torchmd-net
    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(-self.d_cut.squeeze())
        means = torch.linspace(start_value, 1, self.size)
        betas = torch.tensor([(2 / self.size * (1 - start_value)) ** -2] * self.size)
        return means, betas

    def cosine_cutoff(self, d: Tensor) -> Tensor:
        d = d.clamp_max(self.d_cut)
        return 0.5 * (torch.cos(d * torch.pi / self.d_cut) + 1)

    def forward(self, d: Tensor) -> Tensor:
        cutoff = self.cosine_cutoff(d)
        rbf = torch.exp(-self.betas * (torch.exp(-d) - self.means).pow(2))
        return cutoff * rbf


class MessageLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dist_emb = WithLinearProjection(
            HIDDEN_CHANNELS, ExpnormRBFEmbedding(HIDDEN_CHANNELS, MAX_D)
        )
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * HIDDEN_CHANNELS, HIDDEN_CHANNELS),
            nn.SiLU(),
            nn.Linear(HIDDEN_CHANNELS, HIDDEN_CHANNELS),
            nn.SiLU(),
        )

        self.gate = nn.Sequential(
            nn.Linear(HIDDEN_CHANNELS, HIDDEN_CHANNELS), nn.Sigmoid()
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(TIME_DIM, HIDDEN_CHANNELS * 2),
            nn.SiLU(),
            nn.Linear(HIDDEN_CHANNELS * 2, HIDDEN_CHANNELS * 2),
        )

        self.comb_mlp = nn.Sequential(
            nn.Linear(2 * HIDDEN_CHANNELS, HIDDEN_CHANNELS),
            nn.SiLU(),
            nn.Linear(HIDDEN_CHANNELS, HIDDEN_CHANNELS),
        )

        self.coord_message_mlp = nn.Sequential(
            nn.Linear(HIDDEN_CHANNELS, HIDDEN_CHANNELS),
            nn.SiLU(),
            nn.Linear(HIDDEN_CHANNELS, 1),
        )

    def forward(
        self, x: Tensor, edge_index: Tensor, pos: Tensor, time: Tensor, batch: Tensor
    ) -> Tensor:

        # diffusion time step interacts with node features
        scale_shift = self.time_mlp(time)
        scale_shift = scale_shift[batch]
        scale = scale_shift[:, :HIDDEN_CHANNELS]
        shift = scale_shift[:, HIDDEN_CHANNELS:]
        x = silu(x * (1 + scale) + shift)

        # node features and positions and distances for all edges
        i, j = edge_index
        x_i, x_j = x[i], x[j]
        pos_i, pos_j = pos[i], pos[j]
        pos_diff = pos_i - pos_j
        dist = (pos_i - pos_j).pow(2).sum(dim=1, keepdim=True).sqrt()
        dist_emb = self.dist_emb(dist)

        m_ij = (
            self.message_mlp(torch.cat((x_i, x_j), dim=1)) * dist_emb * self.gate(x_j)
        )

        m_i = x.new_zeros(x.size())
        scatter(m_ij, j, dim=0, out=m_i, reduce=REDUCE)

        x = silu(x + self.comb_mlp(torch.cat((x, m_i), dim=1)))

        pos_updates = pos_diff * self.coord_message_mlp(m_ij)
        pos_update_i = pos.new_zeros(pos.size())
        scatter(pos_updates, j, dim=0, out=pos_update_i, reduce="mean")
        new_pos = pos + pos_update_i

        return x, new_pos


class EGNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([MessageLayer() for _ in range(NUM_LAYERS)])

    def forward(self, x, pos, t, batch):
        for layer in self.layers:
            edge_index = radius_graph(pos, 5.0, batch)
            x, pos = layer(x, edge_index, pos, t, batch)
        return x, pos


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class EDM(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.T = T
        self.beta = nn.Parameter(cosine_beta_schedule(self.T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_bar = nn.Parameter(torch.cumprod(self.alpha, 0), requires_grad=False)
        self.egnn = EGNN()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(HIDDEN_CHANNELS),
            nn.Linear(HIDDEN_CHANNELS, TIME_DIM),
            nn.GELU(),
            nn.Linear(TIME_DIM, TIME_DIM),
        )
        self.initial_node_embed = nn.Embedding(100, HIDDEN_CHANNELS)

    @torch.no_grad()
    def sample(self):
        ...

    def training_step(self, data, *args, **kwargs):
        batch_size = data.batch.max() + 1
        t = torch.randint(0, self.T, (batch_size,), device=data.pos.device)
        t_batch = t[data.batch]
        alpha_bar_t = self.alpha_bar[t_batch]
        alpha_bar_t = alpha_bar_t.unsqueeze(1).expand(-1, data.pos.size(1))
        eps = torch.randn_like(data.pos)

        pos = centered_pos(data.pos, data.batch)
        noised_pos = alpha_bar_t.sqrt() * pos + (1 - alpha_bar_t).sqrt() * eps

        x = self.initial_node_embed(data.z)
        t_emb = self.time_mlp(t.float())
        _, eps_pred = self.egnn(x, noised_pos, t_emb, data.batch)

        loss = scatter(
            (eps - eps_pred).pow(2).sum(dim=1).sqrt(), data.batch, 0, reduce="sum"
        )
        N = nodes_per_graph(data.batch)
        loss = loss / N
        loss = loss.mean()
        self.log("train_loss", loss, batch_size=batch_size, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
