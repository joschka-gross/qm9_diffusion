import math
import wandb

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import LongTensor, Tensor
from torch.nn.functional import silu, pad
from torch_cluster import radius_graph
from torch_scatter import scatter
from torch_geometric.data import Data

import config
from util import centered_pos, cosine_beta_schedule, nodes_per_graph
from visualization import plot_point_cloud_3d
import matplotlib.pyplot as plt


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
            config.HIDDEN_CHANNELS,
            ExpnormRBFEmbedding(config.HIDDEN_CHANNELS, config.MAX_D),
        )
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * config.HIDDEN_CHANNELS, config.HIDDEN_CHANNELS),
            nn.SiLU(),
            nn.Linear(config.HIDDEN_CHANNELS, config.HIDDEN_CHANNELS),
            nn.SiLU(),
        )

        self.gate = nn.Sequential(
            nn.Linear(config.HIDDEN_CHANNELS, config.HIDDEN_CHANNELS), nn.Sigmoid()
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(config.TIME_DIM, config.HIDDEN_CHANNELS * 2),
            nn.SiLU(),
            nn.Linear(config.HIDDEN_CHANNELS * 2, config.HIDDEN_CHANNELS * 2),
        )

        self.comb_mlp = nn.Sequential(
            nn.Linear(2 * config.HIDDEN_CHANNELS, config.HIDDEN_CHANNELS),
            nn.SiLU(),
            nn.Linear(config.HIDDEN_CHANNELS, config.HIDDEN_CHANNELS),
        )

        self.coord_message_mlp = nn.Sequential(
            nn.Linear(config.HIDDEN_CHANNELS, config.HIDDEN_CHANNELS),
            nn.SiLU(),
            nn.Linear(config.HIDDEN_CHANNELS, 1),
        )

    def forward(
        self, x: Tensor, edge_index: Tensor, pos: Tensor, time: Tensor, batch: Tensor
    ) -> Tensor:

        # diffusion time step interacts with node features
        scale_shift = self.time_mlp(time)
        scale_shift = scale_shift[batch]
        scale = scale_shift[:, : config.HIDDEN_CHANNELS]
        shift = scale_shift[:, config.HIDDEN_CHANNELS :]
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
        scatter(m_ij, j, dim=0, out=m_i, reduce=config.REDUCE)

        x = silu(x + self.comb_mlp(torch.cat((x, m_i), dim=1)))

        pos_updates = pos_diff * self.coord_message_mlp(m_ij)
        pos_update_i = pos.new_zeros(pos.size())
        scatter(pos_updates, j, dim=0, out=pos_update_i, reduce="mean")
        new_pos = pos + pos_update_i

        return x, new_pos


class EGNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([MessageLayer() for _ in range(config.NUM_LAYERS)])

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
        self._initialize_variance_schedule()
        self.egnn = EGNN()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(config.HIDDEN_CHANNELS),
            nn.Linear(config.HIDDEN_CHANNELS, config.TIME_DIM),
            nn.GELU(),
            nn.Linear(config.TIME_DIM, config.TIME_DIM),
        )
        self.initial_node_embed = nn.Embedding(100, config.HIDDEN_CHANNELS)

    def _initialize_variance_schedule(self):
        self.T = config.T
        self.beta = nn.Parameter(cosine_beta_schedule(self.T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_bar = nn.Parameter(torch.cumprod(self.alpha, 0), requires_grad=False)
        self.alpha_bar_prev = nn.Parameter(
            pad(torch.cumprod(self.alpha, 0)[:-1], (1, 0), value=1.0),
            requires_grad=False,
        )
        self.posterior_std = (
            self.beta * (1 - self.alpha_bar_prev) / (1 - self.alpha_bar)
        ).sqrt()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=config.LEARNING_RATE)

    def forward(self, perturbed_data: Data) -> Tensor:
        x = self.initial_node_embed(perturbed_data.z)
        t_emb = self.time_mlp(perturbed_data.t.float())
        _, eps_pred = self.egnn(
            x, perturbed_data.perturbed_pos, t_emb, perturbed_data.batch
        )
        return eps_pred

    def training_step(self, perturbed_data, *args, **kwargs):
        batch_size = perturbed_data.batch.max() + 1

        eps_pred = self.forward(perturbed_data)

        loss = scatter(
            (perturbed_data.eps - eps_pred).pow(2).sum(dim=1).sqrt(),
            perturbed_data.batch,
            0,
            reduce="sum",
        )

        N = nodes_per_graph(perturbed_data.batch)
        loss = loss / N
        loss = loss.mean()
        self.log("train_loss", loss, batch_size=batch_size, on_epoch=True)
        return loss

    def predict_mean_pos(self, data: Data) -> Data:
        eps_pred = self.forward(data)

        t_expanded_to_node = data.t[data.batch]
        alpha_t = self.alpha[t_expanded_to_node].unsqueeze(1)
        alpha_bar_t = self.alpha_bar[t_expanded_to_node].unsqueeze(1)
        beta_t = self.beta[t_expanded_to_node].unsqueeze(1)

        delta = data.perturbed_pos - (beta_t / (1 - alpha_bar_t).sqrt()) * eps_pred
        return (1 / alpha_t.sqrt()) * delta

    def _sample_step(self, data: Data) -> Data:
        data.perturbed_pos = centered_pos(data.perturbed_pos, data.batch)
        mean_pos = self.predict_mean_pos(data)
        noise = torch.randn_like(mean_pos)

        t_expanded = data.t[data.batch]
        sigma_t = self.posterior_std[t_expanded]
        new_pos = mean_pos + sigma_t.unsqueeze(1) * noise
        data.perturbed_pos = new_pos
        return data

    @torch.no_grad()
    def sample(self, perturbed_data: Data) -> Data:
        assert (perturbed_data.t == self.T - 1).all(), (self.T, perturbed_data.t)
        while (perturbed_data.t > 0).any():
            perturbed_data = self._sample_step(perturbed_data)
            perturbed_data.t = perturbed_data.t - 1
        return self.predict_mean_pos(perturbed_data)

    def validation_step(self, perturbed_data, data_idx, *args, **kwargs):
        if data_idx == 0:
            denoised_pos = self.sample(perturbed_data)
            batch_idx_zero = torch.where(perturbed_data.batch == 0)

            pos = denoised_pos[batch_idx_zero].detach().view(3, -1).numpy()
            color = perturbed_data.z[batch_idx_zero].detach().numpy()

            fig = plt.figure()
            ax = plot_point_cloud_3d(fig, 111, color, pos)
            img = wandb.Image(fig)
            wandb.log({"recon": img})
        return
