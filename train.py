from torch_geometric.datasets import QM9
from pytorch_lightning.loggers.wandb import WandbLogger
import pytorch_lightning as pl
import wandb

import config
from model import EDM
from data import make_data_module
from transform import PerturbNodePositions

if __name__ == "__main__":
    wandb.init(project="toy_diffusion")

    logger = WandbLogger()

    model = EDM()
    train_perturbation = PerturbNodePositions(alpha_bar=model.alpha_bar)
    val_peturbation = PerturbNodePositions(alpha_bar=model.alpha_bar, t=config.T)
    dataset = QM9("data")
    data_module = make_data_module(
        dataset,
        config.BATCH_SIZE,
        0,
        train_size=0.9,
        val_size=0.1,
        train_transform=train_perturbation,
        val_transform=val_peturbation,
    )

    trainer = pl.Trainer(
        logger=logger,
        auto_select_gpus=True,
        max_epochs=300,
        accelerator="cpu",
    )

    trainer.fit(model, data_module)
