from model import EDM, BATCH_SIZE, config
from data import make_data_module
from torch_geometric.datasets import QM9
from pytorch_lightning.loggers.wandb import WandbLogger
import pytorch_lightning as pl
import wandb

if __name__ == "__main__":
    wandb.init(project="toy_diffusion", config=config)

    logger = WandbLogger()

    dataset = QM9("data")
    data_module = make_data_module(
        dataset,
        BATCH_SIZE,
        32
    )

    model = EDM()
    trainer = pl.Trainer(
        logger=logger,
        auto_select_gpus=True,
        max_epochs=300,
        accelerator="gpu",
    )

    trainer.fit(model, data_module)
