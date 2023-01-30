from model import EDM, BATCH_SIZE, config
from data import make_data_module
from torch_geometric.datasets import QM9
import pytorch_lightning as pl
import wandb

if __name__ == "__main__":
    wandb.init(project="toy_diffusion", config=config)

    dataset = QM9("data")
    data_module = make_data_module(
        dataset,
        BATCH_SIZE,
        0,
    )
    model = EDM()
    trainer = pl.Trainer()

    trainer.fit(model, data_module)
