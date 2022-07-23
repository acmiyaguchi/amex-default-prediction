import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class StrawmanNet(pl.LightningModule):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax(dim=1),
        )

    # pylint: disable=arguments-differ
    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def _step(self, batch, *args, **kwargs):
        x, y = batch["features"], batch["label"]
        z = self(x)
        return F.cross_entropy(z, y)

    def training_step(self, train_batch, batch_idx):
        loss = self._step(train_batch)
        self.log("loss", loss)
        return loss

    def validation_step(self, val_batch, batch_id):
        loss = self._step(val_batch)
        self.log("val_loss", loss)
        return loss
