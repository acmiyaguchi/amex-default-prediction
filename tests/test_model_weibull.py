import pytorch_lightning as pl
import torch

from amex_default_prediction.model.weibull import AmexDataModule, Net


def test_data_module_has_fields(spark, synthetic_train_data_path):
    data_module = AmexDataModule(spark, "file:///tmp", synthetic_train_data_path)
    data_module.setup()
    dataloader = data_module.train_dataloader()
    for batch in dataloader:
        assert set(batch.keys()) == {"features", "label"}
        assert isinstance(batch["features"], torch.Tensor)
        assert isinstance(batch["label"], torch.Tensor)
        break


def test_trainer_accepts_data_module(spark, synthetic_train_data_path):
    data_module = AmexDataModule(spark, "file:///tmp", synthetic_train_data_path)
    trainer = pl.Trainer(fast_dev_run=True)
    model = Net(input_size=3)
    trainer.fit(model, datamodule=data_module)
