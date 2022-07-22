import pytorch_lightning as pl
import torch

from amex_default_prediction.torch.data_module import (
    ArrowDataModule,
    PetastormDataModule,
    get_parquet_feature_size,
)
from amex_default_prediction.torch.net import StrawmanNet


def test_get_parquet_feature_size(synthetic_train_data_torch_path):
    feature_size = get_parquet_feature_size(synthetic_train_data_torch_path)
    assert feature_size == 3


def test_petastorm_data_module_has_fields(spark, synthetic_train_data_path):
    data_module = PetastormDataModule(spark, "file:///tmp", synthetic_train_data_path)
    data_module.setup()
    dataloader = data_module.train_dataloader()
    for batch in dataloader:
        assert set(batch.keys()) == {"features", "label"}
        assert isinstance(batch["features"], torch.Tensor)
        assert isinstance(batch["label"], torch.Tensor)
        assert batch["features"].shape == torch.Size([32, 3])
        assert batch["label"].shape == torch.Size([32])
        break


def test_arrow_data_module_has_fields(synthetic_train_data_torch_path):
    data_module = ArrowDataModule(synthetic_train_data_torch_path)
    data_module.setup()
    dataloader = data_module.train_dataloader()
    for batch in dataloader:
        assert set(batch.keys()) == {"features", "label"}
        assert isinstance(batch["features"], torch.Tensor)
        assert isinstance(batch["label"], torch.Tensor)
        assert batch["features"].shape == torch.Size([32, 3])
        assert batch["label"].shape == torch.Size([32])
        break


def test_trainer_accepts_petastorm_data_module(spark, synthetic_train_data_path):
    data_module = PetastormDataModule(spark, "file:///tmp", synthetic_train_data_path)
    trainer = pl.Trainer(fast_dev_run=True)
    model = StrawmanNet(input_size=3)
    trainer.fit(model, datamodule=data_module)


def test_trainer_accepts_arrow_data_module(synthetic_train_data_torch_path):
    data_module = ArrowDataModule(synthetic_train_data_torch_path)
    trainer = pl.Trainer(fast_dev_run=True)
    model = StrawmanNet(input_size=3)
    trainer.fit(model, datamodule=data_module)
