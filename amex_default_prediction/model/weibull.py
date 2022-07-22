from ast import In
from pathlib import Path

import click
import pyarrow  # noqa: F401 pylint: disable=W0611
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as sparkF
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from amex_default_prediction.utils import spark_session

from .base import read_train_data


class Net(pl.LightningModule):
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


class AmexDataModule(pl.LightningDataModule):
    def __init__(self, spark, cache_dir, train_data_preprocessed_path, train_ratio=0.8):
        super().__init__()
        spark.conf.set(
            SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, Path(cache_dir).as_posix()
        )
        self.spark = spark
        self.train_data_preprocessed_path = train_data_preprocessed_path
        self.train_ratio = train_ratio

    def setup(self, stage=None):
        # read the data so we can do stuff with it
        _, train_df, val_df = read_train_data(
            self.spark,
            Path(self.train_data_preprocessed_path).as_posix(),
            self.train_ratio,
        )

        def transform_df(df):
            F = sparkF
            return df.select(
                vector_to_array("features").cast("array<float>").alias("features"),
                F.col("label").cast("long"),
            ).repartition(32)

        self.input_size = val_df.head().features.size
        self.converter_train = make_spark_converter(transform_df(train_df))
        self.converter_val = make_spark_converter(transform_df(val_df))

    def train_dataloader(self):
        with self.converter_train.make_torch_dataloader() as loader:
            for batch in loader:
                yield batch

    def val_dataloader(self):
        with self.converter_val.make_torch_dataloader() as loader:
            for batch in loader:
                yield batch


@click.command()
@click.argument("train_data_preprocessed_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--train-ratio", default=0.8, type=float)
@click.option("--parallelism", default=1, type=int)
@click.option("--cache-dir", default="file:///tmp")
def fit(train_data_preprocessed_path, output_path, train_ratio, parallelism, cache_dir):
    spark = spark_session()

    # get the input size for the model
    df, _, _ = read_train_data(spark, train_data_preprocessed_path, cache=False)
    input_size = df.head().features.size
    model = Net(input_size=input_size)
    print(model)

    data_module = AmexDataModule(
        spark, cache_dir, train_data_preprocessed_path, train_ratio
    )

    trainer = pl.Trainer(
        gpus=-1,
        default_root_dir=output_path,
        detect_anomaly=True,
        logger=TensorBoardLogger(output_path, log_graph=True),
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min"),
            ModelCheckpoint(monitor="val_loss", auto_insert_metric_name=True),
        ],
    )
    trainer.fit(model, data_module)
