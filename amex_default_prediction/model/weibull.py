import click
import pyarrow  # noqa: F401 pylint: disable=W0611
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pyspark.ml.functions import vector_to_array

from amex_default_prediction.utils import spark_session

from .base import read_train_data


class Net(pl.Module):
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

    def configure_optimizer(self):
        return torch.optim.Adam(self.parameters, lr=1e-3)

    def _step(self, batch, *args, **kwargs):
        x, y = batch["features"], batch["label"]
        z = self(x)
        return F.cross_entropy(z, y)

    def training_step(self, train_batch, batch_idx):
        self._step(train_batch)

    def validation_step(self, val_batch, batch_id):
        self._step(val_batch)


@click.command()
@click.argument("train_data_preprocessed_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--train-ratio", default=0.8, type=float)
@click.option("--parallelism", default=1, type=int)
@click.option("--cache-dir", default="data/tmp/spark")
def fit(train_data_preprocessed_path, output_path, train_ratio, parallelism, cache_dir):
    spark = spark_session()
    spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, cache_dir)
    # read the data so we can do stuff with it
    df, train_df, val_df = read_train_data(
        spark,
        train_data_preprocessed_path,
        train_ratio,
        train_most_recent_only=False,
        validation_most_recent_only=False,
    )

    def transform_df(df):
        return df.select(vector_to_array("features").alias("features"), "label")

    input_size = val_df.head().features.size
    print(input_size)

    converter_train = make_spark_converter(transform_df(train_df))
    converter_val = make_spark_converter(transform_df(val_df))
