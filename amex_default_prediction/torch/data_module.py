from pathlib import Path

import numpy as np
import pyarrow  # noqa: F401 pylint: disable=W0611
import pyarrow.dataset as ds
import pytorch_lightning as pl
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from amex_default_prediction.model.base import read_train_data


def get_spark_feature_size(spark, path):
    df, _, _ = read_train_data(spark, path, cache=False)
    return df.head().features.shape[0]


def transform_vector_to_array(df, partitions=32):
    """Cast the features and labels fields from the v2 transformed dataset to
    align with the expectations of torch."""
    return (
        df.withColumn("features", vector_to_array("features").cast("array<float>"))
        .withColumn("label", F.col("label").cast("long"))
        .repartition(partitions)
    )


class PetastormDataModule(pl.LightningDataModule):
    def __init__(
        self,
        spark,
        cache_dir,
        train_data_preprocessed_path,
        train_ratio=0.8,
        batch_size=32,
        num_partitions=20,
    ):
        super().__init__()
        spark.conf.set(
            SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, Path(cache_dir).as_posix()
        )
        self.spark = spark
        self.train_data_preprocessed_path = train_data_preprocessed_path
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.num_partitions = num_partitions

    def setup(self, stage=None):
        # read the data so we can do stuff with it
        _, train_df, val_df = read_train_data(
            self.spark,
            Path(self.train_data_preprocessed_path).as_posix(),
            self.train_ratio,
        )

        self.input_size = val_df.head().features.size
        self.converter_train = make_spark_converter(
            transform_vector_to_array(train_df, self.num_partitions).select(
                "features", "label"
            )
        )
        self.converter_val = make_spark_converter(
            transform_vector_to_array(val_df, self.num_partitions).select(
                "features", "label"
            )
        )

    def train_dataloader(self):
        with self.converter_train.make_torch_dataloader(
            batch_size=self.batch_size, num_epochs=1
        ) as loader:
            for batch in loader:
                yield batch

    def val_dataloader(self):
        with self.converter_val.make_torch_dataloader(
            batch_size=self.batch_size, num_epochs=1
        ) as loader:
            for batch in loader:
                yield batch
