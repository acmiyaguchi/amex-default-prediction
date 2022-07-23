from pathlib import Path

import numpy as np
import pyarrow  # noqa: F401 pylint: disable=W0611
import pyarrow.dataset as ds
import pytorch_lightning as pl
import torch
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from torch.utils.data import DataLoader, IterableDataset

from amex_default_prediction.model.base import read_train_data


def get_parquet_feature_size(path, field="features"):
    """Get the size of the feature column in the parquet file."""
    files = sorted(Path(path).glob("**/*.parquet"))
    for batch in ds.dataset(files[0], format="parquet").to_batches(batch_size=1):
        df = batch.to_pandas()
        return len(df[field].iloc[0])


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


class ArrowDataset(IterableDataset):
    def __init__(self, path, filter=None):
        self.path = path
        self.filter = filter

    def __iter__(self):
        files = sorted(Path(self.path).glob("*.parquet"))
        if not files:
            raise ValueError("No parquet files found in {}".format(self.path))

        # https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        worker_info = torch.utils.data.get_worker_info()
        num_workers = 1 if worker_info is None else worker_info.num_workers
        worker_id = 0 if worker_info is None else worker_info.id

        # compute number of rows per worker
        rows_per_worker = int(np.ceil(len(files) / num_workers))
        start = worker_id * rows_per_worker
        end = start + rows_per_worker

        if not files[start:end]:
            # there is no work for this worker
            return
        dataset = ds.dataset(files[start:end], format="parquet")

        # https://arrow.apache.org/cookbook/py/io.html
        for batch in dataset.to_batches(filter=self.filter):
            df = batch.to_pandas()
            for item in df.itertuples():
                # this is not ideal because it doesn't take advantage of batching
                # achieves ~50it/s
                yield dict(features=torch.from_numpy(item.features), label=item.label)


class ArrowDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_preprocessed_path,
        train_ratio=0.8,
        batch_size=32,
        num_workers=8,
        **kwargs,
    ):
        super().__init__()
        self.train_data_preprocessed_path = train_data_preprocessed_path
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            **kwargs,
        )

    def setup(self, stage=None):
        # read the data so we can do stuff with it
        self.train_ds = ArrowDataset(
            self.train_data_preprocessed_path,
            filter=ds.field("sample_id") < self.train_ratio * 100,
        )
        self.val_ds = ArrowDataset(
            self.train_data_preprocessed_path,
            filter=ds.field("sample_id") >= self.train_ratio * 100,
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, **self.kwargs)

    def val_dataloader(self):
        return DataLoader(self.train_ds, **self.kwargs)
