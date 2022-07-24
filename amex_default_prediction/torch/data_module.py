import functools
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow  # noqa: F401 pylint: disable=W0611
import pyarrow.dataset as ds
import pytorch_lightning as pl
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pyspark.ml.functions import array_to_vector, vector_to_array
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


def create_transformer_pair(pdf: pd.DataFrame, length: int) -> pd.DataFrame:
    """Utility pandas udf for creating transformer pairs."""
    # fmt: off
    # import multiprocessing
    # f = open('fault_%s.log' % multiprocessing.current_process().name, 'w')
    # import faulthandler; faulthandler.enable(file=f, all_threads=True)
    # fmt: on
    features = pdf.sort_values("age_days").features.values

    # to create this data, we want to ensure that we try to fill up the src
    # sequence before we fill up the tgt sequence. We pad the src on the left,
    # while padding the tgt on the right.
    if len(features) <= length:
        src = features[-(length + 1) : -1]
        tgt = features[-1:]
    elif len(features) <= 2 * length:
        # we have enough to fill up the src but not tgt
        src = features[:length]
        tgt = features[length:]
    else:
        # we have more data than we need, lets fill up the dst
        src = features[-(2 * length) : -length]
        tgt = features[-length:]

    k = len(src)
    src_key_padding_mask = [False] * (length - k) + [True] * k
    k = len(tgt)
    tgt_key_padding_mask = [True] * k + [False] * (length - k)

    res = pd.DataFrame(
        [
            dict(
                customer_ID=pdf.customer_ID.iloc[0],
                src=np.stack(src),
                tgt=np.stack(tgt),
                src_key_padding_mask=np.array(src_key_padding_mask),
                tgt_key_padding_mask=np.array(tgt_key_padding_mask),
            )
        ],
    )
    res.info(verbose=True)
    print(res.to_dict("records"), flush=True)
    return res


def transform_into_transformer_pairs(df, length=4):
    """Convert the training/test dataset for use in a transformer."""
    return (
        df.withColumn("features", vector_to_array("features"))
        .select("customer_ID", "features", "age_days")
        .groupBy("customer_ID")
        .applyInPandas(
            functools.partial(create_transformer_pair, length=length),
            schema=",".join(
                [
                    "customer_ID string",
                    "src array<array<double>>",
                    "tgt array<array<double>>",
                    "src_key_padding_mask array<boolean>",
                    "tgt_key_padding_mask array<boolean>",
                ]
            ),
        )
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
