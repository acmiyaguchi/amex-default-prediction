import functools
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow  # noqa: F401 pylint: disable=W0611
import pyarrow.dataset as ds
import pytorch_lightning as pl
from petastorm.codecs import CompressedImageCodec, NdarrayCodec, ScalarCodec
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import Window
from pyspark.sql import functions as F

from amex_default_prediction.model.base import read_train_data


def get_spark_feature_size(spark, path, pca_model_path=None):
    df, _, _ = read_train_data(spark, path, cache=False)
    if pca_model_path:
        pca_model = PipelineModel.read().load(pca_model_path)
        df = pca_model.transform(df).withColumn("features", F.col("features_pca"))
    return len(
        df.select(vector_to_array("features").cast("array<float>").alias("features"))
        .head()
        .features
    )


def transform_vector_to_array(df, partitions=32):
    """Cast the features and labels fields from the v2 transformed dataset to
    align with the expectations of torch."""
    return (
        df.withColumn("features", vector_to_array("features").cast("array<float>"))
        .withColumn("label", F.col("label").cast("long"))
        .repartition(partitions)
    )


def pad_src(field, pad, length):
    return F.concat(
        F.array_repeat(pad, F.lit(length) - F.size(field)),
        F.col(field),
    )


def pad_tgt(field, pad, length):
    return F.concat(
        F.col(field),
        F.array_repeat(pad, F.lit(length) - F.size(field)),
    )


def features_age_list(df):
    w = Window.partitionBy("customer_ID").orderBy("age_days")
    return (
        df.withColumn("features", vector_to_array("features").cast("array<float>"))
        .select(
            "customer_ID",
            F.collect_list("features").over(w).alias("features_list"),
            # age encoded into a sequence position
            F.collect_list((F.col("age_days") + 1).astype("long"))
            .over(w)
            .alias("age_days_list"),
        )
        .groupBy("customer_ID")
        .agg(
            F.max("features_list").alias("features_list"),
            F.max("age_days_list").alias("age_days_list"),
        )
        .withColumn("n", F.size("features_list"))
        .withColumn("dim", F.size(F.col("features_list")[0]))
    )


def transform_into_transformer_pairs(df, length=4):
    """Convert the training/test dataset for use in a transformer."""

    def slice_src(field, length):
        return F.when(
            F.col("n") <= length,
            F.slice(F.col(field), 1, F.col("n") - 1),
        ).otherwise(
            F.when(
                F.col("n") <= 2 * length,
                F.slice(F.col(field), 1, length),
            ).otherwise(F.slice(F.col(field), -(2 * length) + 1, length))
        )

    def slice_tgt(field, length):
        return F.when(F.col("n") <= length, F.slice(F.col(field), -1, 1)).otherwise(
            F.when(
                F.col("n") <= 2 * length, F.slice(F.col(field), length, length)
            ).otherwise(F.slice(F.col(field), -length, length))
        )

    return (
        features_age_list(df)
        .where("n > 1")
        # this is not pleasant to read, but at least it doesn't require a UDF...
        .withColumn("src", slice_src("features_list", length))
        .withColumn("tgt", slice_tgt("features_list", length))
        .withColumn("src_pos", slice_src("age_days_list", length))
        .withColumn("tgt_pos", slice_tgt("age_days_list", length))
        # create padding mask before we actually pad src/tgt
        .withColumn("k_src", F.size("src"))
        .withColumn("k_tgt", F.size("tgt"))
        # pad src and tgt with arrays filled with zeroes
        .withColumn(
            "src", pad_src("src", F.array_repeat(F.lit(0.0), F.col("dim")), length)
        )
        .withColumn(
            "tgt", pad_tgt("tgt", F.array_repeat(F.lit(0.0), F.col("dim")), length)
        )
        .withColumn("src_pos", pad_src("src_pos", F.lit(0), length))
        .withColumn("tgt_pos", pad_tgt("tgt_pos", F.lit(0), length))
        .withColumn(
            "src_key_padding_mask",
            F.concat(
                F.array_repeat(F.lit(1), F.lit(length) - F.col("k_src")),
                F.array_repeat(F.lit(0), F.col("k_src")),
            ),
        )
        .withColumn(
            "tgt_key_padding_mask",
            F.concat(
                F.array_repeat(F.lit(0), F.col("k_tgt")),
                F.array_repeat(F.lit(1), F.lit(length) - F.col("k_tgt")),
            ),
        )
        # now lets flatten the src and tgt rows
        .withColumn("src", F.flatten("src"))
        .withColumn("tgt", F.flatten("tgt"))
        .select(
            "customer_ID",
            "src",
            "tgt",
            "src_key_padding_mask",
            "tgt_key_padding_mask",
            "src_pos",
            "tgt_pos",
        )
    )


def transform_into_transformer_predict_pairs(df, length=4):
    def slice_src(field, length):
        return F.when(F.col("n") <= length, F.col(field)).otherwise(
            F.slice(F.col(field), -length, length)
        )

    return (
        features_age_list(df)
        .where("n >= 1")
        .withColumn("src", slice_src("features_list", length))
        .withColumn("src_pos", slice_src("age_days_list", length))
        # measure the length before creating a mask and padding
        .withColumn("k_src", F.size("src"))
        .withColumn(
            "src", pad_src("src", F.array_repeat(F.lit(0.0), F.col("dim")), length)
        )
        .withColumn("src_pos", pad_src("src_pos", F.lit(0), length))
        .withColumn(
            "src_key_padding_mask",
            F.concat(
                F.array_repeat(F.lit(1), F.lit(length) - F.col("k_src")),
                F.array_repeat(F.lit(0), F.col("k_src")),
            ),
        )
        .withColumn("src", F.flatten("src"))
        .select(
            "customer_ID",
            "src",
            "src_key_padding_mask",
            "src_pos",
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
        num_partitions=32,
        workers_count=16,
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
        self.workers_count = workers_count

    def setup(self, stage=None):
        # read the data so we can do stuff with it
        df, train_df, val_df = read_train_data(
            self.spark,
            Path(self.train_data_preprocessed_path).as_posix(),
            self.train_ratio,
        )

        def make_converter(df):
            return make_spark_converter(
                transform_vector_to_array(df, self.num_partitions).select(
                    "features", "label"
                )
            )

        self.converter_train = make_converter(train_df)
        self.converter_val = make_converter(val_df)

    def _dataloader(self, converter):
        with converter.make_torch_dataloader(
            batch_size=self.batch_size,
            num_epochs=1,
            workers_count=self.workers_count,
        ) as loader:
            for batch in loader:
                yield batch

    def train_dataloader(self):
        for batch in self._dataloader(self.converter_train):
            yield batch

    def val_dataloader(self):
        for batch in self._dataloader(self.converter_val):
            yield batch

    def predict_dataloader(self):
        if self.converter_predict:
            for batch in self._dataloader(self.converter_predict):
                yield batch
        else:
            raise Exception("No converter for predict")


class PetastormTransformerDataModule(PetastormDataModule):
    def __init__(
        self,
        spark,
        cache_dir,
        train_data_preprocessed_path,
        pca_model_path=None,
        subsequence_length=8,
        **kwargs
    ):
        super().__init__(spark, cache_dir, train_data_preprocessed_path, **kwargs)
        self.pca_model_path = pca_model_path
        self.subsequence_length = subsequence_length

    def setup(self, stage=None):
        # read the data so we can do stuff with it
        full_df, train_df, val_df = read_train_data(
            self.spark,
            Path(self.train_data_preprocessed_path).as_posix(),
            self.train_ratio,
            data_most_recent_only=False,
            train_most_recent_only=False,
            validation_most_recent_only=False,
            cache=False,
        )

        if self.pca_model_path:
            pca_model = PipelineModel.read().load(self.pca_model_path)
            train_df, val_df, full_df = [
                pca_model.transform(df).withColumn("features", F.col("features_pca"))
                for df in [train_df, val_df, full_df]
            ]

        def make_train_converter(df):
            return make_spark_converter(
                transform_into_transformer_pairs(
                    df,
                    self.subsequence_length,
                )
                .select(
                    "src",
                    "tgt",
                    "src_key_padding_mask",
                    "tgt_key_padding_mask",
                    "src_pos",
                    "tgt_pos",
                )
                .repartition(self.num_partitions)
            )

        self.converter_train = make_train_converter(train_df)
        self.converter_val = make_train_converter(val_df)
        self.converter_predict = make_spark_converter(
            transform_into_transformer_predict_pairs(full_df, self.subsequence_length)
            .select(
                "src",
                "src_key_padding_mask",
                "src_pos",
                F.row_number()
                .over(Window.orderBy("customer_ID"))
                .alias("customer_index"),
            )
            .repartition(self.num_partitions)
        )
