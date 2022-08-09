from pathlib import Path

import pyarrow  # noqa: F401 pylint: disable=W0611
import pytorch_lightning as pl
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pyspark.ml.functions import vector_to_array
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import Window
from pyspark.sql import functions as F

from amex_default_prediction.model.base import read_train_data

from .transform import (
    transform_into_transformer_pairs,
    transform_into_transformer_predict_pairs,
    transform_vector_to_array,
)


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
