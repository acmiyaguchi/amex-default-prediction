from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
import torch
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import SQLTransformer, VectorAssembler
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.ml.param.shared import (
    HasInputCol,
    HasNumFeatures,
    HasOutputCol,
    Param,
    Params,
    TypeConverters,
)
from pyspark.ml.pipeline import Transformer
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import Window
from pyspark.sql import functions as F

from amex_default_prediction.evaluation import AmexMetricEvaluator
from amex_default_prediction.torch.net import TransformerModel
from amex_default_prediction.torch.transform import (
    transform_into_transformer_predict_pairs,
)

from ..utils import spark_session


class HasIndexCol(Params):
    """
    Mixin for param indexCol: index column name.
    """

    indexCol: "Param[int]" = Param(
        Params._dummy(),
        "indexCol",
        "index column name.",
        typeConverter=TypeConverters.toInt,
    )

    def __init__(self) -> None:
        super().__init__()

    def getIndexCol(self) -> int:
        """
        Gets the value of indexCol or its default value.
        """
        return self.getOrDefault(self.indexCol)


class HasCheckpointPath(Params):
    checkpointPath: "Param[str]" = Param(
        Params._dummy(),
        "checkpointPath",
        "checkpoint path",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self) -> None:
        super().__init__()

    def getCheckpointPath(self) -> str:
        return self.getOrDefault(self.checkpointPath)


class HasSequenceLength(Params):
    sequenceLength: "Param[int]" = Param(
        Params._dummy(),
        "sequenceLength",
        "sequence length",
        typeConverter=TypeConverters.toInt,
    )

    def __init__(self) -> None:
        super().__init__()

    def getSequenceLength(self) -> int:
        return self.getOrDefault(self.sequenceLength)


class ExtractVectorIndexTransformer(
    Transformer,
    HasInputCol,
    HasOutputCol,
    HasIndexCol,
    DefaultParamsWritable,
    DefaultParamsReadable,
):
    """https://stackoverflow.com/a/52501479

    Also see evaluation.AmexMetricEvaluator for details on how to persist a
    custom pipeline element. Old solutions relied on a normal udf, this is
    probably more performant.
    - https://stackoverflow.com/a/44505571
    - https://stackoverflow.com/a/44064252
    - https://csyhuang.github.io/2020/08/01/custom-transformer/
    - https://github.com/apache/spark/blob/master/python/pyspark/ml/param/shared.py
    """

    def __init__(self, inputCol="probability", outputCol="pred", indexCol=1):
        super().__init__()
        self._setDefault(inputCol=inputCol, outputCol=outputCol, indexCol=indexCol)

    def _transform(self, dataset):
        return dataset.withColumn(
            self.getOutputCol(),
            vector_to_array(self.getInputCol())[self.getIndexCol()].cast("float"),
        )


class LogFeatureTransformer(
    Transformer,
    HasInputCol,
    HasOutputCol,
    DefaultParamsWritable,
    DefaultParamsReadable,
):
    def __init__(self, inputCol="features", outputCol="features_log"):
        super().__init__()
        self._setDefault(inputCol=inputCol, outputCol=outputCol)

    def _transform(self, dataset):
        @F.pandas_udf("array<double>")
        def log_features(features):
            return features.apply(lambda x: np.log(x + 1.0))

        return dataset.withColumn(
            self.getOutputCol(),
            array_to_vector(log_features(vector_to_array(self.getInputCol()))),
        )


class TransformerInferenceTransformer(
    Transformer,
    HasInputCol,
    HasCheckpointPath,
    HasNumFeatures,
    HasSequenceLength,
    HasOutputCol,
    DefaultParamsWritable,
    DefaultParamsReadable,
):
    def __init__(
        self,
        inputCol="features",
        checkpointPath="checkpoint",
        sequenceLength=8,
        numFeatures=128,
        outputCol="prediction",
    ):
        super().__init__()
        self._setDefault(
            inputCol=inputCol,
            checkpointPath=checkpointPath,
            sequenceLength=sequenceLength,
            numFeatures=numFeatures,
            outputCol=outputCol,
        )

    def _transform(self, dataset):
        """https://docs.databricks.com/_static/notebooks/deep-learning/pytorch-images.html"""

        checkpoint = self.getCheckpointPath()
        num_features = self.getNumFeatures()

        @F.pandas_udf("array<float>")
        def predict_batch_udf(
            it: Iterator[Tuple[pd.Series, pd.Series, pd.Series]]
        ) -> Iterator[pd.Series]:
            model = TransformerModel.load_from_checkpoint(
                checkpoint, d_model=num_features
            )
            for src, src_key_padding_mask, src_pos in it:
                value = {
                    "src": torch.from_numpy(np.stack(src.values)).float(),
                    "src_key_padding_mask": (
                        torch.from_numpy(np.stack(src_key_padding_mask.values))
                    ),
                    "src_pos": torch.from_numpy(np.stack(src_pos.values)),
                }
                try:
                    z = model.predict_step(value, 0).cpu().detach().numpy()
                except Exception as e:
                    print(e, flush=True)
                    raise e
                res = pd.Series(list(z[0]))
                yield res

        transformed = transform_into_transformer_predict_pairs(
            dataset.select("customer_ID", "age_days", "features"),
            length=self.getSequenceLength(),
        ).select(
            "customer_ID",
            predict_batch_udf(
                F.col("src"),
                F.col("src_key_padding_mask"),
                F.col("src_pos"),
            ).alias(self.getOutputCol()),
        )

        return dataset.join(transformed, on="customer_ID")


def read_train_data(
    spark,
    train_data_preprocessed_path,
    train_ratio=0.8,
    cache=True,
    data_most_recent_only=True,
    train_most_recent_only=True,
    validation_most_recent_only=True,
):
    data = spark.read.parquet((Path(train_data_preprocessed_path) / "data").as_posix())
    if cache:
        data = data.cache()

    train_data = data.where(f"sample_id < {train_ratio*100}")
    validation_data = data.where(f"sample_id >= {train_ratio*100}")
    if train_most_recent_only:
        train_data = train_data.where("most_recent")
    if validation_most_recent_only:
        validation_data = validation_data.where("most_recent")
    if data_most_recent_only:
        data = data.where("most_recent")

    # debugging information
    if "label" in data.columns:
        train_count = train_data.select(
            F.count("*").alias("total"), F.sum("label").alias("positive")
        ).collect()[0]
        validation_count = validation_data.select(
            F.count("*").alias("total"), F.sum("label").alias("positive")
        ).collect()[0]

        print(f"training ratio: {train_ratio}")
        print(f"train_count: {train_count.total}, positive: {train_count.positive}")
        print(
            f"validation_count: {validation_count.total}, "
            f"positive: {validation_count.positive}"
        )
    else:
        print("total count: ", data.count())

    return data, train_data, validation_data


def fit_generic(
    spark,
    model,
    evaluator,
    train_data_preprocessed_path,
    output_path,
    read_func=read_train_data,
    train_ratio=0.8,
    data_most_recent_only=True,
    train_most_recent_only=True,
    validation_most_recent_only=True,
):
    """Fit function that can be used with a variety of models for quick
    iteration. Assumes that preprocessing has already been done.

    :param spark: SparkSession
    :param model: Estimator
    :param evaluator: Evaluator
    :param train_data_preprocessed_path: Path to preprocessed training data
    :param output_path: Path to save model
    :param train_ratio: Ratio of training data to use
    :param read_func: Function to read data (path, ratio) -> (data, train, validation)
    :param train_most_recent_only: If True, only use the most recent training data
    :param validation_most_recent_only: If True, only use the most recent validation data
    """
    data, train_data, validation_data = read_func(
        spark,
        train_data_preprocessed_path,
        train_ratio,
        data_most_recent_only,
        train_most_recent_only,
        validation_most_recent_only,
    )

    fit_model = model.fit(train_data)

    if evaluator:
        train_eval = evaluator.evaluate(fit_model.transform(train_data))
        validation_eval = evaluator.evaluate(fit_model.transform(validation_data))
        total_eval = evaluator.evaluate(fit_model.transform(data))
        print(f"train eval: {train_eval}")
        print(f"validation eval: {validation_eval}")
        print(f"total eval: {total_eval}")

    fit_model.write().overwrite().save(Path(output_path).as_posix())
    print(f"wrote to {output_path}")


def fit_simple(
    spark,
    model,
    grid,
    train_data_preprocessed_path,
    output_path,
    parallelism=4,
    **kwargs,
):
    """Fit function that handles binary prediction using the Amex metric"""
    evaluator = AmexMetricEvaluator(predictionCol="pred", labelCol="label")
    fit_generic(
        spark,
        CrossValidator(
            estimator=Pipeline(
                stages=[
                    model,
                    ExtractVectorIndexTransformer(
                        inputCol="probability", outputCol="pred", indexCol=1
                    ),
                ]
            ),
            estimatorParamMaps=grid,
            evaluator=evaluator,
            parallelism=parallelism,
        ),
        evaluator,
        train_data_preprocessed_path,
        output_path,
        **kwargs,
    )


def fit_simple_with_aft(
    spark,
    model,
    grid,
    train_data_preprocessed_path,
    aft_model_path,
    output_path,
    parallelism=4,
    **kwargs,
):
    aft_model = CrossValidatorModel.read().load(aft_model_path)
    evaluator = AmexMetricEvaluator(predictionCol="pred", labelCol="label")
    fit_generic(
        spark,
        CrossValidator(
            estimator=Pipeline(
                stages=[
                    aft_model.bestModel,
                    VectorAssembler(
                        inputCols=["features", "quantiles_probability"],
                        outputCol="features_with_aft",
                    ),
                    # lets keep a subset of fields
                    SQLTransformer(
                        statement="""
                        SELECT
                            customer_ID,
                            features_with_aft as features,
                            label
                        FROM __THIS__
                    """
                    ),
                    model,
                    ExtractVectorIndexTransformer(
                        inputCol="probability", outputCol="pred", indexCol=1
                    ),
                ]
            ),
            estimatorParamMaps=grid,
            evaluator=evaluator,
            parallelism=parallelism,
        ),
        evaluator,
        train_data_preprocessed_path,
        output_path,
        **kwargs,
    )


def fit_simple_with_pca(
    spark,
    model,
    grid,
    train_data_preprocessed_path,
    pca_model_path,
    output_path,
    parallelism=4,
    **kwargs,
):
    pca_model = PipelineModel.read().load(pca_model_path)
    evaluator = AmexMetricEvaluator(predictionCol="pred", labelCol="label")
    fit_generic(
        spark,
        CrossValidator(
            estimator=Pipeline(
                stages=[
                    pca_model,
                    SQLTransformer(
                        statement="""
                        SELECT
                            customer_ID,
                            features_pca as features,
                            label
                        FROM __THIS__
                    """
                    ),
                    model,
                    ExtractVectorIndexTransformer(
                        inputCol="probability", outputCol="pred", indexCol=1
                    ),
                ]
            ),
            estimatorParamMaps=grid,
            evaluator=evaluator,
            parallelism=parallelism,
        ),
        evaluator,
        train_data_preprocessed_path,
        output_path,
        **kwargs,
    )


def fit_simple_with_transformer(
    spark,
    model,
    grid,
    train_data_preprocessed_path,
    train_transformer_path,
    output_path,
    parallelism=4,
    truncate_predict=None,
    transformer_only=False,
    **kwargs,
):
    def read_func(spark, train_data_preprocessed_path, train_ratio, *args, **kwargs):
        transformer_df = spark.read.parquet(train_transformer_path)
        df, _, _ = read_train_data(
            spark, train_data_preprocessed_path, train_ratio, *args, **kwargs
        )
        df = (
            df.withColumn("customer_index", F.hash("customer_ID"))
            .join(transformer_df, on="customer_index", how="inner")
            .withColumn(
                "transformer_feature",
                array_to_vector(
                    F.co("prediction")
                    if truncate_predict is None
                    else F.slice("prediction", 1, truncate_predict)
                ),
            )
            .repartition(spark.sparkContext.defaultParallelism * 2)
            .cache()
        )
        return (
            df,
            df.where(f"sample_id < {train_ratio*100}"),
            df.where(f"sample_id >= {train_ratio*100}"),
        )

    evaluator = AmexMetricEvaluator(predictionCol="pred", labelCol="label")
    fit_generic(
        spark,
        CrossValidator(
            estimator=Pipeline(
                stages=[
                    # vector assembler
                    *(
                        [
                            VectorAssembler(
                                inputCols=["features", "transformer_feature"],
                                outputCol="features_with_transformer",
                            )
                        ]
                        if not transformer_only
                        else []
                    ),
                    SQLTransformer(
                        statement=f"""
                        SELECT
                            customer_ID,
                            {
                                "features_with_transformer"
                                if not transformer_only
                                else "features"
                            } as feature,
                            label
                        FROM __THIS__
                    """
                    ),
                    model,
                    ExtractVectorIndexTransformer(
                        inputCol="probability", outputCol="pred", indexCol=1
                    ),
                ]
            ),
            estimatorParamMaps=grid,
            evaluator=evaluator,
            parallelism=parallelism,
        ),
        evaluator,
        train_data_preprocessed_path,
        output_path,
        read_func=read_func,
        **kwargs,
    )
