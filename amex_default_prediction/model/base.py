from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.feature import SQLTransformer
from pyspark.ml.functions import vector_to_array
from pyspark.ml.param.shared import (
    HasInputCol,
    HasOutputCol,
    Param,
    Params,
    TypeConverters,
)
from pyspark.ml.pipeline import Transformer
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import functions as F
from pyspark.sql import types as T

from amex_default_prediction.evaluation import AmexMetricEvaluator


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


def read_train_data(spark, train_data_preprocessed_path, train_ratio):
    data = (
        spark.read.parquet((Path(train_data_preprocessed_path) / "data").as_posix())
        .withColumn("label", F.col("target").cast("float"))
        .withColumn("sample_id", F.crc32("customer_ID") % 100)
        .cache()
    )
    train_data = data.where(f"sample_id < {train_ratio*100}")

    # some debugging information
    validation_data = data.where(f"sample_id >= {train_ratio*100}")
    train_count = train_data.select(
        F.count("*").alias("total"), F.sum("label").alias("positive")
    ).collect()[0]
    validation_count = validation_data.select(
        F.count("*").alias("total"), F.sum("label").alias("positive")
    ).collect()[0]

    print(f"training ratio: {train_ratio}")
    print(f"train_count: {train_count.total}, positive: {train_count.positive}")
    print(
        f"validation_count: {validation_count.total}, positive: {validation_count.positive}"
    )

    return data, train_data, validation_data


def fit_generic(
    spark,
    pipeline,
    grid,
    evaluator,
    train_data_preprocessed_path,
    output_path,
    train_ratio=0.8,
    parallelism=4,
):
    """Fit function that can be used with a variety of models for quick
    iteration. Assumes that preprocessing has already been done."""
    data, train_data, validation_data = read_train_data(
        spark, train_data_preprocessed_path, train_ratio
    )
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        parallelism=parallelism,
    )
    cv_model = cv.fit(train_data)

    train_eval = evaluator.evaluate(cv_model.transform(train_data))
    validation_eval = evaluator.evaluate(cv_model.transform(validation_data))
    total_eval = evaluator.evaluate(cv_model.transform(data))
    print(f"train eval: {train_eval}")
    print(f"validation eval: {validation_eval}")
    print(f"total eval: {total_eval}")

    cv_model.write().overwrite().save(Path(output_path).as_posix())
    print(f"wrote to {output_path}")


def fit_simple(
    spark,
    model,
    grid,
    train_data_preprocessed_path,
    output_path,
    train_ratio=0.8,
    parallelism=4,
):
    """Fit function that handles binary prediction using the Amex metric"""
    fit_generic(
        spark,
        Pipeline(
            stages=[
                model,
                ExtractVectorIndexTransformer(
                    inputCol="probability", outputCol="pred", indexCol=1
                ),
            ]
        ),
        grid,
        AmexMetricEvaluator(predictionCol="pred", labelCol="label"),
        train_data_preprocessed_path,
        output_path,
        train_ratio,
        parallelism,
    )
