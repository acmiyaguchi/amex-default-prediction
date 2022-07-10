from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.feature import SQLTransformer
from pyspark.ml.functions import vector_to_array
from pyspark.ml.pipeline import Transformer
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import functions as F
from pyspark.sql import types as T

from amex_default_prediction.evaluation import AmexMetricEvaluator


class PredictionTransformer(Transformer, DefaultParamsWritable, DefaultParamsReadable):
    """https://stackoverflow.com/a/52501479

    Also see evaluation.AmexMetricEvaluator for details on how to persist a
    custom pipeline element. Old solutions relied on a normal udf, this is
    probably more performant.
    - https://stackoverflow.com/a/44505571
    - https://stackoverflow.com/a/44064252
    """

    def __init__(self, inputCol="probability", outputCol="pred"):
        self.inputCol = inputCol
        self.outputCol = outputCol
        # for the default params writable
        self.uid = self.__class__.__name__
        self._paramMap = {}
        self._defaultParamMap = {}
        # for the default params readable
        self._params = {}

    def _transform(self, dataset):
        return dataset.withColumn(
            self.outputCol, vector_to_array(self.inputCol)[1].cast("float")
        )


def fit_simple(
    spark,
    model,
    grid,
    train_data_preprocessed_path,
    output_path,
    train_ratio=0.8,
    parallelism=4,
):
    """Fit function that can be used with a variety of models for quick
    iteration. Assumes that preprocessing has already been done."""
    data = (
        spark.read.parquet((Path(train_data_preprocessed_path) / "data").as_posix())
        .withColumn("label", F.col("target").cast("float"))
        .withColumn("sample_id", F.crc32("customer_ID") % 100)
        .cache()
    )
    train_data = data.where(f"sample_id < {train_ratio*100}")
    validation_data = data.where(f"sample_id >= {train_ratio*100}")
    train_count = train_data.count()
    validation_count = validation_data.count()

    print(f"training ratio: {train_ratio}")
    print(f"train_count: {train_count}")
    print(f"validation_count: {validation_count}")

    # we also extract out the first column of the probability
    pipeline = Pipeline(
        stages=[model, PredictionTransformer(inputCol="probability", outputCol="pred")]
    )
    evaluator = AmexMetricEvaluator(predictionCol="pred", labelCol="label")
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
