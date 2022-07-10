from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.feature import SQLTransformer
from pyspark.ml.tuning import CrossValidator
from pyspark.sql import functions as F
from pyspark.sql import types as T

from amex_default_prediction.evaluation import AmexMetricEvaluator


def fit_simple(
    spark, model, grid, train_data_preprocessed_path, output_path, parallelism
):
    """Fit function that can be used with a variety of models for quick
    iteration. Assumes that preprocessing has already been done."""
    train_data = (
        spark.read.parquet((Path(train_data_preprocessed_path) / "data").as_posix())
        .withColumn("label", F.col("target").cast("float"))
        .cache()
    )

    # we also extract out the first column of the probability
    # https://stackoverflow.com/a/44505571
    # https://stackoverflow.com/a/44064252
    spark.udf.register("extract_pred", F.udf(lambda v: float(v[1]), T.FloatType()))
    pipeline = Pipeline(
        stages=[
            model,
            SQLTransformer(
                statement="SELECT *, extract_pred(probability) as pred FROM __THIS__"
            ),
        ]
    )
    evaluator = AmexMetricEvaluator(predictionCol="pred", labelCol="label")
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        parallelism=parallelism,
    )
    cv_model = cv.fit(train_data)
    # TODO: evaluate on an actual validation set
    print(evaluator.evaluate(cv_model.transform(train_data)))
    cv_model.write().overwrite().save(output_path)
