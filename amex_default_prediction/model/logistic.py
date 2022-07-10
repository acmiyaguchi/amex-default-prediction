from pathlib import Path

import click
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import SQLTransformer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from amex_default_prediction.evaluation import AmexMetricEvaluator


@click.group
def logistic():
    pass


@logistic.command()
@click.argument("train_data_preprocessed_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def fit(train_data_preprocessed_path, output_path):
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    train_data = (
        spark.read.parquet((Path(train_data_preprocessed_path) / "data").as_posix())
        .withColumn("label", F.col("target").cast("float"))
        .cache()
    )

    lr = LogisticRegression(
        maxIter=10,
        regParam=0.3,
        elasticNetParam=0.8,
        family="binomial",
        probabilityCol="probability",
    )

    # we also extract out the first column of the probability
    # https://stackoverflow.com/a/44505571
    # https://stackoverflow.com/a/44064252
    spark.udf.register("extract_pred", F.udf(lambda v: float(v[1]), T.FloatType()))
    pipeline = Pipeline(
        stages=[
            lr,
            SQLTransformer(
                statement="SELECT *, extract_pred(probability) as pred FROM __THIS__"
            ),
        ]
    )
    grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.1, 1, 5, 10])
        .addGrid(lr.elasticNetParam, [0, 0.5, 1])
        .build()
    )
    evaluator = AmexMetricEvaluator(predictionCol="pred", labelCol="label")
    cv = CrossValidator(
        estimator=pipeline, estimatorParamMaps=grid, evaluator=evaluator, parallelism=8
    )
    cv_model = cv.fit(train_data)
    # TODO: evaluate on an actual validation set
    print(evaluator.evaluate(cv_model.transform(train_data)))
    cv_model.write().overwrite().save(output_path)
