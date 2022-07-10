from pathlib import Path

import click
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


@click.group
def logistic():
    pass


@logistic.command()
@click.argument("train_data_preprocessed_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def fit(train_data_preprocessed_path, output_path):
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    train_data = spark.read.parquet(
        (Path(train_data_preprocessed_path) / "data").as_posix()
    ).withColumn("label", F.col("target").cast("integer"))

    lr = LogisticRegression(
        maxIter=10,
        regParam=0.3,
        elasticNetParam=0.8,
        family="binomial",
        probabilityCol="probability",
    )
    grid = (
        ParamGridBuilder()
        .addGrid(lr.maxIter, [0, 10])
        .addGrid(lr.regParam, [0.1, 1, 5, 10])
        .addGrid(lr.elasticNetParam, [0, 0.5, 1])
        .build()
    )
    evaluator = BinaryClassificationEvaluator()
    cv = CrossValidator(
        estimator=lr, estimatorParamMaps=grid, evaluator=evaluator, parallelism=4
    )
    cvModel = cv.fit(train_data)
    print(evaluator.evaluate(cvModel.transform(train_data)))

    # model.write.overwrite().save(output_path)
