import click
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder

from amex_default_prediction.utils import spark_session

from .base import fit_simple


@click.group
def logistic():
    pass


@logistic.command()
@click.argument("train_data_preprocessed_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--parallelism", default=8, type=int)
def fit(train_data_preprocessed_path, output_path, parallelism):
    spark = spark_session()
    lr = LogisticRegression(family="binomial", probabilityCol="probability")
    grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.1, 1, 5, 10])
        .addGrid(lr.elasticNetParam, [0, 0.5, 1])
        .build()
    )
    fit_simple(spark, lr, grid, train_data_preprocessed_path, output_path, parallelism)
