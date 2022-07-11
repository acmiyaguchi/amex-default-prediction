import click
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder

from amex_default_prediction.utils import spark_session

from .base import fit_simple


@click.command()
@click.argument("train_data_preprocessed_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--train-ratio", default=0.8, type=float)
@click.option("--parallelism", default=8, type=int)
def fit(train_data_preprocessed_path, output_path, train_ratio, parallelism):
    spark = spark_session()
    model = LogisticRegression(family="binomial", probabilityCol="probability")
    grid = (
        ParamGridBuilder()
        .addGrid(model.regParam, [0.1, 1])
        .addGrid(model.elasticNetParam, [0, 0.5, 1])
        .build()
    )
    fit_simple(
        spark,
        model,
        grid,
        train_data_preprocessed_path,
        output_path,
        train_ratio,
        parallelism,
    )
