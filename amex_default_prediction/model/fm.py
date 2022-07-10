import click
from pyspark.ml.classification import FMClassifier
from pyspark.ml.tuning import ParamGridBuilder

from amex_default_prediction.utils import spark_session

from .base import fit_simple


@click.group
def fm():
    pass


@fm.command()
@click.argument("train_data_preprocessed_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--parallelism", default=8, type=int)
def fit(train_data_preprocessed_path, output_path, parallelism):
    spark = spark_session()
    model = FMClassifier()
    grid = (
        ParamGridBuilder()
        .addGrid(model.regParam, [0.1, 1, 10])
        .addGrid(model.factorSize, [2, 8, 16])
        .build()
    )
    fit_simple(
        spark, model, grid, train_data_preprocessed_path, output_path, parallelism
    )
