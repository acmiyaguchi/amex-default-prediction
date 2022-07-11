import click
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.tuning import ParamGridBuilder

from amex_default_prediction.utils import spark_session

from .base import fit_simple


@click.group
def gbt():
    pass


@gbt.command()
@click.argument("train_data_preprocessed_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--train-ratio", default=0.8, type=float)
@click.option("--parallelism", default=2, type=int)
def fit(train_data_preprocessed_path, output_path, train_ratio, parallelism):
    spark = spark_session()
    model = GBTClassifier()
    grid = (
        ParamGridBuilder()
        .addGrid(model.maxIter, [100])
        .addGrid(model.stepSize, [0.1])
        .addGrid(model.maxDepth, [2, 3, 4])
        .addGrid(model.maxBins, [32])
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
