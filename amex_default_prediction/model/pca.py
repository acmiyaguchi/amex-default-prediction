import click
from pyspark.ml import Pipeline
from pyspark.ml.feature import PCA, Imputer, MinMaxScaler, RobustScaler, StandardScaler

from amex_default_prediction.utils import spark_session

from .base import fit_generic


@click.command()
@click.argument("train_data_preprocessed_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--train-ratio", default=1.0, type=float)
def fit(train_data_preprocessed_path, output_path, train_ratio):
    spark = spark_session()
    fit_generic(
        spark,
        Pipeline(
            stages=[
                StandardScaler(
                    inputCol="features",
                    outputCol="features_scaled",
                    withStd=True,
                    withMean=True,
                ),
                PCA(k=128, inputCol="features_scaled", outputCol="features_pca"),
            ]
        ),
        None,
        train_data_preprocessed_path,
        output_path,
        train_ratio=train_ratio,
        train_most_recent_only=False,
    )
