import click
import torch
import torch.nn as nn
from pyspark.ml import Pipeline, PipelineModel
from sparktorch import SparkTorch, create_spark_torch_model, serialize_torch_obj

from amex_default_prediction.evaluation import AmexMetricEvaluator
from amex_default_prediction.utils import spark_session

from ..torch.data_module import get_spark_feature_size
from ..torch.net import TransformerModel
from .base import fit_generic, read_train_data


@click.command()
@click.argument("train_data_preprocessed_path", type=click.Path(exists=True))
@click.argument("train_transformer_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--train-ratio", default=0.8, type=float)
@click.option("--parallelism", default=8, type=int)
def fit_with_transformer(
    train_data_preprocessed_path,
    train_transformer_path,
    output_path,
    train_ratio,
    parallelism,
):
    spark = spark_session()
    fit_generic(
        spark,
        model,
        None,
        train_data_preprocessed_path,
        train_transformer_path,
        output_path,
        train_ratio=train_ratio,
        parallelism=parallelism,
    )
