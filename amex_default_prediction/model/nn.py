import click
import torch
import torch.nn as nn
from sparktorch import SparkTorch, serialize_torch_obj

from amex_default_prediction.evaluation import AmexMetricEvaluator
from amex_default_prediction.utils import spark_session

from .base import fit_generic, read_train_data


@click.command()
@click.argument("train_data_preprocessed_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--train-ratio", default=0.8, type=float)
@click.option("--parallelism", default=1, type=int)
def fit(train_data_preprocessed_path, output_path, train_ratio, parallelism):
    spark = spark_session()
    default_parallelism = spark.sparkContext.defaultParallelism

    # read the data so we can do stuff with it
    df, train_df, val_df = read_train_data(
        spark, train_data_preprocessed_path, train_ratio
    )

    def read_func(*args, **kwargs):
        # We override the read_func since we've already read and cached data.
        # This makes for terrible code organization, but it needs to be done for
        # neural nets. We also need to repartition the data
        return df, train_df, val_df

    input_size = val_df.head().features.size
    print(input_size, default_parallelism)

    network = nn.Sequential(
        nn.Linear(input_size, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 10),
        nn.Softmax(dim=1),
    )
    print(network)

    # Build the pytorch object
    torch_obj = serialize_torch_obj(
        model=network,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        lr=0.0001,
    )

    # Create a SparkTorch Model with torch distributed. Barrier execution is on
    # by default for this mode.
    model = SparkTorch(
        inputCol="features",
        labelCol="label",
        predictionCol="predictions",
        torchObj=torch_obj,
        iters=50,
        verbose=1,
        mode="hogwild",
        port=7077,
        useBarrier=True,
        partitions=default_parallelism,
        validationPct=0.2,
        earlyStopPatience=20,
        miniBatch=64,
        # TODO: implement support for this in a fork of sparktorch
        # device="gpu",
    )

    # TODO: cross validation

    fit_generic(
        spark,
        model,
        AmexMetricEvaluator(predictionCol="predictions", labelCol="label"),
        train_data_preprocessed_path,
        output_path,
        train_ratio,
        read_func=read_func,
    )
