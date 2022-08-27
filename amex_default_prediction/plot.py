from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import umap
import umap.plot
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import Window
from pyspark.sql import functions as F

from amex_default_prediction.model.base import read_train_data
from amex_default_prediction.utils import spark_session


@click.group()
def plot():
    pass


@plot.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("train_data_preprocessed_path", type=click.Path(exists=True))
@click.option("--output-path", type=click.Path(exists=False))
def plot_pca(model_path, train_data_preprocessed_path, output_path):
    spark = spark_session()
    model = PipelineModel.read().load(model_path)

    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    var = model.stages[-1].explainedVariance
    print(np.array(var))
    print(var.cumsum())
    plt.title("scree plot")
    plt.plot(np.log(var))
    if output_path:
        plt.savefig(output_path / "scree.png")
    else:
        plt.show()

    df, _, _ = read_train_data(spark, train_data_preprocessed_path, cache=False)
    pdf = (
        model.transform(df)
        .select(vector_to_array("features_pca").alias("feature"), "label")
        .toPandas()
    )
    print(pdf.head())
    print(pdf.shape)

    X = np.stack(pdf.feature.values)
    plt.title("scatter plot of first two principle components")
    plt.scatter(X[:, 0], X[:, 1], c=pdf.label.values, s=2)
    if output_path:
        plt.savefig(output_path / "scatter.png")
    else:
        plt.show()

    mapper = umap.UMAP(n_components=2, n_neighbors=20).fit(X)
    p = umap.plot.interactive(
        mapper, color_key_cmap="Paired", labels=pdf.label.values, point_size=2
    )
    if output_path:
        umap.plot.output_file(output_path / "umap.html")
    umap.plot.show(p)


@plot.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("train_data_preprocessed_path", type=click.Path(exists=True))
@click.argument("train_transformer_path", type=click.Path(exists=True))
@click.option("--output-path", type=click.Path(exists=False))
@click.option("--truncate-predict", default=None, type=int)
def plot_transformer(
    model_path,
    train_data_preprocessed_path,
    train_transformer_path,
    output_path,
    truncate_predict,
):
    spark = spark_session()
    train_df, _, _ = read_train_data(spark, train_data_preprocessed_path, cache=False)
    train_transformer_df = spark.read.parquet(train_transformer_path)

    df = train_df.select(
        "customer_ID",
        F.hash("customer_ID").alias("customer_index"),
        "label",
    ).join(train_transformer_df, on="customer_index", how="inner")

    pdf = df.select("prediction", "label").toPandas()
    print(pdf.head())
    print(pdf.shape)

    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    X = np.stack(pdf.prediction.values)
    if truncate_predict:
        X = X[:, :truncate_predict]
    plt.title("scatter plot of first two components")
    plt.scatter(X[:, 0], X[:, 1], c=pdf.label.values, s=2)
    if output_path:
        plt.savefig(output_path / "scatter.png")
    else:
        plt.show()

    mapper = umap.UMAP(n_components=2, n_neighbors=20).fit(X)
    p = umap.plot.interactive(
        mapper, color_key_cmap="Paired", labels=pdf.label.values, point_size=2
    )
    if output_path:
        umap.plot.output_file(output_path / "umap.html")
    umap.plot.show(p)
