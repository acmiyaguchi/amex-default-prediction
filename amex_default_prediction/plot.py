import click
import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array

from amex_default_prediction.model.base import read_train_data
from amex_default_prediction.utils import spark_session


@click.group()
def plot():
    pass


@plot.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("train_data_preprocessed_path", type=click.Path(exists=True))
def plot_pca(model_path, train_data_preprocessed_path):

    spark = spark_session()
    model = PipelineModel.read().load(model_path)

    var = model.stages[1].explainedVariance
    plt.title("scree plot")
    plt.plot(var)
    plt.show()

    df, _, _ = read_train_data(spark, train_data_preprocessed_path, cache=False)
    subset = df.where("sample_id = 0").limit(5000)
    pdf = (
        model.transform(subset)
        .select(vector_to_array("features_pca").alias("feature"))
        .toPandas()
    )
    print(pdf.head())
    X = np.stack(pdf.feature.values)

    plt.title("scatter plot of first two principle components")
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()
