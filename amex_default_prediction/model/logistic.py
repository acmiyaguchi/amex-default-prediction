import click
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import SQLTransformer, StandardScaler, VectorAssembler
from pyspark.ml.tuning import CrossValidatorModel, ParamGridBuilder

from amex_default_prediction.evaluation import AmexMetricEvaluator
from amex_default_prediction.utils import spark_session

from .base import ExtractVectorIndexTransformer, fit_generic, fit_simple


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


@click.command()
@click.argument("train_data_preprocessed_path", type=click.Path(exists=True))
@click.argument("aft_model_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--train-ratio", default=0.8, type=float)
@click.option("--parallelism", default=8, type=int)
def fit_with_aft(
    train_data_preprocessed_path, aft_model_path, output_path, train_ratio, parallelism
):
    spark = spark_session()
    aft_model = CrossValidatorModel.read().load(aft_model_path)
    model = LogisticRegression(
        featuresCol="features_with_aft",
        family="binomial",
        probabilityCol="probability",
    )
    grid = (
        ParamGridBuilder()
        .addGrid(model.regParam, [0.1, 1])
        .addGrid(model.elasticNetParam, [0, 0.5, 1])
        .build()
    )
    fit_generic(
        spark,
        Pipeline(
            stages=[
                aft_model.bestModel,
                StandardScaler(
                    inputCol="quantiles_probability",
                    outputCol="aft_scaled",
                    withStd=True,
                    withMean=True,
                ),
                VectorAssembler(
                    inputCols=["features", "aft_scaled"],
                    outputCol="features_with_aft",
                ),
                # lets keep a subset of fields
                SQLTransformer(
                    statement="SELECT customer_ID, features_with_aft, label FROM __THIS__"
                ),
                model,
                ExtractVectorIndexTransformer(
                    inputCol="probability", outputCol="pred", indexCol=1
                ),
            ]
        ),
        grid,
        AmexMetricEvaluator(predictionCol="pred", labelCol="label"),
        train_data_preprocessed_path,
        output_path,
        train_ratio,
        parallelism,
    )
