from pathlib import Path

import click
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import SQLTransformer, VectorSlicer
from pyspark.ml.regression import AFTSurvivalRegression
from pyspark.ml.tuning import ParamGridBuilder

from amex_default_prediction.utils import spark_session

from .base import ExtractVectorIndexTransformer, fit_generic


@click.command()
@click.argument("train_data_preprocessed_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--train-ratio", default=0.8, type=float)
@click.option("--parallelism", default=2, type=int)
def fit(train_data_preprocessed_path, output_path, train_ratio, parallelism):
    spark = spark_session()
    train_data_pipeline = PipelineModel.read().load(
        (Path(train_data_preprocessed_path) / "pipeline").as_posix()
    )
    params = {
        k.name: v for k, v in train_data_pipeline.stages[-1].extractParamMap().items()
    }
    age_col_idx = [i for i, v in enumerate(params["inputCols"]) if v == "age"][0]
    cols_except_age = [i for i, v in enumerate(params["inputCols"]) if v != "age"]

    model = AFTSurvivalRegression(
        featuresCol="features_except_age", labelCol="age_plus_one", censorCol="censor"
    )
    grid = ParamGridBuilder().addGrid(model.aggregationDepth, [2]).build()
    fit_generic(
        spark,
        Pipeline(
            stages=[
                VectorSlicer(
                    inputCol="features",
                    indices=cols_except_age,
                    outputCol="features_except_age",
                ),
                ExtractVectorIndexTransformer(
                    inputCol="features", outputCol="age", indexCol=age_col_idx
                ),
                # create the censor column, using the label column. If the is a
                # default event, it means our data has not been censored
                SQLTransformer(
                    statement="""
                        SELECT
                            *,
                            if(label=1, 0, 1) as censor,
                            age + 1 as age_plus_one
                        FROM __THIS__
                    """
                ),
                model,
            ]
        ),
        grid,
        RegressionEvaluator(),
        train_data_preprocessed_path,
        output_path,
        train_ratio,
        parallelism,
    )
