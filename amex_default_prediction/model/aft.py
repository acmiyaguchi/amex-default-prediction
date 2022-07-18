from pathlib import Path

import click
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import SQLTransformer, VectorSlicer
from pyspark.ml.regression import AFTSurvivalRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

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
    # get the last stage with the vector assembler in it
    assembler_index = [
        i
        for i, x in enumerate(train_data_pipeline.stages)
        if "VectorAssembler" in x.__class__.__name__
    ][-1]
    params = {
        k.name: v
        for k, v in train_data_pipeline.stages[assembler_index]
        .extractParamMap()
        .items()
    }
    age_col_idx = [i for i, v in enumerate(params["inputCols"]) if v == "age_days"][0]
    cols_except_age = [i for i, v in enumerate(params["inputCols"]) if v != "age_days"]

    model = AFTSurvivalRegression(
        featuresCol="features_except_age",
        labelCol="age_plus_one",
        censorCol="censor",
        quantilesCol="quantiles_probability",
    )
    grid = ParamGridBuilder().addGrid(model.aggregationDepth, [2]).build()
    evaluator = RegressionEvaluator()
    fit_generic(
        spark,
        CrossValidator(
            estimator=Pipeline(
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
                            label as censor,
                            age + 1 as age_plus_one
                        FROM __THIS__
                    """
                    ),
                    model,
                ]
            ),
            estimatorParamMaps=grid,
            evaluator=evaluator,
            parallelism=parallelism,
        ),
        evaluator,
        train_data_preprocessed_path,
        output_path,
        train_ratio=train_ratio,
        train_most_recent_only=False,
        validation_most_recent_only=True,
        data_most_recent_only=True,
    )
