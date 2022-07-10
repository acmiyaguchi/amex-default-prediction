import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from amex_default_prediction.evaluation import AmexEvaluator, amex_metric_pandas


@pytest.fixture(scope="session")
def spark():
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def test_all_right_ones(spark):
    n = 10
    y_true = pd.DataFrame({"target": np.ones(n)})
    y_pred = pd.DataFrame({"prediction": np.ones(n)})
    assert amex_metric_pandas(y_true, y_pred) == 0.5
    assert (
        AmexEvaluator("prediction", "target").evaluate(
            spark.createDataFrame(pd.concat([y_true, y_pred], axis="columns")),
        )
        == 0.5
    )


def test_random_target_prediction(spark):
    n = 1000
    y_true = pd.DataFrame({"target": np.random.randint(0, 2, n)})
    y_pred = pd.DataFrame({"prediction": np.random.randint(0, 2, n)})
    assert AmexEvaluator("prediction", "target").evaluate(
        spark.createDataFrame(pd.concat([y_true, y_pred], axis="columns")),
    ) == amex_metric_pandas(y_true, y_pred)
