import numpy as np
import pandas as pd

from amex_default_prediction.evaluation import AmexMetricEvaluator, amex_metric


def test_all_right_ones(spark):
    n = 10
    y_true = pd.DataFrame({"target": np.ones(n)})
    y_pred = pd.DataFrame({"prediction": np.ones(n)})
    assert amex_metric(y_true, y_pred) == 0.5
    assert (
        AmexMetricEvaluator(predictionCol="prediction", labelCol="target").evaluate(
            spark.createDataFrame(pd.concat([y_true, y_pred], axis="columns")),
        )
        == 0.5
    )


def test_random_target_prediction(spark):
    n = 1000
    y_true = pd.DataFrame({"target": np.random.randint(0, 2, n)})
    y_pred = pd.DataFrame({"prediction": np.random.randint(0, 2, n)})
    assert AmexMetricEvaluator(predictionCol="prediction", labelCol="target").evaluate(
        spark.createDataFrame(pd.concat([y_true, y_pred], axis="columns")),
    ) == amex_metric(y_true, y_pred)


def test_amex_metric_evaluator_persists(tmp_path):
    evaluator = AmexMetricEvaluator()
    evaluator.write().overwrite().save(str(tmp_path))
    loaded_evaluator = AmexMetricEvaluator.read().load(str(tmp_path))
