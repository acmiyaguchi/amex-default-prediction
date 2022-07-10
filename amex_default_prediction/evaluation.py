import pandas as pd
from pyspark.ml.evaluation import Evaluator
from pyspark.sql import functions as F


def amex_metric_pandas(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """Metric taken from the competition notebook.

    https://www.kaggle.com/code/inversion/amex-competition-metric-python
    """

    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = pd.concat([y_true, y_pred], axis="columns").sort_values(
            "prediction", ascending=False
        )
        df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
        four_pct_cutoff = int(0.04 * df["weight"].sum())
        df["weight_cumsum"] = df["weight"].cumsum()
        df_cutoff = df.loc[df["weight_cumsum"] <= four_pct_cutoff]
        return (df_cutoff["target"] == 1).sum() / (df["target"] == 1).sum()

    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = pd.concat([y_true, y_pred], axis="columns").sort_values(
            "prediction", ascending=False
        )
        df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
        df["random"] = (df["weight"] / df["weight"].sum()).cumsum()
        total_pos = (df["target"] * df["weight"]).sum()
        df["cum_pos_found"] = (df["target"] * df["weight"]).cumsum()
        df["lorentz"] = df["cum_pos_found"] / total_pos
        df["gini"] = (df["lorentz"] - df["random"]) * df["weight"]
        return df["gini"].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={"target": "prediction"})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)


class AmexEvaluator(Evaluator):
    def __init__(self, predictionCol="prediction", labelCol="label"):
        self.predictionCol = predictionCol
        self.labelCol = labelCol

    def _evaluate(self, dataset):
        df = dataset.select(
            F.col(self.labelCol).alias("target"),
            F.col(self.predictionCol).alias("prediction"),
        ).toPandas()
        return amex_metric_pandas(df[["target"]], df[["prediction"]])

    def isLargerBetter(self):
        return True
