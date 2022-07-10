import pandas as pd
from pyspark.ml.evaluation import Evaluator
from pyspark.sql import Window
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

    def _top_four_percent_captured(self, dataset):
        window = Window.orderBy(F.desc(self.predictionCol)).rangeBetween(
            Window.unboundedPreceding, 0
        )
        window_all = Window.partitionBy()
        return (
            dataset.orderBy(F.desc(self.predictionCol))
            .withColumn("weight", F.when(F.col(self.labelCol) == 0, 20).otherwise(1))
            .withColumn("four_pct_cutoff", 0.04 * F.sum("weight").over(window_all))
            .withColumn("weight_cumsum", F.sum("weight").over(window))
            .select(
                (
                    F.sum(F.expr("weight_cumsum <= four_pct_cutoff").cast("float"))
                    / F.sum(self.labelCol)
                ).alias("top_four_percent_captured")
            )
            .collect()[0]["top_four_percent_captured"]
        )

    def _weighted_gini(self, dataset):
        window = Window.orderBy(F.desc(self.predictionCol)).rangeBetween(
            Window.unboundedPreceding, 0
        )
        window_all = Window.partitionBy()
        return (
            dataset.orderBy(F.desc(self.predictionCol))
            .withColumn("weight", F.when(F.col(self.labelCol) == 0, 20).otherwise(1))
            .withColumn(
                "random",
                F.sum(F.col("weight") / F.sum("weight").over(window_all)).over(window),
            )
            .withColumn(
                "total_pos",
                F.sum(F.col(self.labelCol) * F.col("weight")).over(window_all),
            )
            .withColumn(
                "cum_pos_found",
                F.sum(F.col(self.labelCol) * F.col("weight")).over(window),
            )
            .withColumn(
                "lorentz",
                F.expr("cum_pos_found / total_pos"),
            )
            .withColumn("gini", F.expr("(lorentz - random) * weight"))
            .select(F.sum("gini").alias("weighted_gini"))
            .collect()[0]["weighted_gini"]
        )

    def _evaluate(self, dataset):
        g = self._weighted_gini(dataset)
        d = self._top_four_percent_captured(dataset)
        print(g, d)
        return 0.5 * (g + d)

    def isLargerBetter(self):
        return True
