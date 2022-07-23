from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidatorModel, ParamGridBuilder

from amex_default_prediction.model.base import LogFeatureTransformer, fit_simple


def test_logistic_regression_loads(tmp_path, spark, synthetic_train_data_path):
    model = LogisticRegression(
        maxIter=0, family="binomial", probabilityCol="probability"
    )
    grid = ParamGridBuilder().build()
    output = tmp_path / "model"
    fit_simple(
        spark,
        model,
        grid,
        synthetic_train_data_path,
        output,
        parallelism=1,
        train_ratio=0.8,
    )
    cv_model = CrossValidatorModel.read().load(output.as_posix())
    assert cv_model.avgMetrics[0]


def test_log_feature_transformer(synthetic_train_df):
    transformer = LogFeatureTransformer()
    transformed_df = transformer.transform(synthetic_train_df)
    assert "features_log" in transformed_df.columns
    transformed_df.select("features", "features_log").show(
        n=5, vertical=True, truncate=80
    )
