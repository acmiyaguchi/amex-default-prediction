from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.functions import array_to_vector
from pyspark.ml.tuning import CrossValidatorModel, ParamGridBuilder

from amex_default_prediction.model.base import fit_simple


@pytest.fixture
def test_data_path(spark, tmp_path):
    num_rows = 20
    num_features = 3
    pdf = pd.DataFrame(
        dict(
            customer_ID=[str(uuid4()) for _ in range(num_rows)],
            target=np.random.randint(0, 2, num_rows),
            features=[np.random.rand(num_features) for _ in range(num_rows)],
        )
    )
    df = (
        spark.createDataFrame(pdf)
        .withColumn("features", array_to_vector("features"))
        .repartition(1)
    )
    df.printSchema()
    # our training procedure dumps out a data and pipeline directory
    output = tmp_path / "test_data"
    df.write.parquet((output / "data").as_posix())
    yield output


def test_logistic_regression_loads(tmp_path, spark, test_data_path):
    model = LogisticRegression(
        maxIter=0, family="binomial", probabilityCol="probability"
    )
    grid = ParamGridBuilder().build()
    output = tmp_path / "model"
    fit_simple(spark, model, grid, test_data_path, output, 0.8, 1)
    cv_model = CrossValidatorModel.read().load(output.as_posix())
    assert cv_model.avgMetrics[0]
