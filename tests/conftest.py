from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from pyspark.ml.functions import array_to_vector
from pyspark.sql import functions as F

from amex_default_prediction.torch.data_module import transform_vector_to_array
from amex_default_prediction.utils import spark_session


@pytest.fixture(scope="session")
def spark():
    ss = spark_session()
    yield ss
    ss.stop()


@pytest.fixture(scope="session")
def data_path():
    return Path(__file__).parent / "data"


@pytest.fixture
def cache_path():
    return Path(__file__).parent.parent / "data/tmp/pytest"


@pytest.fixture
def synthetic_train_df(spark):
    num_rows = 40
    num_features = 3
    pdf = pd.DataFrame(
        dict(
            customer_ID=[str(uuid4()) for _ in range(num_rows)],
            label=np.random.randint(0, 2, num_rows),
            features=[np.random.rand(num_features) for _ in range(num_rows)],
        )
    )
    df = (
        spark.createDataFrame(pdf)
        .withColumn("features", array_to_vector("features"))
        .withColumn("sample_id", F.crc32(F.col("customer_ID")) % 100)
        .withColumn("most_recent", F.lit(True))
        .repartition(1)
    )
    df.printSchema()
    return df


@pytest.fixture
def synthetic_train_data_path(synthetic_train_df, tmp_path):
    df = synthetic_train_df
    # our training procedure dumps out a data and pipeline directory
    output = tmp_path / "test_data"
    df.write.parquet((output / "data").as_posix())
    yield output


@pytest.fixture
def synthetic_train_data_torch_path(synthetic_train_df, tmp_path):
    df = synthetic_train_df
    output = tmp_path / "test_data"
    transform_vector_to_array(df, partitions=4).write.parquet(output.as_posix())
    yield output
