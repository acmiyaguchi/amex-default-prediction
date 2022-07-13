import pytest
from click.testing import CliRunner
from pyspark.ml import PipelineModel
from pyspark.ml import functions as mlF
from pyspark.sql import functions as F

from amex_default_prediction.transform import (
    PrepareDatasetTransformer,
    preprocess_training_dataset,
    raw_to_parquet,
)

# subset of total columns
NUMERIC_COLS = ["B_1", "D_41"]
CATEGORICAL_COLS = ["B_30", "D_114"]


@pytest.fixture(scope="session")
def train_data_df(spark, data_path):
    return spark.read.csv(
        (data_path / "train_data.csv").as_posix(), header=True
    ).cache()


@pytest.fixture(scope="session")
def train_labels_df(spark, data_path):
    return spark.read.csv(
        (data_path / "train_labels.csv").as_posix(), header=True
    ).cache()


@pytest.fixture
def data_path_with_imputed(data_path, tmp_path):
    return data_path / "data_with_imputed"


def test_train_data_properties(train_data_df):
    assert train_data_df.count() == 20
    assert train_data_df.groupby("customer_ID").count().count() == 2


def test_train_labels_properties(train_labels_df):
    assert train_labels_df.count() == 20
    assert train_labels_df.groupby("customer_ID").count().count() == 20


def test_prepare_dataset_transformer(train_data_df):
    transformer = PrepareDatasetTransformer(CATEGORICAL_COLS, NUMERIC_COLS)
    assert transformer.getCategoricalCols() == CATEGORICAL_COLS
    assert transformer.getNumericalCols() == NUMERIC_COLS + ["n_statements", "age_days"]

    prepared_df = transformer.transform(train_data_df)
    assert prepared_df.count() == 20
    assert prepared_df.groupby("customer_ID").count().count() == 2
    assert set(prepared_df.columns) == {
        "customer_ID",
        "statement_date",
        "most_recent",
        *NUMERIC_COLS,
        *CATEGORICAL_COLS,
        "n_statements",
        "age_days",
        "sample_id",
    }

    # numeric types are correct
    dtypes = dict(prepared_df.dtypes)
    assert all([dtypes[col] == "float" for col in NUMERIC_COLS])
    assert dtypes["statement_date"] == "date"
    assert dtypes["most_recent"] == "boolean"

    # numeric values are correct
    row = prepared_df.select(
        *[F.sum(col).alias(col) for col in NUMERIC_COLS]
    ).collect()[0]
    assert all([row[col] > 0 for col in NUMERIC_COLS])

    # assert number of statements is increasing for a customer, row numbers
    # start at 1
    rows = (
        prepared_df.select("customer_ID", "n_statements")
        .orderBy("customer_ID", "n_statements")
        .collect()
    )
    customer_id = rows[0].customer_ID
    cur = 1
    for row in rows:
        if row.customer_ID != customer_id:
            continue
        assert row.n_statements == cur
        cur += 1

    assert prepared_df.where("most_recent").count() == 2


def test_preprocess_training_dataset(spark, data_path, tmp_path, train_data_df):
    train_path = tmp_path / "train"
    label_path = tmp_path / "label"
    output_path = tmp_path / "output"

    CliRunner().invoke(
        raw_to_parquet,
        [
            (data_path / "train_data.csv").as_posix(),
            train_path.as_posix(),
            "--num-partitions",
            "1",
        ],
        catch_exceptions=False,
    )

    CliRunner().invoke(
        raw_to_parquet,
        [
            (data_path / "train_labels.csv").as_posix(),
            label_path.as_posix(),
            "--num-partitions",
            "1",
        ],
        catch_exceptions=False,
    )

    CliRunner().invoke(
        preprocess_training_dataset,
        [
            train_path.as_posix(),
            label_path.as_posix(),
            output_path.as_posix(),
            "--drop-columns",
            # obtain this list by running the command without it first. these
            # are all numeric columns that can't be imputed because they're null
            (
                "D_42,D_49,D_53,B_17,D_73,D_76,R_9,B_29,D_87,D_88,D_106,R_26,D_108,"
                "D_110,D_111,B_39,B_42,D_132,D_134,D_135,D_136,D_137,D_138,D_142"
            ),
        ],
        catch_exceptions=False,
    )

    # assert some properties about the transformed data
    df = spark.read.parquet((output_path / "data").as_posix()).cache()
    assert df.count() == 20
    assert df.groupby("customer_ID").count().count() == 2

    # read the pipeline and transform the data
    pipeline = PipelineModel.read().load((output_path / "pipeline").as_posix())
    assert pipeline.stages[0].__class__.__name__ == "PrepareDatasetTransformer"
    assert len(pipeline.stages) == 6

    def length(df):
        return (
            df.select(F.size(mlF.vector_to_array("features")).alias("len"))
            .limit(1)
            .collect()[0]
            .len
        )

    transformed = pipeline.transform(train_data_df)
    transformed.printSchema()
    assert length(transformed) == length(df)
    assert set(df.columns) - set(transformed.columns) == {"label"}
