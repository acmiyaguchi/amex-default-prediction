from pathlib import Path

import click
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Imputer,
    OneHotEncoder,
    SQLTransformer,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.param.shared import Param, Params, TypeConverters
from pyspark.ml.pipeline import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import Window
from pyspark.sql import functions as F

from .utils import spark_session


@click.group(name="transform")
def transform_group():
    pass


@transform_group.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--num-partitions", default=64, type=int)
def raw_to_parquet(input_path, output_path, num_partitions):
    df = spark_session().read.csv(input_path, header=True).repartition(num_partitions)
    df.write.parquet(output_path, mode="overwrite")


class HasCategoricalCols(Params):
    categoricalCols = Param(
        Params._dummy(),
        "categoricalCols",
        "categorical column names.",
        typeConverter=TypeConverters.toListString,
    )

    def __init__(self) -> None:
        super().__init__()

    def getCategoricalCols(self):
        return self.getOrDefault(self.categoricalCols)


class HasNumericalCols(Params):
    numericalCols = Param(
        Params._dummy(),
        "numericalCols",
        "numerical column names.",
        typeConverter=TypeConverters.toListString,
    )

    def __init__(self) -> None:
        super().__init__()

    def getNumericalCols(self):
        return self.getOrDefault(self.numericalCols)


class PrepareDatasetTransformer(
    Transformer,
    HasCategoricalCols,
    HasNumericalCols,
    DefaultParamsWritable,
    DefaultParamsReadable,
):
    """A transformer for preparing the parquet dataset for training"""

    def __init__(self, categoricalCols=[], numericalCols=[]) -> None:
        super().__init__()
        self._setDefault(
            categoricalCols=categoricalCols,
            # the passed in values plus derived values
            numericalCols=numericalCols + ["n_statements", "age_days"],
        )

    def _transform(self, dataset):
        min_age = dataset.groupby("customer_ID").agg(
            F.min("S_2").alias("min_date"), F.max("S_2").alias("max_date")
        )
        window = Window.partitionBy("customer_ID").orderBy(F.asc("S_2"))
        return (
            dataset.join(min_age, on="customer_ID")
            # cumulative number of statements
            .withColumn("n_statements", F.row_number().over(window))
            # cumulative age of the customer
            .withColumn("age_days", F.datediff("S_2", "min_date")).select(
                "customer_ID",
                F.col("S_2").cast("date").alias("statement_date"),
                (F.col("S_2") == F.col("max_date"))
                .cast("boolean")
                .alias("most_recent"),
                *[F.col(c).cast("float").alias(c) for c in self.getNumericalCols()],
                *self.getCategoricalCols(),
                (F.crc32("customer_ID") % 32).alias("sample_id"),
            )
        )


@transform_group.command()
@click.argument("train_data_path", type=click.Path(exists=True))
@click.argument("train_labels_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option(
    "--drop-columns", type=str, help="comma-delimited list of columns to drop"
)
@click.option("--limit", default=None, type=int)
def preprocess_training_dataset(
    train_data_path, train_labels_path, output_path, drop_columns, limit
):
    spark = spark_session()
    train_data = spark.read.parquet(train_data_path)
    if limit:
        train_data = train_data.limit(limit)
    train_labels = spark.read.parquet(train_labels_path).select(
        "customer_ID", F.col("target").cast("float").alias("label")
    )

    drop_columns = (
        [col.strip() for col in drop_columns.split(",")] if drop_columns else []
    )

    categorical_cols = [
        "B_30",
        "B_38",
        "D_114",
        "D_116",
        "D_117",
        "D_120",
        "D_126",
        "D_63",
        "D_64",
        "D_66",
        "D_68",
    ]

    # Played around with the style here a bit, although I flip-flop between the
    # style that's most aesthetically pleasing.
    prep_transformer = PrepareDatasetTransformer(
        numericalCols=[
            c
            for c in train_data.columns[2:-1]
            if (c not in categorical_cols) and (c not in drop_columns)
        ],
        categoricalCols=[c for c in categorical_cols if c not in drop_columns],
    )
    string_indexer = StringIndexer(
        inputCols=prep_transformer.getCategoricalCols(),
        outputCols=[f"{c}_str" for c in prep_transformer.getCategoricalCols()],
        handleInvalid="keep",
    )
    one_hot_encoder = OneHotEncoder(
        inputCols=string_indexer.getOutputCols(),
        outputCols=[f"{c}_ohe" for c in prep_transformer.getCategoricalCols()],
        handleInvalid="keep",
    )
    # NOTE: it is important to actually call getNumericalCols here because the
    # prep transformer adds extra columns
    imputer = Imputer(
        inputCols=prep_transformer.getNumericalCols(),
        outputCols=prep_transformer.getNumericalCols(),
        strategy="median",
    )

    pipeline = Pipeline(
        stages=[
            prep_transformer,
            string_indexer,
            one_hot_encoder,
            imputer,
            VectorAssembler(
                inputCols=imputer.getOutputCols() + one_hot_encoder.getOutputCols(),
                outputCol="features",
            ),
            SQLTransformer(
                statement="""
                    SELECT
                        sample_id,
                        customer_ID,
                        statement_date,
                        most_recent,
                        features
                    FROM
                        __THIS__
                """
            ),
        ]
    )

    # ensure that all of our columns are actually floating point values
    transforms = pipeline.fit(train_data)

    df = (
        transforms.transform(train_data).join(train_labels, on="customer_ID")
        # only return the label/default event on the most recent row of data
        .withColumn(
            "label", F.when(F.col("most_recent"), F.col("label")).otherwise(F.lit(0.0))
        )
    )
    df.printSchema()

    transforms.write().overwrite().save((Path(output_path) / "pipeline").as_posix())
    df.write.parquet((Path(output_path) / "data").as_posix(), mode="overwrite")
