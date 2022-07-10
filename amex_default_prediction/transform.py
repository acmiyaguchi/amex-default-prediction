from pathlib import Path

import click
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import Window
from pyspark.sql import functions as F

from .utils import spark_session


@click.group()
def transform():
    pass


@transform.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--num-partitions", default=64, type=int)
def raw_to_parquet(input_path, output_path, num_partitions):
    df = spark_session().read.csv(input_path, header=True).repartition(num_partitions)
    df.write.parquet(output_path, mode="overwrite")


def prepare_dataset(train_data, train_labels):
    window = Window.partitionBy("customer_ID").orderBy(F.desc("S_2"))

    min_age = (
        train_data.groupby("customer_ID")
        .agg(
            F.min("S_2").alias("min_date"),
            F.max("S_2").alias("max_date"),
            F.count("*").alias("n_statements"),
        )
        .withColumn("age", F.datediff("max_date", "min_date"))
        .select("customer_ID", "age", "n_statements")
    )

    return (
        train_data.withColumn(
            "rank",
            F.row_number().over(window),
        )
        .where("rank = 1")
        .drop("rank")
        .join(min_age, on="customer_ID")
        .join(train_labels, on="customer_ID")
    )


@transform.command()
@click.argument("train_data_path", type=click.Path(exists=True))
@click.argument("train_labels_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--limit", default=None, type=int)
def preprocess_training_dataset(train_data_path, train_labels_path, output_path, limit):
    spark = spark_session()
    train_data = spark.read.parquet(train_data_path)
    if limit:
        train_data = train_data.limit(limit)
    train_labels = spark.read.parquet(train_labels_path)
    prepared_df = prepare_dataset(train_data, train_labels).cache()

    categorical_labels = [
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

    feature_columns = [
        c for c in prepared_df.columns[2:-1] if c not in categorical_labels
    ]
    string_indexed_columns = [f"{c}_str" for c in categorical_labels]
    ohe_columns = [f"{c}_ohe" for c in categorical_labels]

    string_indexer = StringIndexer(
        inputCols=categorical_labels,
        outputCols=string_indexed_columns,
        handleInvalid="keep",
    )
    ohe = OneHotEncoder(inputCols=string_indexed_columns, outputCols=ohe_columns)
    imputer = Imputer(inputCols=feature_columns, outputCols=feature_columns)
    assembler = VectorAssembler(
        inputCols=feature_columns + ohe_columns, outputCol="features"
    )
    pipeline = Pipeline(stages=[string_indexer, ohe, imputer, assembler])

    # ensure that all of our columns are actually floating point values
    casted_df = prepared_df.select(
        "customer_ID",
        "target",
        *[F.col(c).cast("float").alias(c) for c in feature_columns],
        *categorical_labels,
    )
    transforms = pipeline.fit(casted_df)

    df = transforms.transform(casted_df).select("customer_ID", "features", "target")
    df.show(n=3, vertical=True, truncate=100)

    transforms.write().overwrite().save((Path(output_path) / "pipeline").as_posix())
    df.write.parquet((Path(output_path) / "data").as_posix(), mode="overwrite")
