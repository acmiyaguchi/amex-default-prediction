import click
from pyspark.sql import SparkSession


@click.group()
def transform():
    pass


@transform.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--num-partitions", default=64, type=int)
def raw_to_parquet(input_path, output_path, num_partitions):
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.csv(input_path).repartition(num_partitions)
    df.write.parquet(output_path, mode="overwrite")
