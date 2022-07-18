from pathlib import Path

import click
from pyspark.sql import functions as F

from amex_default_prediction.utils import spark_session


@click.command()
@click.option("--intermediate-path", default="data/intermediate")
@click.option("--num-scores", default=100, type=int)
def main(intermediate_path, num_scores):
    spark = spark_session()
    intermediate_root = Path(intermediate_path)
    spark.sql("set spark.sql.files.ignoreCorruptFiles=true")
    df = spark.read.json(f"{intermediate_root}/models/*/*/metadata/part-*")
    (
        df.where('class="pyspark.ml.tuning.CrossValidatorModel"')
        .withColumn("filename_parts", F.split(F.input_file_name(), "/"))
        .withColumn("scores", F.explode("avgMetrics"))
        .groupby(
            F.expr("filename_parts[7]").alias("model"),
            F.expr("filename_parts[8]").alias("version"),
        )
        .agg(F.max("scores").alias("bestScore"))
        .orderBy(F.desc("version"))
    ).show(n=num_scores, truncate=False)


if __name__ == "__main__":
    main()
