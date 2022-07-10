from pyspark.sql import SparkSession


def spark_session():
    spark = SparkSession.builder.config(
        "spark.sql.execution.arrow.pyspark.enabled", True
    ).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
