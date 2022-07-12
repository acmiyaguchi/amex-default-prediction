from pyspark.sql import SparkSession


def spark_session():
    spark = (
        SparkSession.builder
        # configuration shared across contexts
        .config("spark.pyspark.python", "python")
        .config("spark.sql.execution.arrow.pyspark.enabled", True)
        .config("spark.local.dir", "data/tmp/spark")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
