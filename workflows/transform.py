import os
from pathlib import Path

import luigi
import luigi.contrib.spark
import pyspark


def get_python_path():
    venv_path = os.environ.get("VIRTUAL_ENV")
    if venv_path is None:
        return "python"
    # get the python in the virtual environment if it exists
    python_exe = list(Path(venv_path).glob("**/python.exe"))
    if python_exe:
        return python_exe[0].as_posix()
    return "python"


os.environ["SPARK_HOME"] = pyspark.__path__[0]


class RawIntoParquet(luigi.contrib.spark.SparkSubmitTask):
    app = "main.py"
    master = "local[*]"

    spark_submit = "spark-submit.cmd"
    pyspark_python = "python"
    py_files = [sorted(Path("dist").glob("*.egg"))[-1].as_posix()]
    filename = "train_labels"

    def output(self):
        return luigi.LocalTarget(f"data/intermediate/{self.filename}")

    def app_options(self):
        return [
            "transform",
            "raw-to-parquet",
            f"data/raw/amex-default-prediction/{self.filename}.csv",
            self.output().path,
            "--num-partitions",
            "64",
        ]
