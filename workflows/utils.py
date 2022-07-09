from pathlib import Path
from subprocess import run


def build_wheel():
    run("python setup.py bdist_wheel", shell=True, check=True)


def get_wheel():
    return sorted(Path("dist").glob("*.whl"))[-1].as_posix()


def run_spark(task, master="local[*]", spark_driver_memory="32g", print_only=False):
    cmd = " ".join(
        [
            "spark-submit",
            f"--master {master}",
            "--conf spark.pyspark.python=python",
            f"--conf spark.driver.memory={spark_driver_memory}",
            # use the local disk in the data directory for spill
            "--conf spark.local.dir=data/tmp/spark",
            f"--py-files {get_wheel()}",
            "main.py",
            task,
        ]
    )
    print(cmd)
    if print_only:
        return None
    return run(cmd, shell=True, check=True)
