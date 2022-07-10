import datetime
import re
from pathlib import Path
from subprocess import run


def build_wheel():
    run("python setup.py bdist_wheel", shell=True, check=True)


def get_wheel():
    return sorted(Path("dist").glob("*.whl"))[-1].as_posix()


def get_date_string():
    """Date string down to the seconds, only digits"""
    return re.sub(r"[^\d]+", "", datetime.datetime.utcnow().isoformat())[:14]


def get_head_rev_hash():
    return (
        run("git rev-parse --short HEAD", shell=True, check=True, capture_output=True)
        .stdout.decode()
        .strip()
    )


def get_package_version():
    return (
        run("python setup.py --version", shell=True, check=True, capture_output=True)
        .stdout.decode()
        .strip()
    )


def unique_name():
    return f"{get_date_string()}-{get_package_version()}-{get_head_rev_hash()}"


def run_spark(task, master="local[*]", spark_driver_memory="32g", print_only=False):
    Path("data/logging").mkdir(parents=True, exist_ok=True)
    cmd = " ".join(
        [
            "spark-submit",
            f"--master {master}",
            "--conf spark.pyspark.python=python",
            f"--conf spark.driver.memory={spark_driver_memory}",
            # use the local disk in the data directory for spill
            "--conf spark.local.dir=data/tmp/spark",
            # https://spark.apache.org/docs/latest/monitoring.html#viewing-after-the-fact
            "--conf spark.eventLog.enabled=true",
            "--conf spark.eventLog.dir=data/logging",
            # https://stackoverflow.com/a/46897622
            "--conf spark.ui.showConsoleProgress=true",
            "--conf spark.sql.execution.arrow.pyspark.enabled=true",
            f"--py-files {get_wheel()}",
            "main.py",
            task,
        ]
    )
    print(cmd)
    if print_only:
        return None
    return run(cmd, shell=True, check=True)
