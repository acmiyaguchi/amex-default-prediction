from pathlib import Path

import click

from .utils import build_wheel, run_spark, unique_name


@click.command()
@click.argument("model_name", type=str)
def main(model_name):
    data_root = Path("data")
    intermediate_root = data_root / "intermediate"
    build_wheel()

    output = intermediate_root / "models" / model_name / unique_name()
    dataset = "train_data_preprocessed_v2"
    spark_driver_memory = "20g"

    run_spark(
        " ".join(
            [
                "fit",
                model_name,
                (intermediate_root / dataset).as_posix(),
                output.as_posix(),
            ]
        ),
        spark_driver_memory=spark_driver_memory,
    )


if __name__ == "__main__":
    main()
