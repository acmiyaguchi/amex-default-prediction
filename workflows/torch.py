from pathlib import Path

import click

from .utils import build_wheel, run_spark, unique_name


@click.command()
@click.argument("model_name", type=click.Choice(["torch-strawman"]))
@click.option(
    "--data-module", default="petastorm", type=click.Choice(["petastorm", "arrow"])
)
def main(model_name, data_module):
    data_root = Path("data")
    intermediate_root = data_root / "intermediate"
    build_wheel()

    output = intermediate_root / "models" / model_name / unique_name()
    if data_module == "petastorm":
        dataset = "train_data_preprocessed_v2"
        spark_driver_memory = "40g"
    elif data_module == "arrow":
        dataset = "train_data_preprocessed_torch_v2"
        spark_driver_memory = "10g"

    run_spark(
        " ".join(
            [
                "fit",
                model_name,
                (intermediate_root / dataset).as_posix(),
                output.as_posix(),
                f"--data-module {data_module}",
            ]
        ),
        spark_driver_memory=spark_driver_memory,
    )


if __name__ == "__main__":
    main()
