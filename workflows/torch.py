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
    dataset = (
        "test_data_preprocessed_v4"
        if "torch-transformer" == model_name
        else "train_data_preprocessed_v4"
    )
    spark_driver_memory = "20g"

    run_spark(
        " ".join(
            [
                "fit",
                model_name,
                (intermediate_root / dataset).as_posix(),
                *(
                    [
                        (
                            intermediate_root
                            / "models/pca/20220723073653-0.15.2-c5aeb38"
                        ).as_posix(),
                    ]
                    if "transformer" in model_name
                    else []
                ),
                *(
                    [
                        (
                            intermediate_root
                            / (
                                "models/torch-transformer/"
                                "20220810060736-0.17.0-36d978d/model.ckpt"
                            )
                        ).as_posix()
                    ]
                    if "torch-transform-transformer" == model_name
                    else []
                ),
                output.as_posix(),
                "--sequence-length",
                "16",
                "--max-position",
                "512",
                "--batch-size",
                "1750",
            ]
        ),
        spark_driver_memory=spark_driver_memory,
    )


if __name__ == "__main__":
    main()
