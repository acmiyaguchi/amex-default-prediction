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
                        ).as_posix()
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
                                "20220725054744-0.16.2-6d73fff/"
                                "lightning_logs_amex-default-prediction/0_21aakrzj/"
                                "checkpoints/epoch=8-step=1656.ckpt"
                            )
                        ).as_posix()
                    ]
                    if "torch-transform-transformer" == model_name
                    else []
                ),
                output.as_posix(),
            ]
        ),
        spark_driver_memory=spark_driver_memory,
    )


if __name__ == "__main__":
    main()
