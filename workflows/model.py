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
    run_spark(
        " ".join(
            [
                "fit",
                model_name,
                (
                    intermediate_root
                    / (
                        "test_data_preprocessed_v4"
                        if model_name == "pca"
                        else "train_data_preprocessed_v4"
                    )
                ).as_posix(),
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
                    if "nn-transformer" == model_name
                    else []
                ),
                *(
                    [
                        (
                            intermediate_root
                            / "models/aft/20220718061207-0.12.0-2d69426"
                        ).as_posix()
                    ]
                    if "with-aft" in model_name
                    else []
                ),
                *(
                    [
                        (
                            intermediate_root
                            / "models/pca/20220723073653-0.15.2-c5aeb38"
                        ).as_posix()
                    ]
                    if "with-pca" in model_name
                    else []
                ),
                output.as_posix(),
            ]
        ),
    )


if __name__ == "__main__":
    main()
