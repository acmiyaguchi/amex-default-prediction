from pathlib import Path

import click

from .utils import build_wheel, run_spark, unique_name


@click.command()
def main():
    data_root = Path("data")
    intermediate_root = data_root / "intermediate"
    build_wheel()

    sequence_length = 8
    max_position = 24
    batch_size = 1750
    layers = 3
    age_months = True

    model_name = "torch-transformer"
    torch_transformer_output = intermediate_root / "models" / model_name / unique_name()
    run_spark(
        " ".join(
            [
                "fit",
                model_name,
                (intermediate_root / "test_data_preprocessed_v4").as_posix(),
                (
                    intermediate_root / "models/pca/20220723073653-0.15.2-c5aeb38"
                ).as_posix(),
                torch_transformer_output.as_posix(),
                "--sequence-length",
                str(sequence_length),
                "--max-position",
                str(max_position),
                "--batch-size",
                str(batch_size),
                "--layers",
                str(layers),
                "--age-months" if age_months else "",
            ]
        ),
        spark_driver_memory="20g",
    )

    model_name = "torch-transform-transformer"
    torch_transform_transformer_output = (
        intermediate_root / "models" / model_name / unique_name()
    )
    run_spark(
        " ".join(
            [
                "fit",
                model_name,
                (intermediate_root / "train_data_preprocessed_v4").as_posix(),
                (
                    intermediate_root / "models/pca/20220723073653-0.15.2-c5aeb38"
                ).as_posix(),
                (torch_transformer_output / "model.ckpt").as_posix(),
                torch_transform_transformer_output.as_posix(),
                "--sequence-length",
                str(sequence_length),
                "--max-position",
                str(max_position),
                "--batch-size",
                str(batch_size),
                "--age-months" if age_months else "",
            ]
        ),
        spark_driver_memory="20g",
    )

    run_spark(
        " ".join(
            [
                "plot",
                "plot-transformer",
                (
                    intermediate_root / "models/pca/20220723073653-0.15.2-c5aeb38"
                ).as_posix(),
                (intermediate_root / "train_data_preprocessed_v4").as_posix(),
                torch_transform_transformer_output.as_posix(),
                "--output-path",
                (intermediate_root / "plot-transformer" / unique_name()).as_posix(),
            ]
        ),
        spark_driver_memory="10g",
    )

    model_name = "logistic-with-transformer"
    output = intermediate_root / "models" / model_name / unique_name()
    run_spark(
        " ".join(
            [
                "fit",
                model_name,
                (intermediate_root / "train_data_preprocessed_v4").as_posix(),
                torch_transform_transformer_output.as_posix(),
                output.as_posix(),
            ]
        ),
    )


if __name__ == "__main__":
    main()
