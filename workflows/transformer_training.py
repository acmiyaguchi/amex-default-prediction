from pathlib import Path

import click

from .utils import build_wheel, run_spark, unique_name


@click.command()
@click.option("--print-only/--no-print-only", default=False)
def main(print_only):
    data_root = Path("data")
    intermediate_root = data_root / "intermediate"
    build_wheel()

    d_model = 64
    d_embed = 128
    sequence_length = 8
    max_position = 24
    batch_size = 4000
    layers = 1
    nhead = 1
    dropout = 0.1
    age_months = True
    pca = True
    epochs = 10
    # predict_reverse = True

    model_name = "torch-transformer-embedding"
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
                "--d-model",
                str(d_model),
                "--d-embed",
                str(d_embed),
                "--sequence-length",
                str(sequence_length),
                "--max-position",
                str(max_position),
                "--batch-size",
                str(batch_size),
                "--layers",
                str(layers),
                "--age-months" if age_months else "",
                "--nhead",
                str(nhead),
                "--dropout",
                str(dropout),
                "--pca" if pca else "--no-pca",
                "--epochs",
                str(epochs),
            ]
        ),
        spark_driver_memory="30g",
        print_only=print_only,
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
                "--batch-size",
                str(20_000),
                "--age-months" if age_months else "",
                "--pca" if pca else "--no-pca",
            ]
        ),
        spark_driver_memory="30g",
        print_only=print_only,
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
        print_only=print_only,
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
        print_only=print_only,
    )


if __name__ == "__main__":
    main()
