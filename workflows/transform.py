from pathlib import Path

import click

from .utils import build_wheel, run_spark


@click.command()
@click.option("--overwrite/--no-overwrite", default=False)
def main(overwrite):
    data_root = Path("data")
    raw_root = data_root / "raw"
    intermediate_root = data_root / "intermediate"
    build_wheel()
    for file in raw_root.glob("**/*.csv"):
        output = intermediate_root / file.name.replace(".csv", "")
        print_only = output.exists() and not overwrite
        run_spark(
            f"transform raw-to-parquet {file} {output} --num-partitions 256",
            print_only=print_only,
        )

    output = intermediate_root / "test_data_preprocessed_v2"
    run_spark(
        " ".join(
            [
                "transform",
                "preprocess-testing-dataset",
                (intermediate_root / "test_data").as_posix(),
                output.as_posix(),
            ]
        ),
        print_only=output.exists() and not overwrite,
    )

    output = intermediate_root / "train_data_preprocessed_v3"
    run_spark(
        " ".join(
            [
                "transform",
                "preprocess-training-dataset-with-pipeline",
                (intermediate_root / "test_data_preprocessed_v2").as_posix(),
                (intermediate_root / "train_data").as_posix(),
                (intermediate_root / "train_labels").as_posix(),
                output.as_posix(),
            ]
        ),
        print_only=output.exists() and not overwrite,
    )


if __name__ == "__main__":
    main()
