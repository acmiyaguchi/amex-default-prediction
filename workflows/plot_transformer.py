from pathlib import Path

import click

from .utils import build_wheel, run_spark


@click.command()
def main():
    data_root = Path("data")
    intermediate_root = data_root / "intermediate"
    build_wheel()

    run_spark(
        " ".join(
            [
                "plot",
                "plot-transformer",
                (
                    intermediate_root / "models/pca/20220723073653-0.15.2-c5aeb38"
                ).as_posix(),
                (intermediate_root / "train_data_preprocessed_v4").as_posix(),
                (
                    intermediate_root
                    / "models/torch-transform-transformer/20220810213247-0.17.1-53636bc"
                ).as_posix(),
            ]
        ),
        spark_driver_memory="10g",
    )


if __name__ == "__main__":
    main()
