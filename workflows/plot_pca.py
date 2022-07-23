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
                "plot-pca",
                (
                    intermediate_root / "models/pca/20220723062755-0.15.1-a8e1240"
                ).as_posix(),
                (intermediate_root / "train_data_preprocessed_v3").as_posix(),
            ]
        ),
    )


if __name__ == "__main__":
    main()
