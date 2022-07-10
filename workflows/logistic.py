from pathlib import Path

import click

from .utils import build_wheel, run_spark


@click.command()
@click.option("--overwrite/--no-overwrite", default=False)
def main(overwrite):
    data_root = Path("data")
    intermediate_root = data_root / "intermediate"
    build_wheel()

    output = intermediate_root / "models" / "logistic"
    run_spark(
        " ".join(
            [
                "logistic",
                "fit",
                (intermediate_root / "train_data_preprocessed").as_posix(),
                output.as_posix(),
            ]
        ),
        print_only=output.exists() and not overwrite,
    )


if __name__ == "__main__":
    main()
