from pathlib import Path

import click

from .utils import build_wheel, run_spark, unique_name


@click.command()
@click.argument("model_name", type=click.Choice(["logistic", "fm"]))
def main(model_name):
    data_root = Path("data")
    intermediate_root = data_root / "intermediate"
    build_wheel()

    output = intermediate_root / "models" / model_name / unique_name()
    run_spark(
        " ".join(
            [
                model_name,
                "fit",
                (intermediate_root / "train_data_preprocessed").as_posix(),
                output.as_posix(),
            ]
        ),
    )


if __name__ == "__main__":
    main()
