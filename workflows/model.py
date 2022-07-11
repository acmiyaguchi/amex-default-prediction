from pathlib import Path

import click

from .utils import build_wheel, run_spark, unique_name


@click.command()
@click.argument(
    "model_name",
    type=click.Choice(["logistic", "fm", "gbt", "aft", "logistic-with-aft"]),
)
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
                (intermediate_root / "train_data_preprocessed").as_posix(),
                *(
                    [
                        (
                            intermediate_root
                            / "models/aft/20220711044610-0.8.0-6bbdfec"
                        ).as_posix()
                    ]
                    if model_name == "logistic-with-aft"
                    else []
                ),
                output.as_posix(),
            ]
        ),
    )


if __name__ == "__main__":
    main()
