from pathlib import Path

import click

from .utils import build_wheel, run_spark, unique_name


@click.command()
@click.argument(
    "model_name",
    type=click.Choice(
        [
            "logistic",
            "gbt",
            "aft",
            "logistic-with-aft",
            "gbt-with-aft",
            "nn",
            "torch-strawman",
        ]
    ),
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
                (intermediate_root / "train_data_preprocessed_v2").as_posix(),
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
                output.as_posix(),
            ]
        ),
    )


if __name__ == "__main__":
    main()
