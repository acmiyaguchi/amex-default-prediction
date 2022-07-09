from pathlib import Path

import click

from .utils import build_wheel, run_spark


@click.command()
@click.option("--print-only/--no-print-only", default=False)
def main(print_only):
    data_root = Path("data")
    raw_root = data_root / "raw"
    intermediate_root = data_root / "intermediate"
    build_wheel()
    for file in raw_root.glob("**/*.csv"):
        output = intermediate_root / file.name.replace(".csv", "")
        run_spark(
            f"transform raw-to-parquet {file} {output} --num-partitions 256",
            print_only=print_only,
        )


if __name__ == "__main__":
    main()
