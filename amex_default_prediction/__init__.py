import click

from .model.fm import fm
from .model.gbt import gbt
from .model.logistic import logistic
from .transform import transform


@click.group()
def cli():
    pass


for command in [transform, logistic, fm, gbt]:
    cli.add_command(command)
