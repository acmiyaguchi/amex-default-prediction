import click

from .model.fm import fm
from .model.logistic import logistic
from .transform import transform


@click.group()
def cli():
    pass


for command in [transform, logistic, fm]:
    cli.add_command(command)
