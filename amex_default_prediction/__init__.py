import click

from .model.logistic import logistic
from .transform import transform


@click.group()
def cli():
    pass


for command in [transform, logistic]:
    cli.add_command(command)
