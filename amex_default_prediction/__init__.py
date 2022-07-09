import click

from .transform import transform


@click.group()
def cli():
    pass


cli.add_command(transform)
