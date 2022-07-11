import click

from .model import fm, gbt, logistic
from .transform import transform


@click.group()
def cli():
    pass


@click.group()
def fit():
    pass


fit.add_command(logistic.fit, "logistic")
fit.add_command(gbt.fit, "gbt")
fit.add_command(fm.fit, "fm")

for command in [transform, fit]:
    cli.add_command(command)
