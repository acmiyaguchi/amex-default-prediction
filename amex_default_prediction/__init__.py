import click

from .model import aft, fm, gbt, logistic
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
fit.add_command(aft.fit, "aft")
fit.add_command(logistic.fit_with_aft, "logistic-with-aft")

for command in [transform, fit]:
    cli.add_command(command)
