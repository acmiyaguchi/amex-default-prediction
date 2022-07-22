import click

from .model import aft, gbt, logistic, nn
from .torch import cmd
from .transform import transform_group


@click.group()
def cli():
    pass


@click.group()
def fit():
    pass


fit.add_command(logistic.fit, "logistic")
fit.add_command(gbt.fit, "gbt")
fit.add_command(aft.fit, "aft")
fit.add_command(logistic.fit_with_aft, "logistic-with-aft")
fit.add_command(gbt.fit_with_aft, "gbt-with-aft")
fit.add_command(nn.fit, "nn")
fit.add_command(cmd.fit_strawman, "torch-strawman")


for command in [transform_group, fit]:
    cli.add_command(command)
