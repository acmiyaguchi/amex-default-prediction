import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from amex_default_prediction.utils import spark_session

from .data_module import (
    ArrowDataModule,
    PetastormDataModule,
    get_parquet_feature_size,
    get_spark_feature_size,
)
from .net import StrawmanNet


@click.command()
@click.argument("train_data_preprocessed_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--train-ratio", default=0.8, type=float)
@click.option("--cache-dir", default="file:///tmp")
@click.option("--batch-size", default=512, type=int)
@click.option(
    "--data-module", default="petastorm", type=click.Choice(["petastorm", "arrow"])
)
def fit_strawman(
    train_data_preprocessed_path,
    output_path,
    train_ratio,
    cache_dir,
    batch_size,
    data_module,
):
    spark = spark_session()

    if data_module == "petastorm":
        input_size = get_spark_feature_size(spark, train_data_preprocessed_path)
    elif data_module == "arrow":
        input_size = get_parquet_feature_size(train_data_preprocessed_path)

    model = StrawmanNet(input_size=input_size)
    print(model)

    if data_module == "petastorm":
        dm = PetastormDataModule(
            spark,
            cache_dir,
            train_data_preprocessed_path,
            train_ratio=train_ratio,
            batch_size=batch_size,
        )
    elif data_module == "arrow":
        dm = ArrowDataModule(
            train_data_preprocessed_path,
            train_ratio=train_ratio,
            batch_size=batch_size,
            num_workers=8,
        )

    trainer = pl.Trainer(
        gpus=-1,
        default_root_dir=output_path,
        detect_anomaly=True,
        logger=TensorBoardLogger(output_path, log_graph=True),
        reload_dataloaders_every_n_epochs=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min"),
            ModelCheckpoint(monitor="val_loss", auto_insert_metric_name=True),
        ],
    )
    trainer.fit(model, dm)
