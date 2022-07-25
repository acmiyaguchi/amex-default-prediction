from pathlib import Path

import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from amex_default_prediction.utils import spark_session

from .data_module import (
    PetastormDataModule,
    PetastormTransformerDataModule,
    get_spark_feature_size,
)
from .net import StrawmanNet, TransformerModel


@click.command()
@click.argument("train_data_preprocessed_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--train-ratio", default=0.8, type=float)
@click.option("--cache-dir", default="file:///tmp")
@click.option("--batch-size", default=128, type=int)
def fit_strawman(
    train_data_preprocessed_path,
    output_path,
    train_ratio,
    cache_dir,
    batch_size,
):
    spark = spark_session()
    input_size = get_spark_feature_size(spark, train_data_preprocessed_path)
    model = StrawmanNet(input_size=input_size)
    print(model)

    dm = PetastormDataModule(
        spark,
        cache_dir,
        train_data_preprocessed_path,
        train_ratio=train_ratio,
        batch_size=batch_size,
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


@click.command()
@click.argument("test_data_preprocessed_path", type=click.Path(exists=True))
@click.argument("pca_model_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--train-ratio", default=0.8, type=float)
@click.option("--cache-dir", default="file:///tmp")
@click.option("--batch-size", default=64, type=int)
def fit_transformer(
    test_data_preprocessed_path,
    pca_model_path,
    output_path,
    train_ratio,
    cache_dir,
    batch_size,
):
    spark = spark_session()
    input_size = get_spark_feature_size(
        spark, test_data_preprocessed_path, pca_model_path
    )
    model = TransformerModel(d_model=input_size)
    print(model)

    dm = PetastormTransformerDataModule(
        spark,
        cache_dir,
        test_data_preprocessed_path,
        pca_model_path=pca_model_path,
        subsequence_length=8,
        train_ratio=train_ratio,
        batch_size=batch_size,
    )

    trainer = pl.Trainer(
        gpus=-1,
        default_root_dir=output_path,
        detect_anomaly=True,
        logger=[
            TensorBoardLogger(output_path, log_graph=True),
            WandbLogger(
                project="amex-default-prediction",
                name=Path(output_path).name,
                save_dir=output_path,
            ),
        ],
        reload_dataloaders_every_n_epochs=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min"),
            ModelCheckpoint(monitor="val_loss", auto_insert_metric_name=True),
        ],
    )
    trainer.fit(model, dm)
