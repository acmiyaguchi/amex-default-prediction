import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from amex_default_prediction.model.base import read_train_data
from amex_default_prediction.utils import spark_session

from .data_module import PetastormDataModule
from .net import StrawmanNet


@click.command()
@click.argument("train_data_preprocessed_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--train-ratio", default=0.8, type=float)
@click.option("--cache-dir", default="file:///tmp")
@click.option("--batch-size", default=512, type=int)
def fit_strawman(
    train_data_preprocessed_path, output_path, train_ratio, cache_dir, batch_size
):
    spark = spark_session()

    # get the input size for the model
    df, _, _ = read_train_data(spark, train_data_preprocessed_path, cache=False)
    input_size = df.head().features.size
    model = StrawmanNet(input_size=input_size)
    print(model)

    data_module = PetastormDataModule(
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
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min"),
            ModelCheckpoint(monitor="val_loss", auto_insert_metric_name=True),
        ],
    )
    trainer.fit(model, data_module)
