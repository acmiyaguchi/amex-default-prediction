from pathlib import Path

import click
import pandas as pd
import pytorch_lightning as pl
import tqdm
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
@click.option("--batch-size", default=4000, type=int)
@click.option("--sequence-length", default=8, type=int)
@click.option("--max-position", default=1024, type=int)
@click.option("--max-position", default=1024, type=int)
@click.option("--layers", default=6, type=int)
def fit_transformer(
    test_data_preprocessed_path,
    pca_model_path,
    output_path,
    train_ratio,
    cache_dir,
    batch_size,
    sequence_length,
    max_position,
    layers,
):
    spark = spark_session()
    input_size = get_spark_feature_size(
        spark, test_data_preprocessed_path, pca_model_path
    )
    model = TransformerModel(
        d_model=input_size,
        max_len=max_position,
        num_encoder_layers=layers,
        num_decoder_layers=layers,
    )
    print(model)

    wandb_logger = WandbLogger(
        project="amex-default-prediction",
        name=Path(output_path).name,
        save_dir=output_path,
    )
    wandb_logger.experiment.config.update(
        {
            "sequence_length": sequence_length,
            "batch_size": batch_size,
        }
    )

    dm = PetastormTransformerDataModule(
        spark,
        cache_dir,
        test_data_preprocessed_path,
        pca_model_path=pca_model_path,
        subsequence_length=sequence_length,
        train_ratio=train_ratio,
        batch_size=batch_size,
    )

    trainer = pl.Trainer(
        gpus=-1,
        default_root_dir=output_path,
        detect_anomaly=True,
        logger=[
            TensorBoardLogger(output_path, log_graph=True),
            wandb_logger,
        ],
        reload_dataloaders_every_n_epochs=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min"),
            ModelCheckpoint(monitor="val_loss", auto_insert_metric_name=True),
        ],
    )
    trainer.fit(model, dm)
    trainer.save_checkpoint(f"{output_path}/model.ckpt")
    print(f"wrote checkpoint {output_path}/model.ckpt")


@click.command()
@click.argument("train_data_preprocessed_path", type=click.Path(exists=True))
@click.argument("pca_model_path", type=click.Path(exists=True))
@click.argument("checkpoint_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--cache-dir", default="file:///tmp")
@click.option("--batch-size", default=4000, type=int)
@click.option("--sequence-length", default=8, type=int)
@click.option("--max-position", default=1024, type=int)
def transform_transformer(
    train_data_preprocessed_path,
    pca_model_path,
    checkpoint_path,
    output_path,
    cache_dir,
    batch_size,
    sequence_length,
    max_position,
):
    spark = spark_session()
    input_size = get_spark_feature_size(
        spark, train_data_preprocessed_path, pca_model_path
    )
    model = TransformerModel.load_from_checkpoint(
        checkpoint_path, d_model=input_size, max_len=max_position
    )
    print(model)

    dm = PetastormTransformerDataModule(
        spark,
        cache_dir,
        train_data_preprocessed_path,
        pca_model_path=pca_model_path,
        subsequence_length=sequence_length,
        batch_size=batch_size,
    )
    dm.setup()

    Path(output_path).mkdir(parents=True, exist_ok=True)

    # NOTE: this loop is very slow, I have no idea how to make this faster at
    # the moment...
    acc = []
    n_batches = 10
    for batch_idx, batch in tqdm.tqdm(enumerate(dm.predict_dataloader())):
        cidx = batch["customer_index"].cpu().detach().numpy()
        z = model.predict_step(batch, batch_idx).cpu().detach().numpy()
        # NOTE: requires transposing data
        # data = z.reshape(z.shape[0], -1)
        data = z[0]
        df = pd.DataFrame(
            zip(cidx, data),
            columns=["customer_index", "prediction"],
        )
        acc.append(df)
        if len(acc) > n_batches:
            df = pd.concat(acc)
            df.to_parquet(
                Path(output_path) / f"part_{batch_idx:05}.parquet", index=False
            )
            acc = []
    if acc:
        df = pd.concat(acc)
        df.to_parquet(Path(output_path) / f"part_{batch_idx:05}.parquet", index=False)
