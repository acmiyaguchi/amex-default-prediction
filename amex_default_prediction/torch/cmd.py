from multiprocessing import Pool
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
from .net import StrawmanNet, TransformerEmbeddingModel, TransformerModel


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
        accelerator="gpu",
        devices=-1,
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
@click.option("--d-model", default=128, type=int)
@click.option("--d-embed", default=128, type=int)
@click.option("--sequence-length", default=16, type=int)
@click.option("--max-position", default=1024, type=int)
@click.option("--max-position", default=1024, type=int)
@click.option("--layers", default=6, type=int)
@click.option("--age-months/--no-age-months", default=False, type=bool)
@click.option("--tune", default=False, type=bool)
def fit_transformer_embedding(
    test_data_preprocessed_path,
    pca_model_path,
    output_path,
    train_ratio,
    cache_dir,
    batch_size,
    d_model,
    d_embed,
    sequence_length,
    max_position,
    layers,
    age_months,
    tune,
):
    spark = spark_session()
    input_size = get_spark_feature_size(
        spark, test_data_preprocessed_path, pca_model_path
    )
    model = TransformerEmbeddingModel(
        d_input=input_size,
        d_model=d_model,
        d_embed=d_embed,
        seq_len=sequence_length,
        max_len=max_position,
        num_layers=layers,
        lr=1e-3,
        warmup=100,
        max_iters=2_000,
    )
    # print(model)

    wandb_logger = WandbLogger(
        project="amex-default-prediction",
        name=Path(output_path).name,
        save_dir=output_path,
    )

    wandb_logger.experiment.config.update(
        {
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "age_months": age_months,
            "tune": tune,
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
        age_months=age_months,
        predict_reverse=True,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        **(
            dict(
                auto_lr_find=True,
                auto_scale_batch_size="binsearch",
            )
            if tune
            else {}
        ),
        default_root_dir=output_path,
        detect_anomaly=True,
        logger=[
            TensorBoardLogger(output_path, log_graph=True),
            wandb_logger,
        ],
        reload_dataloaders_every_n_epochs=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min"),
            ModelCheckpoint(
                monitor="val_loss",
                auto_insert_metric_name=True,
                save_top_k=5,
            ),
            ModelCheckpoint(dirpath=output_path, filename="model", monitor="val_loss"),
        ],
        max_epochs=30,
    )
    if tune:
        # optimize lr and batch size
        trainer.tune(model, datamodule=dm)

    trainer.fit(model, datamodule=dm)


@click.command()
@click.argument("test_data_preprocessed_path", type=click.Path(exists=True))
@click.argument("pca_model_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--train-ratio", default=0.8, type=float)
@click.option("--cache-dir", default="file:///tmp")
@click.option("--batch-size", default=4000, type=int)
@click.option("--d-model", default=64, type=int)
@click.option("--sequence-length", default=8, type=int)
@click.option("--max-position", default=1024, type=int)
@click.option("--max-position", default=1024, type=int)
@click.option("--layers", default=6, type=int)
@click.option("--age-months/--no-age-months", default=False, type=bool)
@click.option("--predict-reverse/--no-predict-reverse", default=False, type=bool)
@click.option("--tune", default=False, type=bool)
def fit_transformer(
    test_data_preprocessed_path,
    pca_model_path,
    output_path,
    train_ratio,
    cache_dir,
    batch_size,
    d_model,
    sequence_length,
    max_position,
    layers,
    age_months,
    predict_reverse,
    tune,
):
    spark = spark_session()
    input_size = get_spark_feature_size(
        spark, test_data_preprocessed_path, pca_model_path
    )
    model = TransformerModel(
        d_input=input_size,
        d_model=d_model,
        max_len=max_position,
        num_encoder_layers=layers,
        num_decoder_layers=layers,
        lr=1e-3,
        warmup=100,
        max_iters=2_000,
    )
    # print(model)

    wandb_logger = WandbLogger(
        project="amex-default-prediction",
        name=Path(output_path).name,
        save_dir=output_path,
    )
    wandb_logger.experiment.config.update(
        {
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "age_months": age_months,
            "predict_reverse": predict_reverse,
            "tune": tune,
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
        age_months=age_months,
        predict_reverse=predict_reverse,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        **(
            dict(
                auto_lr_find=True,
                auto_scale_batch_size="binsearch",
            )
            if tune
            else {}
        ),
        default_root_dir=output_path,
        detect_anomaly=True,
        logger=[
            TensorBoardLogger(output_path, log_graph=True),
            wandb_logger,
        ],
        reload_dataloaders_every_n_epochs=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min"),
            ModelCheckpoint(
                monitor="val_loss",
                auto_insert_metric_name=True,
                save_top_k=5,
            ),
            ModelCheckpoint(dirpath=output_path, filename="model", monitor="val_loss"),
        ],
        max_epochs=10,
    )
    if tune:
        # optimize lr and batch size
        trainer.tune(model, datamodule=dm)

    trainer.fit(model, datamodule=dm)


@click.command()
@click.argument("train_data_preprocessed_path", type=click.Path(exists=True))
@click.argument("pca_model_path", type=click.Path(exists=True))
@click.argument("checkpoint_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--cache-dir", default="file:///tmp")
@click.option("--batch-size", default=4000, type=int)
@click.option("--sequence-length", default=8, type=int)
@click.option("--age-months/--no-age-months", default=False, type=bool)
def transform_transformer(
    train_data_preprocessed_path,
    pca_model_path,
    checkpoint_path,
    output_path,
    cache_dir,
    batch_size,
    sequence_length,
    age_months,
    **kwargs,
):
    spark = spark_session()
    model = TransformerEmbeddingModel.load_from_checkpoint(checkpoint_path)
    model.freeze()

    dm = PetastormTransformerDataModule(
        spark,
        cache_dir,
        train_data_preprocessed_path,
        pca_model_path=pca_model_path,
        subsequence_length=sequence_length,
        batch_size=batch_size,
        age_months=age_months,
    )
    dm.setup()

    Path(output_path).mkdir(parents=True, exist_ok=True)

    # NOTE: this loop is very slow, I have no idea how to make this faster at
    # the moment...
    trainer = pl.Trainer(accelerator="gpu", devices=-1)

    for batch_idx, prediction in enumerate(trainer.predict(model, datamodule=dm)):
        df = pd.DataFrame(
            {k: v.cpu().detach().numpy().tolist() for k, v in prediction.items()}
        )
        df.to_parquet(Path(output_path) / f"part_{batch_idx:05}.parquet", index=False)
