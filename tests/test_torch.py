from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
import pytorch_lightning as pl
import torch
from pyspark.ml import functions as mlF
from pyspark.sql import functions as F

from amex_default_prediction.torch.data_module import (
    PetastormDataModule,
    PetastormTransformerDataModule,
    create_transformer_pair,
    get_spark_feature_size,
    transform_into_transformer_pairs,
)
from amex_default_prediction.torch.net import StrawmanNet, TransformerModel


def test_get_parquet_feature_size(spark, synthetic_train_data_path):
    feature_size = get_spark_feature_size(spark, synthetic_train_data_path)
    assert feature_size == 3


def test_petastorm_data_module_has_fields(spark, synthetic_train_data_path):
    batch_size = 10
    data_module = PetastormDataModule(
        spark, "file:///tmp", synthetic_train_data_path, batch_size=batch_size
    )
    data_module.setup()
    dataloader = data_module.train_dataloader()
    batches = 0
    for batch in dataloader:
        batches += 1
        assert set(batch.keys()) == {"features", "label"}
        assert isinstance(batch["features"], torch.Tensor)
        assert isinstance(batch["label"], torch.Tensor)
        assert batch["features"].shape == torch.Size([batch_size, 3])
        assert batch["label"].shape == torch.Size([batch_size])
        break
    assert batches == 1


def test_trainer_accepts_petastorm_data_module(spark, synthetic_train_data_path):
    data_module = PetastormDataModule(spark, "file:///tmp", synthetic_train_data_path)
    trainer = pl.Trainer(fast_dev_run=True)
    model = StrawmanNet(input_size=3)
    trainer.fit(model, datamodule=data_module)


@pytest.fixture
def synthetic_transformer_train_pdf():
    num_customers = 40
    num_features = 8
    max_seen = 8
    rows = []
    for i in range(num_customers):
        customer_id = str(uuid4())
        # create sequences between 2 and max_seen
        for j in range(max((i % max_seen) + 1, 2)):
            features = np.random.rand(num_features)
            rows.append(
                dict(
                    customer_ID=customer_id,
                    features=features,
                    age_days=j,
                    most_recent=False,
                )
            )
        rows[-1]["most_recent"] = True
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_transformer_train_df_path(
    spark, synthetic_transformer_train_pdf, tmp_path
):
    df = (
        spark.createDataFrame(synthetic_transformer_train_pdf)
        .withColumn("features", mlF.array_to_vector("features"))
        .withColumn("sample_id", F.crc32(F.col("customer_ID")) % 100)
    ).repartition(4)

    output = tmp_path / "test_data"
    df.write.parquet((output / "data").as_posix())
    yield output


def test_test_create_transformer_pair(synthetic_transformer_train_pdf):
    pdf = synthetic_transformer_train_pdf
    for key in pdf.customer_ID.unique():
        length = 4
        res = create_transformer_pair(pdf[pdf.customer_ID == key], length)
        assert res.shape[0] == 1
        print(res)
        row = res.iloc[0]
        assert len(row.src) > 0
        assert len(row.src[0]) == 8
        assert len(row.tgt) > 0
        assert len(row.src_key_padding_mask) == length
        assert len(row.tgt_key_padding_mask) == length


def test_synthetic_transformer_train_df(spark, synthetic_transformer_train_pdf):
    df = (
        transform_into_transformer_pairs(
            spark.createDataFrame(synthetic_transformer_train_pdf).withColumn(
                "features", mlF.array_to_vector("features")
            ),
            length=4,
        ).repartition(1)
    ).cache()
    df.printSchema()

    df.show(vertical=True, truncate=80)
    assert df.count() == 40

    pdf = df.select((F.size("src") / F.lit(8)).alias("precondition")).toPandas()
    assert (pdf.precondition != 4).sum() == 0

    pdf = df.select((F.size("tgt") / F.lit(8)).alias("precondition")).toPandas()
    assert (pdf.precondition != 4).sum() == 0

    pdf = df.select(
        (F.size("src_key_padding_mask") + F.size("tgt_key_padding_mask")).alias(
            "precondition"
        )
    ).toPandas()
    assert (pdf.precondition != 8).sum() == 0

    pdf = df.select(
        (F.size("src_pos") + F.size("tgt_pos")).alias("precondition")
    ).toPandas()

    assert (pdf.precondition != 8).sum() == 0
    df.unpersist()


def test_petastorm_transformer_data_module_has_fields(
    spark, synthetic_transformer_train_df_path
):
    batch_size = 10
    subsequence_length = 4
    data_module = PetastormTransformerDataModule(
        spark,
        "file:///tmp",
        synthetic_transformer_train_df_path,
        batch_size=batch_size,
        subsequence_length=subsequence_length,
    )
    data_module.setup()
    dataloader = data_module.train_dataloader()
    batches = 0
    for batch in dataloader:
        batches += 1
        assert set(batch.keys()) == {
            "src",
            "tgt",
            "src_key_padding_mask",
            "tgt_key_padding_mask",
            "src_pos",
            "tgt_pos",
            "subsequence_length",
        }
        assert isinstance(batch["src"], torch.Tensor)
        assert isinstance(batch["tgt"], torch.Tensor)
        assert (
            batch["src"].shape
            == batch["tgt"].shape
            == torch.Size([batch_size, 8 * subsequence_length])
        )
        # check subsequence length
        assert batch["subsequence_length"][0] == subsequence_length
        break
    assert batches == 1


def test_transformer_trainer_accepts_petastorm_transformer_data_module(
    spark, synthetic_transformer_train_df_path
):
    batch_size = 10
    subsequence_length = 4
    data_module = PetastormTransformerDataModule(
        spark,
        "file:///tmp",
        synthetic_transformer_train_df_path,
        batch_size=batch_size,
        subsequence_length=subsequence_length,
    )
    trainer = pl.Trainer(fast_dev_run=True)
    model = TransformerModel(d_model=8)
    trainer.fit(model, datamodule=data_module)
