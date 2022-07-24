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
    create_transformer_pair,
    get_spark_feature_size,
    transform_into_transformer_pairs,
)
from amex_default_prediction.torch.net import StrawmanNet


def test_get_parquet_feature_size(synthetic_train_data_path):
    feature_size = get_spark_feature_size(synthetic_train_data_path)
    assert feature_size == 3


def test_petastorm_data_module_has_fields(spark, synthetic_train_data_path):
    data_module = PetastormDataModule(spark, "file:///tmp", synthetic_train_data_path)
    data_module.setup()
    dataloader = data_module.train_dataloader()
    batches = 0
    for batch in dataloader:
        batches += 1
        assert set(batch.keys()) == {"features", "label"}
        assert isinstance(batch["features"], torch.Tensor)
        assert isinstance(batch["label"], torch.Tensor)
        assert batch["features"].shape == torch.Size([32, 3])
        assert batch["label"].shape == torch.Size([32])
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
    num_features = 3
    max_seen = 8
    rows = []
    for i in range(num_customers):
        customer_id = str(uuid4())
        # create sequences between 2 and max_seen
        for j in range(max((i % max_seen) + 1, 2)):
            features = np.random.rand(num_features)
            rows.append(dict(customer_ID=customer_id, features=features, age_days=j))
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_transformer_train_df(spark, synthetic_transformer_train_pdf):
    df = (
        transform_into_transformer_pairs(
            spark.createDataFrame(synthetic_transformer_train_pdf).withColumn(
                "features", mlF.array_to_vector("features")
            ),
            length=4,
        )
        .withColumn("sample_id", F.crc32(F.col("customer_ID")) % 100)
        .repartition(1)
    )
    df.printSchema()
    return df


def test_test_create_transformer_pair(synthetic_transformer_train_pdf):
    pdf = synthetic_transformer_train_pdf
    for key in pdf.customer_ID.unique():
        length = 4
        res = create_transformer_pair(pdf[pdf.customer_ID == key], length)
        assert res.shape[0] == 1
        print(res)
        row = res.iloc[0]
        assert len(row.src) > 0
        assert len(row.src[0]) == 3
        assert len(row.tgt) > 0
        assert len(row.src_key_padding_mask) == length
        assert len(row.tgt_key_padding_mask) == length


def test_synthetic_transformer_train_df(synthetic_transformer_train_df):
    df = synthetic_transformer_train_df.cache()
    assert df.count() == 40
    pdf = df.toPandas()
    assert len(pdf.iloc[0].src) == 4
    df.unpersist()
