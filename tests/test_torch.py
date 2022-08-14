from typing import Iterator
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
import pytorch_lightning as pl
import torch
from pyspark.ml import functions as mlF
from pyspark.sql import functions as F

from amex_default_prediction.model.base import TransformerInferenceTransformer
from amex_default_prediction.torch.data_module import (
    PetastormDataModule,
    PetastormTransformerDataModule,
    get_spark_feature_size,
)
from amex_default_prediction.torch.net import StrawmanNet, TransformerModel
from amex_default_prediction.torch.transform import (
    transform_into_transformer_pairs,
    transform_into_transformer_predict_pairs,
    transform_into_transformer_reverse_pairs,
)


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


def test_transform_into_transformer_pairs(spark, synthetic_transformer_train_pdf):
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


def test_transform_into_transformer_reverse_pairs(
    spark, synthetic_transformer_train_pdf
):
    df = (
        transform_into_transformer_reverse_pairs(
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


def test_transform_into_transformer_predict_pairs(
    spark, synthetic_transformer_train_pdf
):
    df = (
        transform_into_transformer_predict_pairs(
            spark.createDataFrame(synthetic_transformer_train_pdf).withColumn(
                "features", mlF.array_to_vector("features")
            ),
            length=4,
        ).repartition(1)
    ).cache()
    df.printSchema()

    df.show(vertical=True, truncate=80)
    assert df.count() == 40
    assert df.columns == ["customer_ID", "src", "src_key_padding_mask", "src_pos"]

    for k, v in [["src", 8 * 4], ["src_key_padding_mask", 4], ["src_pos", 4]]:
        pdf = df.select((F.size(k)).alias("precondition")).toPandas()
        assert (pdf.precondition != v).sum() == 0


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
        }
        assert isinstance(batch["src"], torch.Tensor)
        assert isinstance(batch["tgt"], torch.Tensor)
        assert (
            batch["src"].shape
            == batch["tgt"].shape
            == torch.Size([batch_size, 8 * subsequence_length])
        )
        assert batch["src_key_padding_mask"][0].type(torch.bool).dtype == torch.bool
        assert batch["src_key_padding_mask"].cpu().detach().numpy().sum() > 0
        assert batch["tgt_key_padding_mask"].cpu().detach().numpy().sum() > 0
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
        num_partitions=2,
        workers_count=2,
    )
    trainer = pl.Trainer(gpus=-1, fast_dev_run=True)
    model = TransformerModel(d_input=8, d_model=8)
    trainer.fit(model, datamodule=data_module)

    predictions = trainer.predict(model, datamodule=data_module)
    assert len(predictions) == 1
    assert predictions[0]["prediction"].shape == torch.Size(
        [batch_size, subsequence_length * 8]
    )

    df = pd.DataFrame(
        {k: v.cpu().detach().numpy().tolist() for k, v in predictions[0].items()}
    )
    assert df.shape == (batch_size, 2)


def test_transformer_with_manual_tensor_creation(
    spark,
    synthetic_transformer_train_df_path,
    synthetic_transformer_train_pdf,
    tmp_path,
):
    batch_size = 10
    subsequence_length = 4
    data_module = PetastormTransformerDataModule(
        spark,
        "file:///tmp",
        synthetic_transformer_train_df_path,
        batch_size=batch_size,
        subsequence_length=subsequence_length,
        num_partitions=2,
        workers_count=2,
    )
    trainer = pl.Trainer(gpus=-1, fast_dev_run=True)
    model = TransformerModel(d_input=8, d_model=8)
    trainer.fit(model, datamodule=data_module)

    predictions = trainer.predict(model, datamodule=data_module)
    assert len(predictions) == 1
    assert predictions[0].shape == torch.Size([subsequence_length, batch_size, 8])

    model_checkpoint = tmp_path / "model.ckpt"
    trainer.save_checkpoint(model_checkpoint.as_posix())
    model = TransformerModel.load_from_checkpoint(
        model_checkpoint.as_posix(), d_model=8
    )

    df = (
        transform_into_transformer_predict_pairs(
            spark.createDataFrame(synthetic_transformer_train_pdf).withColumn(
                "features", mlF.array_to_vector("features")
            ),
            length=4,
        )
        .limit(10)
        .toPandas()
    )
    batch = {
        "src": torch.from_numpy(np.stack(df.src.values)).float(),
        "src_key_padding_mask": torch.from_numpy(
            np.stack(df.src_key_padding_mask.values)
        ),
        "src_pos": torch.from_numpy(np.stack(df.src_pos.values)),
    }
    z = model.predict_step(batch, 0)["prediction"].cpu().detach().numpy()
    series = pd.Series([list(z)])
    print(series)


@pytest.mark.skip(reason="known to fail")
def test_transformer_inference_transformer(
    spark,
    synthetic_transformer_train_df_path,
    synthetic_transformer_train_pdf,
    tmp_path,
):
    batch_size = 10
    subsequence_length = 4
    data_module = PetastormTransformerDataModule(
        spark,
        "file:///tmp",
        synthetic_transformer_train_df_path,
        batch_size=batch_size,
        subsequence_length=subsequence_length,
        num_partitions=2,
        workers_count=2,
    )
    trainer = pl.Trainer(gpus=-1, fast_dev_run=True)
    model = TransformerModel(d_input=8, d_model=8)
    trainer.fit(model, datamodule=data_module)

    model_checkpoint = tmp_path / "model.ckpt"
    trainer.save_checkpoint(model_checkpoint.as_posix())

    # load model in a udf and ensure it doesn't break
    @F.pandas_udf("float")
    def test_udf(it: Iterator[pd.Series]) -> Iterator[pd.Series]:
        # TODO: this breaks the UDF because the Spark worker dies (for some
        # reason unknown to me)
        model = TransformerModel.load_from_checkpoint(
            model_checkpoint.as_posix(), d_model=8
        )
        for item in it:
            print(item, flush=True)
            yield item

    spark.createDataFrame([{"feature": 1.0} for _ in range(10)]).select(
        test_udf(F.col("feature"))
    ).show()

    transformer_model = TransformerInferenceTransformer(
        inputCol="features",
        checkpointPath=model_checkpoint.as_posix(),
        sequenceLength=subsequence_length,
        numFeatures=8,
    )
    print(transformer_model)

    res = transformer_model.transform(
        spark.createDataFrame(synthetic_transformer_train_pdf).withColumn(
            "features", mlF.array_to_vector("features")
        )
    )
    res.printSchema()
    assert "prediction" in res.columns
    res.show(vertical=True, truncate=80, n=2)
    assert res.count() > 40
