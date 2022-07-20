import pytorch_lightning as pl

from amex_default_prediction.model.weibull import AmexDataModule, Net


def test_data_module_has_fields(spark, cache_path, synthetic_train_data_path):
    data_module = AmexDataModule(spark, cache_path, synthetic_train_data_path)
    data_module.setup()
    dataloader = data_module.train_dataloader()
    print(next(dataloader))
