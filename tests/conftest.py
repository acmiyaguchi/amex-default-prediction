from pathlib import Path

import pytest

from amex_default_prediction.utils import spark_session


@pytest.fixture(scope="session")
def spark():
    ss = spark_session()
    yield ss
    ss.stop()


@pytest.fixture(scope="session")
def data_path():
    return Path(__file__).parent / "data"
