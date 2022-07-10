import pytest

from amex_default_prediction.utils import spark_session


@pytest.fixture(scope="session")
def spark():
    return spark_session()
