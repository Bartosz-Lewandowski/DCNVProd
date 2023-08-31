import pytest


def pytest_configure():
    pytest.temp_dir = "temp_dir"
