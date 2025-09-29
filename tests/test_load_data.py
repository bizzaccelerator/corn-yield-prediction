import pandas as pd
import pytest

from scripts.support_scripts import (
    load_and_prepare_base_data,
    load_and_prepare_test_data,
)


def test_base_loader_returns_dataframe():
    df = load_and_prepare_base_data.load_data("tests/sample.csv")
    assert isinstance(df, pd.DataFrame)


def test_test_loader_returns_dataframe():
    df = load_and_prepare_test_data.load_data("tests/sample.csv")
    assert isinstance(df, pd.DataFrame)


def test_loader_missing_file():
    with pytest.raises(FileNotFoundError):
        load_and_prepare_base_data.load_data("nonexistent.csv")
