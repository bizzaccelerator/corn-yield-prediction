import pytest

from scripts.support_scripts import model_fetch, model_registry


def test_register_and_fetch_model(tmp_path):
    model = {"name": "test_model", "version": 1}
    model_registry.register(model, path=tmp_path)
    fetched = model_fetch.fetch("test_model", version=1, path=tmp_path)
    assert fetched["name"] == "test_model"


def test_fetch_nonexistent_raises(tmp_path):
    with pytest.raises(Exception):
        model_fetch.fetch("unknown_model", version=1, path=tmp_path)
