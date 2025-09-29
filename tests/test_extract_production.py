import pytest
from scripts.support_scripts import extract_production_model


def test_extract_model(tmp_path):
    model = {"name": "prod_model", "version": "latest"}
    path = tmp_path / "registry"
    path.mkdir()
    (path / "prod_model.pkl").write_text("dummy")
    extracted = extract_production_model.extract("prod_model", path=path)
    assert extracted is not None
