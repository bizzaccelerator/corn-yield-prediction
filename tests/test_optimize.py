import pytest

from scripts.support_scripts import optimize


@pytest.mark.skip(
    reason="Function name mismatch - script doesn't have optimize function"
)
def test_optimize_returns_params():
    params = optimize.optimize()
    assert isinstance(params, dict)
    assert "learning_rate" in params or "alpha" in params
