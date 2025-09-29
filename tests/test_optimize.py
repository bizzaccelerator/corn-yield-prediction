from scripts.support_scripts import optimize


def test_optimize_returns_params():
    params = optimize.optimize()
    assert isinstance(params, dict)
    assert "learning_rate" in params or "alpha" in params
