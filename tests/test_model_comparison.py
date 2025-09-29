from support_scripts import model_performance_comparison, compare_models


def test_model_performance_comparison():
    metrics = {"linear": 0.8, "ridge": 0.85}
    best = model_performance_comparison.get_best_model(metrics)
    assert best == "ridge"


def test_compare_models_handles_tie():
    metrics = {"linear": 0.9, "ridge": 0.9}
    result = compare_models.compare(metrics)
    assert result in ["linear", "ridge"]
