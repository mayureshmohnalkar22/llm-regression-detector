from evals.metrics import compute_metrics
from evals.regression import compare_runs


def test_compute_metrics() -> None:
    metrics = compute_metrics([
        {"is_correct": True, "latency_ms": 10}, {"is_correct": False, "latency_ms": 30}
    ])
    assert metrics.accuracy == 0.5
    assert metrics.mean_latency_ms == 20


def test_detects_accuracy_regression() -> None:
    baseline = {"metrics": {"accuracy": 1.0}, "cases": [{"id": "a", "is_correct": True}]}
    candidate = {"metrics": {"accuracy": 0.8}, "cases": [{"id": "a", "is_correct": False}]}
    report = compare_runs(baseline, candidate)
    assert report["status"] == "critical"
    assert report["regressions"] == ["a"]
