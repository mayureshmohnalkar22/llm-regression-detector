def compare_runs(baseline: dict, candidate: dict, accuracy_drop_threshold: float = 0.03) -> dict:
    baseline_accuracy = baseline["metrics"]["accuracy"]
    candidate_accuracy = candidate["metrics"]["accuracy"]
    delta = candidate_accuracy - baseline_accuracy
    baseline_cases = {case["id"]: case for case in baseline["cases"]}
    regressions = [
        case["id"] for case in candidate["cases"]
        if baseline_cases.get(case["id"], {}).get("is_correct") and not case["is_correct"]
    ]
    if delta <= -accuracy_drop_threshold:
        status = "critical"
    elif regressions:
        status = "warning"
    else:
        status = "pass"
    return {
        "status": status,
        "baseline_run_id": baseline["run_id"],
        "candidate_run_id": candidate["run_id"],
        "accuracy_delta": round(delta, 4),
        "latency_delta_ms": round(
            candidate["metrics"].get("mean_latency_ms", 0) - baseline["metrics"].get("mean_latency_ms", 0), 1
        ),
        "regressions": regressions,
    }
