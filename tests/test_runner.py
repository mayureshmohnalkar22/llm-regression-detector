from evals.runner import parse_runs


def test_parse_runs_handles_legacy_literal_newlines() -> None:
    assert parse_runs('{"run_id":"one"}\\n{"run_id":"two"}') == [
        {"run_id": "one"},
        {"run_id": "two"},
    ]
