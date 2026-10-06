import argparse
import json
from datetime import datetime, timezone

from app.classifier import OllamaClassifier
from app.config import DATA_DIR, ROOT_DIR
from evals.metrics import compute_metrics
from evals.regression import compare_runs


def evaluate(prompt_version: str) -> dict:
    cases = json.loads((ROOT_DIR / "datasets" / "golden_dataset.json").read_text())
    classifier = OllamaClassifier()
    results = []
    for case in cases:
        prediction, latency_ms = classifier.classify(case["input"], prompt_version)
        results.append({
            "id": case["id"], "expected_category": case["expected_category"],
            "predicted_category": prediction.category.value, "summary": prediction.summary,
            "latency_ms": round(latency_ms, 1),
            "is_correct": prediction.category.value == case["expected_category"],
        })
    return {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "prompt_version": prompt_version,
        "metrics": compute_metrics(results).to_dict(),
        "cases": results,
    }


def load_runs() -> list[dict]:
    path = DATA_DIR / "evaluation_runs.jsonl"
    if not path.exists():
        return []
    return parse_runs(path.read_text(encoding="utf-8"))


def parse_runs(content: str) -> list[dict]:
    """Read newline-delimited JSON, including early files written with literal \\n."""
    decoder = json.JSONDecoder()
    runs, index = [], 0
    while index < len(content):
        while index < len(content) and (content[index].isspace() or content.startswith("\\\\n", index)):
            index += 2 if content.startswith("\\\\n", index) else 1
        if index >= len(content):
            break
        run, index = decoder.raw_decode(content, index)
        runs.append(run)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-version", default="v1")
    parser.add_argument("--compare-with", help="Run ID to use as the baseline.")
    parser.add_argument("--list-runs", action="store_true")
    args = parser.parse_args()
    if args.list_runs:
        print(json.dumps([
            {"run_id": run["run_id"], "prompt_version": run["prompt_version"], "accuracy": run["metrics"]["accuracy"]}
            for run in load_runs()
        ], indent=2))
        return
    run = evaluate(args.prompt_version)
    DATA_DIR.mkdir(exist_ok=True)
    with (DATA_DIR / "evaluation_runs.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(run) + "\n")
    output = {"run_id": run["run_id"], "prompt_version": run["prompt_version"], "metrics": run["metrics"]}
    if args.compare_with:
        baseline = next((saved for saved in load_runs() if saved["run_id"] == args.compare_with), None)
        if baseline is None:
            raise SystemExit(f"No saved run found with ID: {args.compare_with}")
        output["regression_report"] = compare_runs(baseline, run)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
