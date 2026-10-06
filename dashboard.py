"""Local Streamlit dashboard for saved LLM evaluation runs."""
import json
from pathlib import Path

import streamlit as st

from evals.regression import compare_runs
from evals.runner import parse_runs

RUNS_PATH = Path(__file__).parent / "data" / "evaluation_runs.jsonl"


def load_runs() -> list[dict]:
    if not RUNS_PATH.exists():
        return []
    return parse_runs(RUNS_PATH.read_text(encoding="utf-8"))


st.set_page_config(page_title="LLM Regression Detector", page_icon="📊", layout="wide")
st.title("LLM Regression Detector")
st.caption("Local-only model evaluation, prompt comparison, and regression monitoring.")

runs = load_runs()
if not runs:
    st.info("No saved evaluation runs yet. Run `python -m evals.runner --prompt-version v1` first.")
    st.stop()

latest = runs[-1]
metrics = latest["metrics"]
first, second, third, fourth = st.columns(4)
first.metric("Accuracy", f"{metrics['accuracy']:.1%}")
second.metric("Correct cases", f"{metrics['correct_cases']} / {metrics['total_cases']}")
third.metric("Average latency", f"{metrics['mean_latency_ms']:.0f} ms")
fourth.metric("P95 latency", f"{metrics['p95_latency_ms']:.0f} ms")

st.subheader("Run history")
history = [
    {"run_id": run["run_id"], "prompt": run["prompt_version"], "accuracy": run["metrics"]["accuracy"],
     "avg_latency_ms": run["metrics"]["mean_latency_ms"]}
    for run in runs
]
st.dataframe(history, use_container_width=True, hide_index=True)
st.line_chart({"accuracy": [item["accuracy"] for item in history], "average latency (ms)": [item["avg_latency_ms"] for item in history]})

st.subheader("Prompt comparison")
if len(runs) < 2:
    st.caption("Run at least two evaluations to compare prompts.")
else:
    labels = [f"{run['run_id']} · {run['prompt_version']}" for run in runs]
    baseline_label = st.selectbox("Baseline", labels, index=max(0, len(labels) - 2))
    candidate_label = st.selectbox("Candidate", labels, index=len(labels) - 1)
    baseline = runs[labels.index(baseline_label)]
    candidate = runs[labels.index(candidate_label)]
    report = compare_runs(baseline, candidate)
    st.write(f"**Status:** {report['status'].upper()}")
    st.write(f"Accuracy change: **{report['accuracy_delta']:+.1%}**")
    st.write(f"Average latency change: **{report['latency_delta_ms']:+.1f} ms**")
    st.write("New regressions:", report["regressions"] or "None")

st.subheader("Latest failed cases")
failed = [case for case in latest["cases"] if not case["is_correct"]]
if failed:
    st.dataframe(failed, use_container_width=True, hide_index=True)
else:
    st.success("No failed cases in the latest run.")
