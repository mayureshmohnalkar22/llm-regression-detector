from dataclasses import asdict, dataclass
from statistics import mean


@dataclass
class Metrics:
    total_cases: int
    correct_cases: int
    accuracy: float
    mean_latency_ms: float
    p95_latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(case_results: list[dict]) -> Metrics:
    total = len(case_results)
    correct = sum(item["is_correct"] for item in case_results)
    latencies = [item["latency_ms"] for item in case_results]
    ordered = sorted(latencies)
    p95_index = max(0, int(len(ordered) * 0.95) - 1)
    return Metrics(
        total,
        correct,
        round(correct / total if total else 0, 4),
        round(mean(latencies) if latencies else 0, 1),
        round(ordered[p95_index] if ordered else 0, 1),
    )
