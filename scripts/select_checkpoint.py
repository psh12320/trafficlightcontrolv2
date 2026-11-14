"""Select a checkpoint using validation metrics and crossing-level constraints."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import statistics
from pathlib import Path


def _crossing_values(row: dict[str, str]) -> dict[str, float]:
    value = row.get("pedestrian_p95_by_crossing", "{}")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(value)
    return {str(key): float(item) for key, item in parsed.items()}


def _mean_crossing_p95(rows: list[dict[str, str]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row in rows:
        for crossing, wait in _crossing_values(row).items():
            values.setdefault(crossing, []).append(wait)
    return {
        crossing: statistics.mean(waits) if waits else 0.0
        for crossing, waits in values.items()
    }


def select_from_metrics(
    metrics_path: Path,
    baseline_path: Path,
    snapshot_dir: Path,
    episodes: list[int],
    output: Path | None = None,
) -> Path:
    with metrics_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    with baseline_path.open(newline="", encoding="utf-8") as file:
        baseline_rows = list(csv.DictReader(file))
    baseline_p95 = _mean_crossing_p95(baseline_rows)
    candidates = []
    for episode in episodes:
        episode_rows = [
            row for row in rows if int(row["training_episode"]) == episode
        ]
        if not episode_rows:
            raise FileNotFoundError(
                f"no validation rows for episode {episode} in {metrics_path}"
            )
        crossing_p95 = _mean_crossing_p95(episode_rows)
        violations = {
            crossing: crossing_p95.get(crossing, 0.0) - baseline_p95.get(crossing, 0.0)
            for crossing in baseline_p95
        }
        max_violation = max(violations.values(), default=0.0)
        candidates.append(
            {
                "controller": snapshot_dir.parent.name,
                "episode": episode,
                "checkpoint": str(snapshot_dir / f"episode-{episode}.pt"),
                "vehicle_time_loss": statistics.mean(
                    float(row["vehicle_time_loss_seconds"])
                    for row in episode_rows
                ),
                "max_crossing_p95_violation": max_violation,
                "normalized_violation": sum(
                    max(violation, 0.0) / max(baseline_p95[crossing], 1.0)
                    for crossing, violation in violations.items()
                ),
            }
        )
    feasible = [
        candidate
        for candidate in candidates
        if candidate["max_crossing_p95_violation"] <= 0
    ]
    selected = min(
        feasible or candidates,
        key=lambda candidate: (
            candidate["vehicle_time_loss"]
            if feasible
            else candidate["normalized_violation"],
            candidate["vehicle_time_loss"],
        ),
    )
    selected_path = Path(selected["checkpoint"])
    if not selected_path.exists():
        raise FileNotFoundError(selected_path)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    return selected_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--episodes", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(
        f"selected={select_from_metrics(args.metrics, args.baseline, args.snapshot_dir, args.episodes, args.output)}"
    )
