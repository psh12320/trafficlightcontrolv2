"""Summarize per-seed experiment CSV files with mean and 95% CI."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


METRICS = (
    "vehicle_time_loss_seconds",
    "vehicle_completed_time_loss_seconds",
    "vehicle_completed",
    "vehicle_unfinished",
    "vehicle_average_waiting_seconds",
    "pedestrian_average_waiting_seconds",
    "vehicle_completed_average_waiting_seconds",
    "pedestrian_completed_average_waiting_seconds",
    "pedestrian_p95_waiting_seconds",
    "pedestrian_completed_p95_waiting_seconds",
    "pedestrian_completed",
    "pedestrian_unfinished",
)


def summarize(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    result = []
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if metric in row]
        if not values:
            continue
        standard_error = (
            statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0
        )
        result.append(
            {
                "metric": metric,
                "n": len(values),
                "mean": statistics.mean(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "ci95_low": statistics.mean(values) - 1.96 * standard_error,
                "ci95_high": statistics.mean(values) + 1.96 * standard_error,
            }
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path)
    args = parser.parse_args()
    for row in summarize(args.csv_file):
        print(row)
