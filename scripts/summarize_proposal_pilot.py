"""Summarize proposal results without treating test seeds as RL replicates."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ("vehicle_time_loss_seconds", "vehicle_mean_time_loss_seconds",
           "vehicle_throughput_per_hour", "vehicle_unfinished",
           "pedestrian_unfinished", "pedestrian_average_waiting_seconds",
           "worst_crossing_p95_waiting_seconds", "pedestrian_max_waiting_seconds")


def summarize(paths: list[Path]) -> list[dict]:
    groups = defaultdict(lambda: defaultdict(list))
    for path in paths:
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                controller = row["scenario_controller"]
                match = re.fullmatch(r"(ippo|mappo)-(vehicle|vehicle_pedestrian)-actions(?:2|4)-seed(\d+)", controller)
                if match:
                    family = f"{match[1]}-{match[2]}"
                    unit = int(match[3])
                else:
                    if controller not in ("fixed_time", "pedestrian_actuated"):
                        raise ValueError(f"unknown controller {controller}")
                    family, unit = controller, int(row["seed"])
                values = {key: float(row[key]) for key in METRICS if row.get(key)}
                departed = float(row.get("vehicle_departed") or row.get("vehicle_arrivals") or 0)
                values["vehicle_mean_time_loss_seconds"] = float(row["vehicle_time_loss_seconds"]) / max(departed, 1)
                values["vehicle_throughput_per_hour"] = float(row["vehicle_completed"]) * 3600 / float(row["duration_seconds"])
                crossing = json.loads(row["pedestrian_p95_by_crossing"])
                if len(crossing) != 12:
                    raise ValueError("expected 12 crossing p95 metrics")
                values["worst_crossing_p95_waiting_seconds"] = max(crossing.values())
                values["test_seed"] = int(row["seed"])
                groups[(row["scenario"], family)][unit].append(values)
    result = []
    for (scenario, family), units in sorted(groups.items()):
        learned = family.startswith(("ippo-", "mappo-"))
        if len(units) != 3:
            raise ValueError(f"expected 3 independent seeds for {scenario}/{family}")
        for rows in units.values():
            expected = 3 if learned else 1
            if len(rows) != expected or len({r["test_seed"] for r in rows}) != expected:
                raise ValueError(f"missing or duplicate test rows for {scenario}/{family}")
        for metric in METRICS:
            if not all(metric in row for rows in units.values() for row in rows):
                continue
            samples = [statistics.mean(row[metric] for row in rows) for rows in units.values()]
            mean = statistics.mean(samples)
            half = 4.302652729911275 * statistics.stdev(samples) / math.sqrt(3)
            result.append({
                "scenario": scenario, "controller": family, "metric": metric,
                "independent_n": 3, "evaluation_n": sum(map(len, units.values())),
                "uncertainty_unit": "training_seed" if learned else "test_seed",
                "mean": mean, "ci95_half_width": half,
                "ci95_low": mean-half, "ci95_high": mean+half,
                "maximum_observed": max(row[metric] for rows in units.values() for row in rows),
            })
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = summarize([args.root/name for name in (
        "final-primary-results.csv", "final-demand-sweep.csv",
        "final-slow-pedestrian-results.csv")])
    with (args.root/"final-summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = ["vehicle_mean_time_loss_seconds", "vehicle_throughput_per_hour",
               "vehicle_unfinished", "pedestrian_average_waiting_seconds",
               "worst_crossing_p95_waiting_seconds"]
    lines = ["# Proposal pilot results", "",
        "Mean ± 95% Student-t CI (n=3). MARL: test seeds are averaged within each training seed; baseline: CI across test seeds.",
        "Vehicle time loss is total measured loss divided by departed vehicles; throughput is completed vehicles per simulated hour.",
        "Worst crossing p95 is the maximum of 12 crossing percentiles per evaluation, then averaged across evaluations/seeds.",
        "Intervals are descriptive and unadjusted for comparisons. Unfinished pedestrians remain in waiting metrics.", ""]
    lines += [
        "Active unfinished vehicles exclude vehicles still queued for insertion. Time loss per vehicle uses departed vehicles as its denominator.",
        "Only 1 of 12 selected runs met every crossing's validation limit; 11 used the predeclared lowest normalized violation fallback.",
        "The high-demand evaluation was rerun with pedestrian injection every simulated second. Accepted high-demand rows each contain 3,588 injected pedestrians; initial high-demand results are diagnostic only.",
        "The selected checkpoints were frozen before held-out testing. Test seeds were held out from this run's training/selection, but were used in earlier exploratory work.",
        "Raw CSVs retain total vehicle time loss; final-summary.csv also reports pedestrian unfinished counts and the maximum observed value for each metric.", ""]
    for scenario in ("primary", "low", "high", "slow"):
        lines += [f"## {scenario}", "", "| Controller | Time loss (s/vehicle) | Throughput (veh/h) | Active unfinished vehicles | Mean pedestrian wait (s) | Worst crossing p95 (s) |", "|---|---:|---:|---:|---:|---:|"]
        families = ["fixed_time", "pedestrian_actuated", "ippo-vehicle", "ippo-vehicle_pedestrian", "mappo-vehicle", "mappo-vehicle_pedestrian"]
        for family in families:
            values = []
            for metric in metrics:
                row = next(row for row in rows if row["scenario"] == scenario and row["controller"] == family and row["metric"] == metric)
                values.append(f"{row['mean']:.2f} ± {row['ci95_half_width']:.2f}")
            lines.append("| " + " | ".join([family]+values) + " |")
        lines.append("")
    (args.root/"FINAL_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
