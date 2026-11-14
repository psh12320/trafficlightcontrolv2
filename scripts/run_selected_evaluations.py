"""Evaluate selected MARL checkpoints and baselines across proposal scenarios."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

if __package__:
    from .build_network import build
    from .evaluate_marl import evaluate
    from .run_baseline import run
else:
    from build_network import build
    from evaluate_marl import evaluate
    from run_baseline import run


def _evaluate_one(task: tuple) -> list[dict[str, object]]:
    (
        controller,
        checkpoint,
        seeds,
        duration,
        pedestrian_speed,
        vehicle_period,
        pedestrian_interval,
        output_root,
    ) = task
    rows = []
    if checkpoint is None:
        for seed in seeds:
            output = output_root / f"{controller}-{seed}.json"
            rows.append(
                run(
                    seed,
                    duration,
                    output,
                    False,
                    controller,
                    37,
                    pedestrian_speed,
                    vehicle_period,
                    pedestrian_interval,
                )
            )
    else:
        rows = evaluate(
            checkpoint,
            seeds,
            duration,
            pedestrian_speed,
            pedestrian_interval,
            vehicle_period,
            skip_build=True,
        )
    for row in rows:
        row["scenario_controller"] = controller
        if checkpoint is not None:
            row["checkpoint"] = str(checkpoint)
    return rows


def evaluate_scenario(
    name: str,
    selected: list[tuple[str, Path]],
    seeds: list[int],
    duration: int,
    pedestrian_speed: float,
    vehicle_period: int,
    pedestrian_interval: int,
    output_root: Path,
) -> Path:
    scenario_root = output_root / name
    scenario_root.mkdir(parents=True, exist_ok=True)
    build(37, pedestrian_speed, vehicle_period)
    tasks = [
        (
            controller,
            checkpoint,
            seeds,
            duration,
            pedestrian_speed,
            vehicle_period,
            pedestrian_interval,
            scenario_root,
        )
        for controller, checkpoint in [
            ("fixed_time", None),
            ("pedestrian_actuated", None),
        ]
        + selected
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=min(len(tasks), 8)) as pool:
        for result in pool.map(_evaluate_one, tasks):
            rows.extend(
                {
                    "scenario": name,
                    "vehicle_period": vehicle_period,
                    "pedestrian_interval": pedestrian_interval,
                    **row,
                }
                for row in result
            )
    output = output_root / f"{name}-results.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        fieldnames = sorted(rows[0])
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, dict)
                else value
                for key, value in row.items()
            }
            for row in rows
        )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--duration", type=int, default=3600)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 12, 13])
    parser.add_argument("--pedestrian-speed", type=float, default=1.4)
    parser.add_argument("--vehicle-period", type=int, default=6)
    parser.add_argument("--pedestrian-interval", type=int, default=20)
    args = parser.parse_args()
    selected = []
    with args.selection.open(encoding="utf-8") as file:
        for row in json.load(file):
            selected.append((row["controller"], Path(row["checkpoint"])))
    evaluate_scenario(
        "primary",
        selected,
        args.seeds,
        args.duration,
        args.pedestrian_speed,
        args.vehicle_period,
        args.pedestrian_interval,
        args.output_root,
    )
