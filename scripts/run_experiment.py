"""Run repeated controller experiments and aggregate waiting-time metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from run_baseline import GENERATED, run


def main(
    seeds: list[int],
    duration: int,
    output: Path,
    controller: str,
    green_duration: int,
    pedestrian_speed: float,
    vehicle_period: int,
    pedestrian_interval: int,
) -> None:
    rows = [
        run(
            seed,
            duration,
            output.with_name(f"{controller}-{seed}.json"),
            True,
            controller,
            green_duration,
            pedestrian_speed,
            vehicle_period,
            pedestrian_interval,
        )
        for seed in seeds
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
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
    print(f"Wrote {len(rows)} baseline runs to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument(
        "--controller",
        choices=("pedestrian_actuated", "fixed_time"),
        default="pedestrian_actuated",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=GENERATED / "baseline-results.csv",
    )
    parser.add_argument("--green-duration", type=int, default=37)
    parser.add_argument("--pedestrian-speed", type=float, default=1.4)
    parser.add_argument("--vehicle-period", type=int, default=6)
    parser.add_argument("--pedestrian-interval", type=int, default=20)
    args = parser.parse_args()
    main(
        args.seeds,
        args.duration,
        args.output,
        args.controller,
        args.green_duration,
        args.pedestrian_speed,
        args.vehicle_period,
        args.pedestrian_interval,
    )
