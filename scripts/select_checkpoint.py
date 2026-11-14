"""Select a training snapshot using validation seeds only."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

if __package__:
    from .evaluate_marl import evaluate
else:
    from evaluate_marl import evaluate


def validation_score(rows: list[dict[str, float]]) -> tuple[float, float, float]:
    vehicle = statistics.mean(row["vehicle_average_waiting_seconds"] for row in rows)
    pedestrian = statistics.mean(
        row["pedestrian_average_waiting_seconds"] for row in rows
    )
    violation = max(vehicle - 31.0, 0.0) / 31.0 + max(pedestrian - 80.0, 0.0) / 80.0
    return violation, vehicle, pedestrian


def select(
    snapshot_dir: Path, seeds: list[int], episodes: list[int]
) -> tuple[Path, list[dict[str, float]]]:
    candidates = []
    for episode in episodes:
        checkpoint = snapshot_dir / f"episode-{episode}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        rows = evaluate(checkpoint, seeds, 300)
        violation, vehicle, pedestrian = validation_score(rows)
        candidates.append(
            {
                "episode": episode,
                "checkpoint": str(checkpoint),
                "validation_violation": violation,
                "validation_vehicle_wait": vehicle,
                "validation_pedestrian_wait": pedestrian,
            }
        )
    selected = min(
        candidates,
        key=lambda row: (
            row["validation_violation"],
            row["validation_vehicle_wait"],
            row["validation_pedestrian_wait"],
        ),
    )
    return Path(selected["checkpoint"]), candidates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--validation-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--episodes", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checkpoint, rows = select(
        args.snapshot_dir, args.validation_seeds, args.episodes
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"selected={checkpoint}")
