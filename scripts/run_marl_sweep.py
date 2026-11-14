"""Run reproducible IPPO/MAPPO controller sweeps and checkpoint evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

if __package__:
    from .evaluate_marl import evaluate
    from .train_marl import ACTION_SIZES, GENERATED, train
else:
    from evaluate_marl import evaluate
    from train_marl import ACTION_SIZES, GENERATED, train


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=sorted(rows[0]))
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


def run_configuration(
    algorithm: str,
    reward_mode: str,
    action_size: int,
    episodes: int,
    horizon: int,
    seed: int,
    evaluation_seeds: list[int],
    evaluation_duration: int,
    pedestrian_speed: float,
    output_root: Path,
) -> Path:
    name = f"{algorithm}-{reward_mode}-actions{action_size}-seed{seed}"
    configuration_root = output_root / name
    snapshot_dir = configuration_root / "snapshots"
    checkpoint = configuration_root / "final.pt"
    metrics_path = configuration_root / "checkpoint-metrics.csv"
    rows: list[dict[str, object]] = []

    def evaluate_checkpoint(path: Path, episode: int) -> None:
        for row in evaluate(path, evaluation_seeds, evaluation_duration):
            rows.append(
                {
                    "algorithm": algorithm,
                    "reward_mode": reward_mode,
                    "action_size": action_size,
                    "training_seed": seed,
                    "training_episode": episode,
                    **row,
                }
            )
        _write_rows(metrics_path, rows)

    train(
        algorithm=algorithm,
        episodes=episodes,
        seed=seed,
        checkpoint=checkpoint,
        horizon=horizon,
        snapshot_dir=snapshot_dir,
        reward_mode=reward_mode,
        action_size=action_size,
        pedestrian_speed=pedestrian_speed,
        snapshot_interval=500,
        checkpoint_callback=evaluate_checkpoint,
    )
    return metrics_path


def main(args: argparse.Namespace) -> None:
    output_root = args.output_root
    for algorithm in args.algorithms:
        for reward_mode in args.reward_modes:
            for action_size in args.action_sizes:
                run_configuration(
                    algorithm,
                    reward_mode,
                    action_size,
                    args.episodes,
                    args.horizon,
                    args.seed,
                    args.evaluation_seeds,
                    args.evaluation_duration,
                    args.pedestrian_speed,
                    output_root,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", nargs="+", choices=("ippo", "mappo"),
                        default=("ippo", "mappo"))
    parser.add_argument(
        "--reward-modes",
        nargs="+",
        choices=("vehicle", "vehicle_pedestrian"),
        default=("vehicle", "vehicle_pedestrian"),
    )
    parser.add_argument("--action-sizes", nargs="+", choices=ACTION_SIZES,
                        type=int, default=ACTION_SIZES)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--horizon", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--evaluation-seeds", nargs="+", type=int, default=(11, 12, 13))
    parser.add_argument("--evaluation-duration", type=int, default=3600)
    parser.add_argument("--pedestrian-speed", type=float, default=1.4)
    parser.add_argument("--output-root", type=Path, default=GENERATED / "marl-sweep")
    main(parser.parse_args())
