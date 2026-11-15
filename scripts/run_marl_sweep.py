"""Run reproducible IPPO/MAPPO controller sweeps and checkpoint evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

if __package__:
    from .build_network import build
    from .evaluate_marl import evaluate
    from .train_marl import ACTION_SIZES, DEFAULT_SWITCH_PENALTY, GENERATED, train
else:
    from build_network import build
    from evaluate_marl import evaluate
    from train_marl import ACTION_SIZES, DEFAULT_SWITCH_PENALTY, GENERATED, train


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
    snapshot_interval: int,
    pedestrian_interval: int,
    vehicle_period: int,
    output_root: Path,
    randomize_demand: bool = False,
    switch_penalty: float = DEFAULT_SWITCH_PENALTY,
    learning_rate: float = 3e-4,
    frame_stack: int = 1,
    bc_episodes: int = 0,
) -> Path:
    name = f"{algorithm}-{reward_mode}-actions{action_size}-seed{seed}"
    configuration_root = output_root / name
    snapshot_dir = configuration_root / "snapshots"
    checkpoint = configuration_root / "final.pt"
    metrics_path = configuration_root / "checkpoint-metrics.csv"
    if metrics_path.exists() or checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite existing training outputs: {configuration_root}")
    configuration_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    def evaluate_checkpoint(path: Path, episode: int) -> None:
        for row in evaluate(
            path,
            evaluation_seeds,
            evaluation_duration,
            pedestrian_speed,
            pedestrian_interval,
            vehicle_period,
            skip_build=True,
        ):
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

    started = time.time()
    with (configuration_root / "training.log").open("w", encoding="utf-8", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
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
            pedestrian_interval=pedestrian_interval,
            vehicle_period=vehicle_period,
            snapshot_interval=snapshot_interval,
            checkpoint_callback=evaluate_checkpoint,
            skip_build=True,
            randomize_demand=randomize_demand,
            switch_penalty=switch_penalty,
            learning_rate=learning_rate,
            frame_stack=frame_stack,
            bc_episodes=bc_episodes,
        )
    (configuration_root / "training-complete.json").write_text(json.dumps({
        "algorithm": algorithm, "reward_mode": reward_mode, "training_seed": seed,
        "episodes": episodes, "horizon": horizon, "randomize_demand": randomize_demand,
        "switch_penalty": switch_penalty, "elapsed_seconds": time.time()-started,
        "learning_rate": learning_rate, "frame_stack": frame_stack, "bc_episodes": bc_episodes,
    }, indent=2), encoding="utf-8")
    return metrics_path


def main(args: argparse.Namespace) -> None:
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "sweep-status.json"
    def status(stage, **extra):
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"stage": stage, "training_seed": args.seed,
            "updated_unix": time.time(), "episodes": args.episodes, "horizon": args.horizon,
            "randomize_demand": args.randomize_demand, "switch_penalty": args.switch_penalty,
            **extra}, indent=2), encoding="utf-8")
        temporary.replace(status_path)
    status("building")
    if not args.skip_build:
        build(37, args.pedestrian_speed, args.vehicle_period)
    configs = [
        (algorithm, reward_mode, action_size)
        for algorithm in args.algorithms
        for reward_mode in args.reward_modes
        for action_size in args.action_sizes
    ]
    status("training", completed_configurations=0)
    try:
        with ProcessPoolExecutor(max_workers=min(len(configs), 8)) as pool:
            futures = [
                pool.submit(
                    run_configuration, algorithm, reward_mode, action_size,
                    args.episodes, args.horizon, args.seed, list(args.evaluation_seeds),
                    args.evaluation_duration, args.pedestrian_speed, args.snapshot_interval,
                    args.pedestrian_interval, args.vehicle_period, output_root,
                    args.randomize_demand, args.switch_penalty,
                    getattr(args, 'learning_rate', 3e-4), getattr(args, 'frame_stack', 1),
                    getattr(args, 'bc_episodes', 0),
                )
                for algorithm, reward_mode, action_size in configs
            ]
            for completed, future in enumerate(as_completed(futures), 1):
                future.result()
                status("training", completed_configurations=completed)
        status("complete", completed_configurations=len(configs))
    except BaseException as error:
        status("failed", error=str(error))
        raise


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
                        type=int, default=(2,))
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--horizon", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--evaluation-seeds", nargs="+", type=int, default=(11, 12, 13))
    parser.add_argument("--evaluation-duration", type=int, default=3600)
    parser.add_argument("--pedestrian-speed", type=float, default=1.4)
    parser.add_argument("--snapshot-interval", type=int, default=500)
    parser.add_argument("--pedestrian-interval", type=int, default=20)
    parser.add_argument("--vehicle-period", type=int, default=6)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--randomize-demand", action="store_true")
    parser.add_argument("--switch-penalty", type=float, default=DEFAULT_SWITCH_PENALTY)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--frame-stack", type=int, choices=(1, 3), default=1)
    parser.add_argument("--bc-episodes", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=GENERATED / "marl-sweep")
    main(parser.parse_args())
