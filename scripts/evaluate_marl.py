"""Evaluate a learned IPPO/MAPPO checkpoint on the baseline demand."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

if __package__:
    from .build_network import GENERATED, build
    from .train_marl import AGENTS, Actor, SignalEnvironment
else:
    from build_network import GENERATED, build
    from train_marl import AGENTS, Actor, SignalEnvironment


def evaluate(
    checkpoint: Path,
    seeds: list[int],
    duration: int,
    pedestrian_speed: float | None = None,
    pedestrian_interval: int | None = None,
    vehicle_period: int | None = None,
    skip_build: bool = False,
) -> list[dict[str, float | str | dict[str, float]]]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = saved.get("metadata", {})
    action_size = metadata.get("action_size", 2)
    green_duration = metadata.get("green_duration", 37)
    configured_speed = metadata.get("pedestrian_speed", 1.4)
    speed = pedestrian_speed if pedestrian_speed is not None else configured_speed
    interval = (
        pedestrian_interval
        if pedestrian_interval is not None
        else metadata.get("pedestrian_interval", 20)
    )
    period = (
        vehicle_period
        if vehicle_period is not None
        else metadata.get("vehicle_period", 6)
    )
    if not skip_build:
        build(green_duration, speed, period)
    actors = [Actor(action_size, metadata.get("observation_size", 25)) for _ in AGENTS]
    for actor, state in zip(actors, saved["actors"]):
        actor.load_state_dict(state)
        actor.eval()

    results = []
    for seed in seeds:
        environment = SignalEnvironment(
            seed,
            duration,
            metadata.get("reward_mode", "raw"),
            metadata.get("vehicle_weight", 1.0),
            metadata.get("pedestrian_weight", 1.0),
            metadata.get("switch_penalty", 0.1),
            metadata.get("observation_mode", "full"),
            action_size,
            green_duration,
            speed,
            interval,
            frame_stack=metadata.get("frame_stack", 1),
        )
        observation = environment.reset()
        try:
            done = False
            while not done:
                masks = environment.action_masks()
                with torch.no_grad():
                    actions = torch.stack(
                        [
                            actor.distribution(
                                observation[index], masks[index]
                            ).probs.argmax()
                            for index, actor in enumerate(actors)
                        ]
                    )
                observation, _, done = environment.step(actions)
            metrics = environment.metrics()
        finally:
            environment.close()
        results.append({"controller": saved["algorithm"], "seed": seed,
                        "duration_seconds": duration, "vehicle_period": period,
                        "pedestrian_interval": interval,
                        "pedestrian_speed_meters_per_second": speed, **metrics})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--pedestrian-speed", type=float)
    parser.add_argument("--pedestrian-interval", type=int)
    parser.add_argument("--vehicle-period", type=int)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = evaluate(
        args.checkpoint,
        args.seeds,
        args.duration,
        args.pedestrian_speed,
        args.pedestrian_interval,
        args.vehicle_period,
        args.skip_build,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    for row in rows:
        print(row)
