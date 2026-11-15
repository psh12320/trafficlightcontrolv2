"""Train, validation-select, and test the cost-bounded proposal experiment.

The caller owns cloud billing and termination. Network files are shared, so each
stage joins all workers before any subsequent network build.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from xml.etree import ElementTree

if __package__:
    from .build_network import GENERATED, build
    from .run_baseline import run
    from .run_marl_sweep import _write_rows, run_configuration
    from .run_selected_evaluations import evaluate_scenario
    from .select_checkpoint import select_from_metrics
else:
    from build_network import GENERATED, build
    from run_baseline import run
    from run_marl_sweep import _write_rows, run_configuration
    from run_selected_evaluations import evaluate_scenario
    from select_checkpoint import select_from_metrics


def _status(root: Path, stage: str, **extra) -> None:
    data = {"stage": stage, "updated_unix": time.time(), **extra}
    temporary = root / "pilot-status.tmp"
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(root / "pilot-status.json")
    print(json.dumps(data), flush=True)


def _train_one(task: tuple) -> str:
    algorithm, reward, seed, args = task
    name = f"{algorithm}-{reward}-actions4-seed{seed}"
    seed_root = args.output_root / f"multiseed-{seed}"
    config_root = seed_root / name
    config_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with (config_root / "training.log").open("w", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            metrics = run_configuration(
                algorithm, reward, 4, args.episodes, args.horizon, seed,
                args.validation_seeds, args.evaluation_duration, 1.4,
                args.snapshot_interval, 20, 6, seed_root,
            )
    (config_root / "training-complete.json").write_text(
        json.dumps({"training_seed": seed, "elapsed_seconds": time.time()-started,
                    "metrics": str(metrics)}, indent=2), encoding="utf-8",
    )
    return str(metrics)


def _validation_baseline(task: tuple) -> dict:
    seed, duration, root = task
    return run(seed, duration, root / f"pedestrian_actuated-{seed}.json",
               False, "pedestrian_actuated", 37, 1.4, 6, 20)


def _combine(paths: list[Path], output: Path) -> None:
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as file:
            rows.extend(csv.DictReader(file))
    _write_rows(output, rows)


def main(args: argparse.Namespace) -> None:
    import torch
    import libsumo  # Fail before training if the fast simulator is missing.
    import importlib.metadata

    if args.episodes < 1 or args.snapshot_interval < 1:
        raise ValueError("episodes and snapshot interval must be positive")
    if args.episodes % args.snapshot_interval:
        raise ValueError("episodes must be a multiple of the snapshot interval")
    if args.workers < 1 or args.horizon < 5 or args.horizon % 5:
        raise ValueError("workers must be positive and horizon a multiple of five")
    if set(args.validation_seeds) & set(args.test_seeds):
        raise ValueError("validation and test seeds must be disjoint")
    if importlib.metadata.version("libsumo") != importlib.metadata.version("eclipse-sumo"):
        raise RuntimeError("libsumo and eclipse-sumo versions must match")
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    if (root / "pilot-status.json").exists():
        raise FileExistsError("output already contains a pilot; use a fresh output root")
    provenance = {
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "libsumo": importlib.metadata.version("libsumo"),
        "eclipse_sumo": importlib.metadata.version("eclipse-sumo"),
        "source_sha256": {str(path.relative_to(Path(__file__).parent)):
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(__file__).parent.glob("*.py")},
        "selection_rule": "lowest vehicle time loss subject to each crossing's mean validation p95 <= actuated; else lowest summed normalized positive violation",
        "started_unix": time.time(),
    }
    (root / "pilot-provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    try:
        _status(root, "build_primary")
        build(37, 1.4, 6)
        validation_root = root / "validation-baseline"
        validation_root.mkdir(exist_ok=True)
        _status(root, "validation_baseline")
        with ProcessPoolExecutor(max_workers=min(3, args.workers)) as pool:
            rows = list(pool.map(_validation_baseline,
                [(seed, args.evaluation_duration, validation_root) for seed in args.validation_seeds]))
        baseline_csv = validation_root / "pedestrian-actuated-results.csv"
        _write_rows(baseline_csv, rows)

        _status(root, "training", configurations=4 * len(args.training_seeds))
        tasks = [(algorithm, reward, seed, args)
                 for seed in args.training_seeds
                 for algorithm in ("ippo", "mappo")
                 for reward in ("vehicle", "vehicle_pedestrian")]
        with ProcessPoolExecutor(max_workers=min(len(tasks), args.workers)) as pool:
            metrics_paths = list(pool.map(_train_one, tasks))

        _status(root, "checkpoint_selection")
        episodes = list(range(args.snapshot_interval, args.episodes+1, args.snapshot_interval))
        selected = []
        records = []
        for metrics_path in map(Path, metrics_paths):
            with metrics_path.open(newline="", encoding="utf-8") as file:
                metric_rows = list(csv.DictReader(file))
            for episode in episodes:
                seeds = [int(row["seed"]) for row in metric_rows
                         if int(row["training_episode"]) == episode]
                if sorted(seeds) != sorted(args.validation_seeds):
                    raise ValueError(f"missing/duplicate validation seeds: {metrics_path}, episode {episode}")
            selection_path = metrics_path.parent / "selected.json"
            checkpoint = select_from_metrics(metrics_path, baseline_csv,
                metrics_path.parent / "snapshots", episodes, selection_path)
            record = json.loads(selection_path.read_text(encoding="utf-8"))
            record["sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            records.append(record)
            selected.append((record["controller"], checkpoint))
        # Freeze choices before observing any test scenario.
        (root / "selected-checkpoints.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

        outputs = {}
        scenarios = [("primary", 1.4, 6, 20), ("low", 1.4, 10, 30),
                     ("high", 1.4, 4, 12), ("slow", 0.9, 6, 20)]
        for name, speed, period, interval in scenarios:
            _status(root, f"test_{name}")
            output = evaluate_scenario(name, selected, args.test_seeds,
                args.evaluation_duration, speed, period, interval, root, args.workers)
            if name == "slow":
                programs = ElementTree.parse(GENERATED / "network.net.xml").getroot().findall(".//tlLogic")
                # Clearance is all-red for vehicles; pedestrian links can be green.
                clearance = [int(program.findall("phase")[index].get("duration"))
                             for program in programs for index in (2, 5)]
                if not clearance or set(clearance) != {8}:
                    raise ValueError(f"expected 8-second all-red clearance, got {clearance}")
            with output.open(newline="", encoding="utf-8") as file:
                scenario_rows = list(csv.DictReader(file))
                if len(scenario_rows) != (2+len(selected))*len(args.test_seeds):
                    raise ValueError(f"incomplete scenario output: {output}")
            expected_people = 12 * sum(now % interval == 0
                for now in range(10, args.evaluation_duration))
            if any(int(row.get("pedestrian_departed") or row.get("pedestrian_arrivals"))
                   != expected_people for row in scenario_rows):
                raise ValueError(f"mismatched pedestrian demand in {name}")
            outputs[name] = output
            if name == "primary":
                _combine([output], root / "final-primary-results.csv")
            elif name == "high":
                _combine([outputs["low"], output], root / "final-demand-sweep.csv")
            elif name == "slow":
                _combine([output], root / "final-slow-pedestrian-results.csv")
        selected_paths = {path.resolve() for _, path in selected}
        manifest = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in root.rglob("*") if path.is_file()
                    and path.name not in ("pilot-status.json", "pilot.log")
                    and (path.suffix in (".csv", ".json", ".log")
                         or path.resolve() in selected_paths)}
        (root / "artifact-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _status(root, "complete", selected_checkpoints=len(selected), artifacts=len(manifest))
    except BaseException as error:
        _status(root, "failed", error=str(error), error_type=type(error).__name__)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[100, 200, 300])
    parser.add_argument("--validation-seeds", nargs="+", type=int, default=[21, 22, 23])
    parser.add_argument("--test-seeds", nargs="+", type=int, default=[11, 12, 13])
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--snapshot-interval", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=3600)
    parser.add_argument("--evaluation-duration", type=int, default=3600)
    main(parser.parse_args())
