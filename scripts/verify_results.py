"""Verify the published experiment results without running SUMO or training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import zipfile
from pathlib import Path

if __package__:
    from .report_v3_results import read_csv
    from .select_checkpoint import _mean_crossing_p95
    from .summarize_proposal_pilot import summarize
else:
    from report_v3_results import read_csv
    from select_checkpoint import _mean_crossing_p95
    from summarize_proposal_pilot import summarize


RESULT_FILES = (
    "final-primary-results.csv",
    "final-demand-sweep.csv",
    "final-slow-pedestrian-results.csv",
)


def contained_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path leaves the results directory: {relative}")
    return path


def require_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError(f"Metric mismatch for {label}: {actual} versus {expected}")


def verify_summary(paths: list[Path], saved_path: Path) -> None:
    recomputed = summarize(paths)
    saved = {(row["scenario"], row["controller"], row["metric"]): row for row in read_csv(saved_path)}
    if set(saved) != {(row["scenario"], row["controller"], row["metric"]) for row in recomputed}:
        raise ValueError(f"Summary metric coverage mismatch: {saved_path}")
    for row in recomputed:
        key = (row["scenario"], row["controller"], row["metric"])
        for metric in ("mean", "ci95_half_width", "ci95_low", "ci95_high", "maximum_observed"):
            require_close(row[metric], saved[key][metric], f"{saved_path.parent.name}/{key}/{metric}")
        for field in ("independent_n", "evaluation_n", "uncertainty_unit"):
            if str(row[field]) != saved[key][field]:
                raise ValueError(f"Summary uncertainty unit mismatch: {saved_path}/{key}/{field}")


def verify_selections(directory: Path, selections: list[dict]) -> int:
    thresholds = _mean_crossing_p95(read_csv(directory / "validation-baseline.csv"))
    metrics_files = {path.parent.name: path for path in directory.rglob("checkpoint-metrics.csv")}
    if set(metrics_files) != {selection["controller"] for selection in selections}:
        raise ValueError(f"Incomplete validation records in {directory.name}")
    validation_count = 0
    for selection in selections:
        rows = read_csv(metrics_files[selection["controller"]])
        validation_count += len(rows)
        candidates = []
        for episode in sorted({int(row["training_episode"]) for row in rows}):
            subset = [row for row in rows if int(row["training_episode"]) == episode]
            if sorted(int(row["seed"]) for row in subset) != [21, 22, 23]:
                raise ValueError(f"Invalid validation seeds for {selection['controller']}")
            crossing = _mean_crossing_p95(subset)
            excess = {name: crossing[name] - limit for name, limit in thresholds.items()}
            candidates.append({
                "episode": episode,
                "vehicle_time_loss": statistics.mean(float(row["vehicle_time_loss_seconds"]) for row in subset),
                "max_crossing_p95_violation": max(excess.values()),
                "normalized_violation": sum(max(value, 0) / max(thresholds[name], 1) for name, value in excess.items()),
            })
        feasible = [candidate for candidate in candidates if candidate["max_crossing_p95_violation"] <= 0]
        chosen = min(feasible or candidates, key=lambda candidate: (
            candidate["vehicle_time_loss"] if feasible else candidate["normalized_violation"],
            candidate["vehicle_time_loss"],
        ))
        if chosen["episode"] != selection["episode"]:
            raise ValueError(f"Frozen selection mismatch for {selection['controller']}")
        for metric in ("vehicle_time_loss", "max_crossing_p95_violation", "normalized_violation"):
            require_close(chosen[metric], selection[metric], f"{selection['controller']}/{metric}")
        checkpoint = contained_path(directory, selection["checkpoint"])
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != selection["sha256"]:
            raise ValueError(f"Selected checkpoint checksum mismatch: {checkpoint}")
    return validation_count


def verify(root: Path) -> None:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for relative, record in manifest["files"].items():
        path = contained_path(root, relative)
        payload = path.read_bytes()
        if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError(f"Published artifact checksum mismatch: {relative}")

    total_tests = total_validation = total_checkpoints = 0
    reference_baselines = {}
    for version, protocol in manifest["versions"].items():
        directory = root / version
        selections = json.loads((directory / "selected-checkpoints.json").read_text(encoding="utf-8"))
        if len(selections) != protocol["selected_checkpoints"]:
            raise ValueError(f"Missing frozen selections in {version}")
        paths = [directory / name for name in RESULT_FILES]
        verify_summary(paths, directory / "final-summary.csv")

        rows = [row for path in paths for row in read_csv(path)]
        expected_rows = (len(selections) + 2) * 3 * 4
        if len(rows) != expected_rows:
            raise ValueError(f"Incomplete accepted test records in {version}")
        groups = {}
        for row in rows:
            groups.setdefault((row["scenario"], row["scenario_controller"]), []).append(int(row["seed"]))
            if int(row["duration_seconds"]) != 3600:
                raise ValueError(f"Unexpected test duration in {version}")
            if row.get("checkpoint") and not contained_path(directory, row["checkpoint"]).is_file():
                raise FileNotFoundError(f"Missing evaluation checkpoint in {version}")
            if row["scenario_controller"] in {"fixed_time", "pedestrian_actuated"}:
                key = (row["scenario"], row["scenario_controller"], row["seed"])
                if version == "v3":
                    reference_baselines[key] = row
                if version in {"v4", "v5"} and row != reference_baselines[key]:
                    raise ValueError(f"Reused baseline differs from V3 for {version}/{key}")
        if any(sorted(seeds) != protocol["test_seeds"] for seeds in groups.values()):
            raise ValueError(f"Missing or duplicate test seeds in {version}")
        with zipfile.ZipFile(directory / "source.zip") as archive:
            if set(archive.namelist()) != set(protocol["source_files"]):
                raise ValueError(f"Source snapshot coverage mismatch in {version}")
            for name, expected in protocol["source_files"].items():
                if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                    raise ValueError(f"Source snapshot checksum mismatch for {version}/{name}")
        validation_count = verify_selections(directory, selections)
        total_validation += validation_count
        total_tests += len(rows)
        total_checkpoints += len(selections)
        print(f"{version}: {len(rows)} test rows, {validation_count} validations, {len(selections)} frozen selections verified")

    bc_rows = read_csv(root / "v5" / "bc-primary-results.csv")
    verify_summary([root / "v5" / "bc-primary-results.csv"], root / "v5" / "bc-primary-summary.csv")
    if len(bc_rows) != 18 or {row["seed"] for row in bc_rows} != {"41", "42", "43"}:
        raise ValueError("Incomplete V5 BC reference records")
    for row in bc_rows:
        if not contained_path(root / "v5", row["checkpoint"]).is_file():
            raise FileNotFoundError("Missing V5 BC reference checkpoint")
    print(f"PASS: {len(manifest['files'])} artifact hashes, {total_tests} main test rows, {len(bc_rows)} BC reference rows, {total_validation} validations, {total_checkpoints} selected checkpoints")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    arguments = parser.parse_args()
    verify(arguments.root.resolve())
