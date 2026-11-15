"""Read V2 artifacts and instrument three local, frozen-policy SUMO rollouts.

Outputs go outside final-v2-results; accepted experiment artifacts are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import time
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FAMILIES = ["fixed_time", "pedestrian_actuated", "ippo-vehicle",
            "ippo-vehicle_pedestrian", "mappo-vehicle", "mappo-vehicle_pedestrian"]
LEARNED = FAMILIES[2:]
JUNCTIONS = ("J0", "J1", "J2")
ACTIONS = ("hold", "switch", "extend_5", "extend_10")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def family(name):
    return re.sub(r"-actions4-seed\d+$", "", name)


def saved_diagnostics(source, output):
    rows = read_csv(source / "final-primary-results.csv")
    assert len(rows) == 42
    by_family = {f: [r for r in rows if family(r["scenario_controller"]) == f]
                 for f in FAMILIES}
    for f, group in by_family.items():
        assert len(group) == (3 if f in FAMILIES[:2] else 9)
        assert set(int(r["seed"]) for r in group) == {11, 12, 13}
    crossings, failures = [], []
    for junction in JUNCTIONS:
        for index, approach in enumerate(("North", "East", "South", "West")):
            crossing = f":{junction}_c{index}"
            means = {f: statistics.mean(json.loads(r["pedestrian_p95_by_crossing"])[crossing]
                                        for r in group)
                     for f, group in by_family.items()}
            crossings.append({"crossing": crossing, "approach": approach,
                              **{f: means[f] for f in FAMILIES[1:]}})
            for f in LEARNED:
                delta = means[f] - means["pedestrian_actuated"]
                failures.append({"controller": f, "crossing": crossing,
                                 "approach": approach, "p95_seconds": means[f],
                                 "actuated_p95_seconds": means["pedestrian_actuated"],
                                 "excess_seconds": delta,
                                 "excess_percent": 100 * delta / means["pedestrian_actuated"]})
    failures.sort(key=lambda row: row["excess_seconds"], reverse=True)
    summary = read_csv(source / "final-summary.csv")
    extremes = []
    for scenario in ("primary", "high", "slow"):
        for f in FAMILIES:
            unfinished = next(r for r in summary if r["scenario"] == scenario
                              and r["controller"] == f and r["metric"] == "pedestrian_unfinished")
            p95 = next(r for r in summary if r["scenario"] == scenario
                       and r["controller"] == f and r["metric"] == "worst_crossing_p95_waiting_seconds")
            extremes.append({"scenario": scenario, "controller": f,
                             "pedestrian_unfinished_mean": float(unfinished["mean"]),
                             "pedestrian_unfinished_ci95": float(unfinished["ci95_half_width"]),
                             "pedestrian_unfinished_maximum_observed": float(unfinished["maximum_observed"]),
                             "maximum_observed_crossing_p95_seconds": float(p95["maximum_observed"]),
                             "maximum_individual_wait_seconds": None,
                             "maximum_individual_wait_status": "not recorded in saved V2 metrics"})
    # Verify that neither the raw JSONs nor the summary contains an individual max.
    for scenario in ("primary", "high", "slow"):
        for path in (source / scenario).glob("*.json"):
            raw = json.loads(path.read_text())
            assert not any("max" in key.lower() and "wait" in key.lower() for key in raw)
    write_csv(output / "primary-crossing-p95.csv", crossings)
    write_csv(output / "crossing-excess-ranked.csv", failures)
    write_csv(output / "saved-unfinished-and-extremes.csv", extremes)
    return {"crossings": crossings, "ranked_excess": failures, "saved_extremes": extremes,
            "primary_rows": len(rows), "summary_rows": len(summary)}


class Observer:
    def __init__(self, traci):
        self.traci = traci
        self.phase_seconds = {j: Counter() for j in JUNCTIONS}
        self.transitions = Counter()
        self.green_ends = Counter()
        self.green_start = {j: 0.0 for j in JUNCTIONS}
        self.green_durations = {j: [] for j in JUNCTIONS}
        self.action_counts = {j: Counter() for j in JUNCTIONS}
        self.green_action_counts = {j: Counter() for j in JUNCTIONS}
        self.commands = {j: Counter() for j in JUNCTIONS}
        self.waits = Counter()
        self.observation_slots = Counter()
        self.environment = None
        self.current_actions = {}
        self.checkpoints = []
        self.decision_rows = []

    def sample_step(self, original, *args, **kwargs):
        t = self.traci
        before_time = t.simulation.getTime()
        before = {j: t.trafficlight.getPhase(j) for j in JUNCTIONS}
        result = original(*args, **kwargs)
        now = t.simulation.getTime()
        for j in JUNCTIONS:
            phase = t.trafficlight.getPhase(j)
            self.phase_seconds[j][before[j]] += now - before_time
            if phase != before[j]:
                self.transitions[j] += 1
                if before[j] in (0, 3):
                    self.green_ends[j] += 1
                    self.green_durations[j].append(now - self.green_start[j])
                if phase in (0, 3):
                    self.green_start[j] = now
        for person in t.person.getIDList():
            if t.person.getSpeed(person) < 0.1:
                self.waits[person] += now - before_time
        if int(now) % 600 == 0:
            print(f"  simulated_seconds={int(now)}", flush=True)
        return result

    def command(self, original, junction, duration):
        t = self.traci
        remaining = max(0.0, t.trafficlight.getNextSwitch(junction) - t.simulation.getTime())
        if duration == 0:
            forced = self.environment is not None and self.environment.phase_age[junction] >= 45
            self.commands[junction]["forced_max" if forced else "early_termination"] += 1
        elif duration > remaining:
            self.commands[junction]["effective_" + ACTIONS[self.current_actions[junction]]] += 1
        return original(junction, duration)

    def decision(self, actions, observation):
        t = self.traci
        outside = Counter()
        for person in t.person.getIDList():
            _, crossing = self.environment.pedestrian_meta[person]
            if t.person.getSpeed(person) < 0.1 and t.person.getRoadID(person) != crossing:
                outside[crossing] += 1
        for index, j in enumerate(JUNCTIONS):
            action = int(actions[index])
            self.current_actions[j] = action
            self.action_counts[j][action] += 1
            phase = t.trafficlight.getPhase(j)
            if phase in (0, 3):
                self.green_action_counts[j][action] += 1
            values = observation[index, 4:8].tolist()
            for c, value in enumerate(values):
                self.observation_slots["total"] += 1
                if value > 0:
                    self.observation_slots["nonzero_observed"] += 1
                if outside[f":{j}_c{c}"] > 0:
                    self.observation_slots["outside_waiting"] += 1
                    if value == 0:
                        self.observation_slots["outside_waiting_with_zero_observation"] += 1
            self.decision_rows.append({"time": t.simulation.getTime(), "junction": j,
                                       "phase": phase, "action": ACTIONS[action],
                                       "phase_age_feature": float(observation[index, 10]),
                                       "pedestrian_features": json.dumps(values)})

    @contextmanager
    def installed(self):
        original_step = self.traci.simulationStep
        original_duration = self.traci.trafficlight.setPhaseDuration
        self.traci.simulationStep = lambda *a, **k: self.sample_step(original_step, *a, **k)
        self.traci.trafficlight.setPhaseDuration = lambda j, d: self.command(original_duration, j, d)
        try:
            yield
        finally:
            self.traci.simulationStep = original_step
            self.traci.trafficlight.setPhaseDuration = original_duration

    def result(self):
        rows = []
        for j in JUNCTIONS:
            counts = self.action_counts[j]
            n = sum(counts.values())
            rows.append({"junction": j, "all_phase_transitions": self.transitions[j],
                         "green_terminations": self.green_ends[j],
                         "mean_completed_green_seconds": statistics.mean(self.green_durations[j]),
                         "min_completed_green_seconds": min(self.green_durations[j]),
                         "max_completed_green_seconds": max(self.green_durations[j]),
                         "phase_seconds": dict(self.phase_seconds[j]),
                         "decisions": n,
                         "action_percent": {ACTIONS[a]: 100 * counts[a] / n if n else None for a in range(4)},
                         "green_action_counts": dict(self.green_action_counts[j]),
                         "effective_commands": dict(self.commands[j])})
            assert sum(self.phase_seconds[j].values()) == 3600
        return {"junctions": rows, "maximum_individual_wait_seconds": max(self.waits.values(), default=0),
                "observed_total_person_wait_seconds": sum(self.waits.values()),
                "observation_slots": dict(self.observation_slots)}


def rollout(source, output, name, configuration):
    import torch
    from scripts import run_baseline as baseline
    from scripts.train_marl import Actor, SignalEnvironment
    try:
        import libsumo as traci
    except ImportError:
        from scripts.run_simulation import _import_traci
        traci = _import_traci()
    observer = Observer(traci)
    start = time.monotonic()
    print(f"Starting local rollout: {name}", flush=True)
    with observer.installed():
        if name == "pedestrian_actuated":
            original_generated = baseline.GENERATED
            baseline.GENERATED = configuration.parent
            try:
                metrics = baseline.run(11, 3600, output / "actuated-local-metrics.json", False,
                                       "pedestrian_actuated", 37, 1.4, 6, 20)
            finally:
                baseline.GENERATED = original_generated
            checkpoint, episode = None, None
        else:
            selected = json.loads((source / "selected-checkpoints-local.json").read_text())
            record = next(r for r in selected if r["controller"] == name + "-actions4-seed100")
            checkpoint = source / "checkpoints" / record["controller"] / f"episode-{record['episode']}.pt"
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == record["sha256"]
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            metadata = saved["metadata"]
            actors = [Actor(4) for _ in JUNCTIONS]
            for actor, state in zip(actors, saved["actors"]):
                actor.load_state_dict(state)
                actor.eval()
            environment = SignalEnvironment(seed=11, duration=3600,
                reward_mode=metadata["reward_mode"], vehicle_weight=metadata["vehicle_weight"],
                pedestrian_weight=metadata["pedestrian_weight"], switch_penalty=metadata["switch_penalty"],
                observation_mode=metadata["observation_mode"], action_size=4, green_duration=37,
                pedestrian_speed=1.4, pedestrian_interval=20, configuration=configuration)
            observer.environment = environment
            observation = environment.reset()
            try:
                done = False
                while not done:
                    masks = environment.action_masks()
                    with torch.no_grad():
                        actions = torch.stack([actor.distribution(observation[i], masks[i]).probs.argmax()
                                               for i, actor in enumerate(actors)])
                    observer.decision(actions, observation)
                    observation, _, done = environment.step(actions)
                metrics = environment.metrics()
            finally:
                environment.close()
            episode = record["episode"]
            assert all(sum(observer.action_counts[j].values()) == 720 for j in JUNCTIONS)
    result = {"controller": name, "training_seed": None if checkpoint is None else 100,
              "checkpoint": str(checkpoint) if checkpoint else None, "selected_episode": episode,
              "evaluation_seed": 11, "duration": 3600, "elapsed_seconds": time.monotonic() - start,
              **observer.result(), "metrics": metrics}
    assert result["observed_total_person_wait_seconds"] == metrics["pedestrian_waiting_seconds"]
    saved_name = name if checkpoint is None else name + "-actions4-seed100"
    accepted = json.loads((source / "primary" / f"{saved_name}-11.json").read_text())
    compare_keys = ["pedestrian_unfinished", "pedestrian_waiting_seconds", "vehicle_completed",
                    "vehicle_time_loss_seconds", "pedestrian_average_waiting_seconds"]
    result["saved_replay_differences"] = {k: float(metrics[k]) - float(accepted[k]) for k in compare_keys}
    result["saved_replay_crossing_p95_max_abs_difference"] = max(
        abs(metrics["pedestrian_p95_by_crossing"][c] - accepted["pedestrian_p95_by_crossing"][c])
        for c in accepted["pedestrian_p95_by_crossing"])
    if checkpoint:
        vehicle_term = metrics["vehicle_waiting_seconds"] / 20.8
        pedestrian_term = metrics["pedestrian_waiting_seconds"] / 81.35
        switches = sum(observer.commands[j]["early_termination"] for j in JUNCTIONS)
        result["reward_components"] = {"vehicle_delay_term": vehicle_term,
            "pedestrian_delay_term_if_included": pedestrian_term, "voluntary_switch_penalty": 1.5 * switches}
    write_csv(output / f"{name}-decisions.csv", observer.decision_rows)
    (output / f"{name}-rollout.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Completed local rollout: {name}; elapsed={result['elapsed_seconds']:.1f}s", flush=True)
    return result


def write_report(data, output):
    saved = data["saved"]
    lines = ["# Local V2 diagnostics", "",
        "Primary saved means: 3 test seeds per training seed, 3 training seeds per learned family; baselines have 3 test seeds.",
        "Bold crossing values exceed the same crossing's actuated test mean. These are descriptive comparisons.", "",
        "| Crossing | Approach | Actuated | IPPO vehicle | IPPO combined | MAPPO vehicle | MAPPO combined |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in saved["crossings"]:
        base = row["pedestrian_actuated"]
        values = [f"{base:.1f}"]
        for f in LEARNED:
            value = f"{row[f]:.1f}"
            values.append(f"**{value}**" if row[f] > base else value)
        lines.append("| " + " | ".join([row["crossing"], row["approach"]] + values) + " |")
    lines += ["", "## Deterministic local primary rollouts", "",
        "Seed 11, 3600 seconds; selected MAPPO training-seed-100 checkpoints: vehicle episode 50, combined episode 250.",
        "Phase transitions include green/yellow/all-red changes. Green terminations count exits from phase 0 or 3.",
        "Mean green duration excludes the unfinished final green interval. Action shares include all 720 decisions per junction.", "",
        "| Controller | Junction | Phase transitions | Green terminations | Mean green (s) | Hold % | Switch % | +5s % | +10s % |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for run in data["rollouts"]:
        for row in run["junctions"]:
            action = ["n.a." if row["action_percent"][a] is None else f"{row['action_percent'][a]:.2f}"
                      for a in ACTIONS]
            lines.append("| " + " | ".join([run["controller"], row["junction"],
                str(row["all_phase_transitions"]), str(row["green_terminations"]),
                f"{row['mean_completed_green_seconds']:.2f}"] + action) + " |")
    lines += ["", "## Saved unfinished counts and tail extremes", "",
        "Individual maximum waits were not recorded in saved V2 CSVs or JSONs and cannot be recovered from p95 summaries.",
        "The final column is the maximum CROSSING P95 across saved evaluations, not an individual maximum.",
        "Unfinished includes moving pedestrians. Counts are mean +/- 95% CI, using the existing three-seed summary.", "",
        "| Scenario | Controller | Unfinished mean +/- CI | Max unfinished | Max crossing p95 (s) |",
        "|---|---|---:|---:|---:|"]
    for row in saved["saved_extremes"]:
        lines.append(f"| {row['scenario']} | {row['controller']} | "
                     f"{row['pedestrian_unfinished_mean']:.2f} +/- {row['pedestrian_unfinished_ci95']:.2f} | "
                     f"{row['pedestrian_unfinished_maximum_observed']:.0f} | "
                     f"{row['maximum_observed_crossing_p95_seconds']:.1f} |")
    lines += ["", "Exact observed individual maximum waits from the three new local rollouts:", "",
        "| Controller | Max individual wait (s) | Unfinished | Replay matched saved metrics |",
        "|---|---:|---:|---|"]
    for run in data["rollouts"]:
        same = all(v == 0 for v in run["saved_replay_differences"].values()) and run["saved_replay_crossing_p95_max_abs_difference"] == 0
        lines.append(f"| {run['controller']} | {run['maximum_individual_wait_seconds']:.0f} | "
                     f"{run['metrics']['pedestrian_unfinished']} | {same} |")
    lines += ["", "The maximum wait includes completed and still-active pedestrians' observed cumulative stopped time through 3600 seconds.",
              "Full action commands, reward components, observation coverage, checksums and replay differences are in diagnostics.json.",
              "No accepted V2 artifact or checkpoint was modified. No cloud resources were used.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "scenario/generated/final-v2-results")
    parser.add_argument("--output", type=Path, default=ROOT / "scenario/generated/v2-local-diagnostics")
    parser.add_argument("--saved-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    saved = saved_diagnostics(args.source, args.output)
    (args.output / "saved-diagnostics.json").write_text(json.dumps(saved, indent=2), encoding="utf-8")
    print("Saved-artifact diagnostics complete", flush=True)
    if args.saved_only:
        return
    deps = args.output / "python-deps"
    if deps.is_dir():
        sys.path.insert(0, str(deps.resolve()))
    import torch
    torch.set_num_threads(1)
    # The local eclipse-sumo wheel bundles TraCI under sumo/tools. Resolve it
    # before libsumo's import can adjust SUMO_HOME to its own package directory.
    from scripts.run_simulation import _import_traci
    _import_traci()
    from scripts.build_network import build
    network = build(37, 1.4, 6, args.output / "primary-network")
    configuration = network.parent / "simulation.sumocfg"
    # Confirm approach labels against generated network geometry, not route names.
    tree = ET.parse(configuration.parent / "network.net.xml")
    geometry = {e.get("id"): e.get("crossingEdges") for e in tree.findall("edge")
                if e.get("function") == "crossing"}
    assert all(f"N{j[1]}" in geometry[f":{j}_c0"] and f"S{j[1]}" in geometry[f":{j}_c2"]
               for j in JUNCTIONS)
    results = [rollout(args.source, args.output, name, configuration)
               for name in ("pedestrian_actuated", "mappo-vehicle", "mappo-vehicle_pedestrian")]
    data = {"saved": saved, "rollouts": results, "network_crossing_edges": geometry,
            "torch_version": torch.__version__}
    (args.output / "diagnostics.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    write_report(data, args.output)
    print("All local diagnostics complete", flush=True)


if __name__ == "__main__":
    main()
