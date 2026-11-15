"""Report frozen V3 evaluations, crossing tails, and descriptive V2 comparisons."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re
import statistics

if __package__:
    from .summarize_proposal_pilot import summarize
else:
    from summarize_proposal_pilot import summarize

FAMILIES = ["fixed_time", "pedestrian_actuated", "ippo-vehicle",
            "ippo-vehicle_pedestrian", "mappo-vehicle", "mappo-vehicle_pedestrian"]
SCENARIOS = ("primary", "low", "high", "slow")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(root, v2):
    plan = json.loads((root / 'RUN_PLAN.json').read_text())
    test_seeds = plan['test_seeds']
    assert len(test_seeds)==3 and not plan.get('test_seed_confirmation_pending',False)
    paths = [root / name for name in ("final-primary-results.csv", "final-demand-sweep.csv",
                                     "final-slow-pedestrian-results.csv")]
    summary = summarize(paths)
    write_csv(root / "final-summary.csv", summary)
    raw = [row for path in paths for row in read_csv(path)]
    assert len(raw) == 168 and all(int(row["seed"]) in test_seeds for row in raw)
    groups = defaultdict(lambda: defaultdict(list))
    for row in raw:
        name = row["scenario_controller"]
        match = re.fullmatch(r"(.+)-actions2-seed(100|200|300)", name)
        family, unit = (match[1], int(match[2])) if match else (name, int(row["seed"]))
        groups[(row["scenario"], family)][unit].append(row)
    crossings = []
    for (scenario, family), units in sorted(groups.items()):
        for junction in range(3):
            for index, approach in enumerate(("North", "East", "South", "West")):
                crossing = f":J{junction}_c{index}"
                samples = [statistics.mean(json.loads(r["pedestrian_p95_by_crossing"])[crossing]
                                           for r in rows) for rows in units.values()]
                half = 4.302652729911275 * statistics.stdev(samples) / math.sqrt(3)
                crossings.append({"scenario": scenario, "controller": family, "crossing": crossing,
                    "approach": approach, "independent_n": 3,
                    "evaluation_n": sum(len(rows) for rows in units.values()),
                    "mean_p95_seconds": statistics.mean(samples), "ci95_half_width": half,
                    "maximum_observed_p95_seconds": max(json.loads(r["pedestrian_p95_by_crossing"])[crossing]
                                                       for rows in units.values() for r in rows)})
    baseline = {(r["scenario"], r["crossing"]): r["mean_p95_seconds"]
                for r in crossings if r["controller"] == "pedestrian_actuated"}
    for row in crossings:
        actuated = baseline[(row["scenario"], row["crossing"])]
        row["actuated_mean_p95_seconds"] = actuated
        row["excess_over_actuated_seconds"] = row["mean_p95_seconds"] - actuated
        row["excess_over_actuated_percent"] = 100 * (row["mean_p95_seconds"] / actuated - 1) if actuated else ""
    write_csv(root / "per-crossing-p95.csv", crossings)
    primary = []
    for j in range(3):
        for c in range(4):
            crossing = f":J{j}_c{c}"
            primary.append({"crossing": crossing, **{f: next(r["mean_p95_seconds"] for r in crossings
                if (r["scenario"], r["controller"], r["crossing"]) == ("primary", f, crossing)) for f in FAMILIES}})
    write_csv(root / "primary-crossing-p95.csv", primary)
    old = {(r["scenario"], r["controller"], r["metric"]): r for r in read_csv(v2 / "final-summary.csv")}
    comparison = []
    actuated = {(r["scenario"], r["metric"]): r["mean"] for r in summary if r["controller"] == "pedestrian_actuated"}
    baseline_comparison = []
    for row in summary:
        key = (row["scenario"], row["controller"], row["metric"])
        base = actuated[(row["scenario"], row["metric"])]
        baseline_comparison.append({"scenario": key[0], "controller": key[1], "metric": key[2],
            "actuated_mean": base, "controller_mean": row["mean"],
            "percent_change": 100*(row["mean"]/base-1) if base else ""})
        if key in old:
            value = float(old[key]["mean"])
            comparison.append({"scenario": key[0], "controller": key[1], "metric": key[2],
                "v2_mean": value, "v3_mean": row["mean"],
                "percent_change": 100*(row["mean"]/value-1) if value else "",
                "interpretation": "descriptive only: different test seeds and several simultaneous changes"})
    write_csv(root / "v2-comparison.csv", comparison)
    write_csv(root / "actuated-comparison.csv", baseline_comparison)
    selected = json.loads((root / "selected-checkpoints.json").read_text())
    feasible = sum(r["max_crossing_p95_violation"] <= 0 for r in selected)
    lines = ["# V3 Observation-Fixed & Fairness-Calibrated MARL", "",
        "Twelve runs from scratch, 300 episodes x 3600s, randomized training demand, training seeds 100/200/300.",
        "21 features: vehicle queues, approach waiting counts, current oldest waits, crossing occupancy, phase indicators/age, neighbor queues. Walking-area queues use the actual next crossing, with assigned start-edge fallback (explicitly approved after a local SUMO probe).",
        "Two actions: hold or safe switch. Existing minimum-green and clearance masks remain active. Hold lets the programmed phase timer advance; it does not issue extension actions.",
        "Vehicle scale 20.8; pedestrian scale 20; switch penalty 8. Combined reward additionally penalizes 0.5*sum((current crossing max wait/60)^2), summed across 12 crossings once per five-second step. Vehicle-only reward omits pedestrian and urgency terms.",
        "The 8.0 switch cost is a reward coefficient, not a measurement of physical clearance duration.",
        "Six snapshots at episodes 50/100/150/200/250/300 were validated on seeds 21/22/23 under primary demand. Frozen checkpoint selection precedes all final test evaluations.",
        "Final test seeds: " + '/'.join(map(str,test_seeds)) + ("; verified absent from prior saved evaluation CSVs." if plan.get('test_seeds_fresh') else "; previously evaluated seeds, not a fresh confirmatory set."),
        f"{feasible}/12 selected runs satisfy every crossing's mean validation p95 threshold. Other selections use the predeclared normalized positive violation fallback.",
        "Means +/- 95% Student-t CI, n=3, df=2. Learned policies average test seeds within each training seed before constructing intervals. Baseline intervals use the three test seeds. Intervals are descriptive and unadjusted for multiple comparisons.",
        "A zero-width learned-policy interval means the three training-seed averages are identical on this fixed test set. It does not mean there is no variation between traffic test seeds. MAPPO-vehicle's recorded metrics match fixed-time control in every evaluated scenario and test seed.",
        "Vehicle loss includes active trips and is divided by departed vehicles. Throughput is completed trips per hour. Unfinished counts exclude demand not inserted. Pedestrian waits include active pedestrians and are observed to the cutoff; active waits are right-censored.",
        "Crossing p95 retains the V2 route-assigned metric definition, including stopped time over that pedestrian's observed trip. It is not a measurement exclusively of red-signal waiting at one physical crossing.",
        "Primary 1.4/6/20, low 1.4/10/30, high 1.4/4/12, slow 0.9/6/20 (speed/vehicle period/pedestrian interval); slow all-red is 8 seconds. All tests last 3600 seconds.",
        "V2 and V3 change observation, action count, reward scaling, urgency and switching cost together, and use different test seeds. Their comparison is an ablation between configurations, not a causal estimate of any individual change.", ""]
    metrics = ["vehicle_mean_time_loss_seconds", "vehicle_throughput_per_hour", "vehicle_unfinished",
               "pedestrian_average_waiting_seconds", "worst_crossing_p95_waiting_seconds", "pedestrian_unfinished"]
    for scenario in SCENARIOS:
        lines += [f"## {scenario}", "",
            "| Controller | Vehicle loss (s/veh) | Throughput (veh/h) | Unfinished vehicles | Mean pedestrian wait (s) | Worst crossing p95 (s) | Unfinished pedestrians |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for family in FAMILIES:
            values = [next(r for r in summary if (r["scenario"], r["controller"], r["metric"]) == (scenario, family, m)) for m in metrics]
            lines.append("| " + " | ".join([family] + [f"{v['mean']:.2f} +/- {v['ci95_half_width']:.2f}" for v in values]) + " |")
        lines.append("")
    lines += ["## Frozen selections", "", "| Controller | Episode | Validation feasible |", "|---|---:|---|"]
    for row in selected:
        lines.append(f"| {row['controller']} | {row['episode']} | {row['max_crossing_p95_violation'] <= 0} |")
    lines += ["", "## Primary crossing p95 (seconds)", "",
        "| Crossing | " + " | ".join(FAMILIES) + " |", "|---|" + "---:|"*6]
    for row in primary:
        lines.append("| " + " | ".join([row["crossing"]] + [f"{row[f]:.1f}" for f in FAMILIES]) + " |")
    (root / "FINAL_RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    result = {"summary_rows": len(summary), "crossing_rows": len(crossings),
              "accepted_test_rows": len(raw), "validation_feasible_runs": feasible}
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--v2-root", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.v2_root)
