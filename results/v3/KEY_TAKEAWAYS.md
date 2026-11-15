# V3 findings



V3 does not establish a consistent MARL improvement over pedestrian-actuated control.

IPPO combined changes primary vehicle loss by +4.54%, mean pedestrian wait by -4.83%, and worst-crossing p95 by +5.14% relative to actuated control. These are point estimates, not significance tests.

MAPPO vehicle-only matches fixed-time recorded metrics in all 36 learned-policy evaluations. Zero-width training-seed intervals reflect identical averages on the fixed traffic test set, not zero traffic-seed variability.

Only 2/12 selected runs meet all validation crossing thresholds; ten use the predeclared normalized-violation fallback. Ten selections are at episodes 50-150, one at 250, and one at 300.

Combined rewards reduce some J0/J2 crossing means, but J1 excesses remain. Full crossing means and confidence intervals for all scenarios are in per-crossing-p95.csv.

V3 uses fresh seeds 41/42/43 after a historical audit found prior use of 31/32/33. V2 uses a different test set, and several settings change together, so v2-comparison.csv is descriptive.

All training logs, six snapshots per run, selected checkpoints, raw test JSONs/CSVs, and individual maximum-wait metrics are saved locally.

348 downloaded manifest entries verified; 368 archived V2 files unchanged; all pods and temporary storage deleted.

Conservative V3 runtime/storage bound: $1.69. Posted billing is incomplete.
