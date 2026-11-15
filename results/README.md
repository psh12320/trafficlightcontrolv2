# Published experiment results

The V1-V5 review records contain 696 accepted main test evaluations, 18 V5
behavior-cloning reference evaluations, 792 checkpoint-validation evaluations,
and 48 frozen selected policies. The main repository README displays the V3
Primary benchmark and links to the full scenario tables. Final figures and captions are in
[report_figures](../report_figures/README.md).

| Version | Training episodes per run | Actor inputs | Actions | Traffic test seeds | Selected policies | Validation-feasible selections |
|---|---:|---:|---:|---|---:|---:|
| [V1](v1/FINAL_RESULTS.md) | 60 | 13 | 4 | 11, 12, 13 | 12 | 1/12 |
| [V2](v2/FINAL_RESULTS.md) | 300 | 13 | 4 | 11, 12, 13 | 12 | 2/12 |
| [V3](v3/FINAL_RESULTS.md) | 300 | 21 | 2 | 41, 42, 43 | 12 | 2/12 |
| [V4](v4/FINAL_RESULTS.md) | 400 | 25 | 2 | 41, 42, 43 | 6 | 0/6 |
| [V5](v5/FINAL_RESULTS.md) | 20 BC + 5 critic warm-up + 250 PPO | 75 | 2 | 41, 42, 43 | 6 | 0/6 |

V1-V3 each train IPPO and MAPPO with vehicle and combined objectives. V4-V5
train MAPPO with the two objectives. Every learned configuration uses training
seeds 100, 200, and 300. Selection uses validation seeds 21, 22, and 23. V2
reuses the V1 traffic test set. V4-V5 reuse the V3 test set for matched
exploratory comparisons.

## Files for each version

- `FINAL_RESULTS.md` contains the recorded experiment report.
- `final-summary.csv` contains means, confidence intervals, uncertainty units,
  and maximum observed metric values.
- `final-primary-results.csv`, `final-demand-sweep.csv`, and
  `final-slow-pedestrian-results.csv` contain all accepted main test records.
  Each row includes the 12 route-assigned crossing percentiles.
- `selected-checkpoints.json` records frozen choices, validation violations,
  model hashes, and checkpoint paths relative to its version directory.
- `checkpoints/` contains all 48 selected models across the five versions.
- `validation-baseline.csv` and the run-level `checkpoint-metrics.csv` files
  retain the validation evidence for every candidate checkpoint.
- Run directories retain training logs, completed-run metadata, randomized
  demand schedules where applicable, and V5 BC diagnostics.
- `source.zip` contains the matching archived Python implementation, tests,
  and requirements. Its implementation files retain their original bytes.
- `AUDIT_RESULTS.json` records the historical audit of the original experiment
  archive. The published records have their own checks in `verify_results.py`.

V5 additionally includes the six BC reference models in `bc-checkpoints/`,
their 18 primary evaluations, imitation-quality measurements, and paired
BC-to-selected comparisons. Crossing breakdowns and comparison CSVs are
included where recorded. Diagnostic evidence for V2 action redundancy and V4
convergence and uncertainty appears under `diagnostics/`.

## Verify the recorded results

From the repository root, run this command with Python 3.10 or newer.

```bash
python scripts/verify_results.py
```

The check uses the Python standard library. It verifies artifact hashes and
source snapshots, recomputes the final means and confidence intervals from
accepted test rows, reproduces frozen selections from validation metrics,
checks selected-model hashes and portable paths, and checks V4-V5 baseline
reuse against V3. The expected totals are 696 main test rows, 18 BC reference
rows, 792 validation rows, and 48 selected policies.

Run the simulator and training regression tests with the project dependencies
and pytest installed.

```bash
python -m pip install -r requirements.txt pytest
python -m pytest tests -q
```

`manifest.json` records each published artifact's SHA-256 and original local
source hash. Accepted test CSVs preserve every metric cell while expressing
checkpoint paths relative to the corresponding version directory. Original
candidate paths inside training records identify historical run locations.
The selected models resolve through `selected-checkpoints.json`.

## Replay a selected V3 policy

The experiments used SUMO 1.28.0. Install the recorded simulator version and
the policy dependencies in a Python environment with `netconvert` and `sumo`
available on PATH.

```bash
python -m pip install eclipse-sumo==1.28.0 libsumo==1.28.0 "numpy<2" "torch>=2.0"
python -m zipfile -e results/v3/source.zip scenario/generated/review-v3
python scenario/generated/review-v3/scripts/evaluate_marl.py --checkpoint results/v3/checkpoints/ippo-vehicle_pedestrian-actions2-seed100/episode-150.pt --duration 3600 --seeds 41 42 43 --output scenario/generated/review-v3/primary-replay.csv
```

The extracted implementation writes its generated network inside its own
directory. For a slower-walking replay, add `--pedestrian-speed 0.9` and use a
separate output CSV. Low demand uses `--vehicle-period 10 --pedestrian-interval
30`; high demand uses `--vehicle-period 4 --pedestrian-interval 12`.

Use the matching version's `source.zip`, checkpoint path, and traffic test seeds
to replay V1, V2, V4, or V5. The versioned source determines the observation
features, reward configuration, and action semantics used by its checkpoints.
Validation metrics retain every evaluated candidate. The model files supplied
here cover the frozen selections and V5 BC reference.

## Metric interpretation

All main evaluations last 3,600 seconds. Vehicle loss is observed total time
loss divided by departed vehicles. Throughput counts completed vehicles per
hour. Active unfinished counts exclude demand awaiting insertion. Pedestrian
waiting includes active trips observed up to the cutoff, so unfinished waits
are right-censored. Worst-crossing p95 takes the maximum of the 12 crossing
percentiles within each rollout and then averages those values.

The reported 95% Student-t intervals have n=3 and df=2. Learned-policy intervals
use three training-seed averages, each averaging three traffic tests. Baseline
intervals use the three traffic seeds directly. A zero-width learned interval
reflects identical training-seed averages on that shared traffic set. The two
interval types quantify different sources of variation.

The saved generated signal program serves north-south cross-street approaches
in phase 0 and east-west arterial approaches in phase 3. Their pedestrian
greens serve crossings c1/c3 and c0/c2, respectively. The generated XML retains
vehicle green in phases 1/4 while pedestrian greens end, followed by vehicle
yellow in phases 2/5. The configured 5 s or 8 s duration applies to phases 2/5.
The recorded outcomes use these generated states. The experiment reports'
historical all-red wording should be interpreted against that XML.
