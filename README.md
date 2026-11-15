# SUMO traffic-light control

This project creates a small SUMO scenario with **3 signalized junctions** and
**12 pedestrian crossings** (four at each junction), then runs an adaptive
TraCI controller.

## Review and reproduce

The [published experiment records](results/README.md) include V1-V5 reports,
accepted evaluation CSVs, training logs, validation metrics, 48 frozen selected
checkpoints, and matching archived source. The [final figures and captions](report_figures/README.md)
are included in the repository.

Verify all published artifact hashes, recompute the result summaries, and check
validation-based selections with Python 3.10 or newer.

```bash
python scripts/verify_results.py
```

The check covers 696 accepted main test rows, 18 V5 BC reference rows, 792
validation evaluations, and 48 selected policies. It uses the Python standard
library. The [replay instructions](results/README.md#replay-a-selected-v3-policy)
show how to evaluate a saved policy using its matching implementation.

| Version | Episodes per run | Actor inputs | Actions | Traffic test seeds | Validation-feasible selections |
|---|---|---:|---:|---|---:|
| V1 | 60 | 13 | 4 | 11-13 | 1/12 |
| V2 | 300 | 13 | 4 | 11-13 | 2/12 |
| V3 | 300 | 21 | 2 | 41-43 | 2/12 |
| V4 | 400 | 25 | 2 | 41-43 | 0/6 |
| V5 | 20 BC + 5 critic warm-up + 250 PPO | 75 | 2 | 41-43 | 0/6 |

Training seeds are 100/200/300 and validation seeds are 21/22/23. V2 reuses V1's
traffic test seeds. V4-V5 reuse V3's traffic test seeds for matched exploratory
comparisons. Several design choices change together between versions.

## Final benchmark results

All values are mean ± 95% Student-t confidence intervals, n=3 and df=2.
Learned-policy intervals use three training-seed averages after averaging the
three traffic tests within each training seed. Baseline intervals use the
three traffic seeds directly. MAPPO-Vehicle's zero-width V3 intervals reflect
identical training-seed averages on the shared test set. Traffic-seed outcomes
still vary.

Every evaluation lasts 3,600 seconds. Vehicle loss is total observed time loss
per departed vehicle. Unfinished counts cover active trips. Pedestrian waits
include active trips observed to the cutoff. Worst crossing p95 is the maximum
of the 12 route-assigned crossing percentiles per rollout, then averaged.

| Regime | Walking speed (m/s) | Vehicle period (s) | Pedestrian batch interval (s) |
|---|---:|---:|---:|
| Primary | 1.4 | 6 | 20 |
| Low | 1.4 | 10 | 30 |
| High | 1.4 | 4 | 12 |
| Slower-Walking | 0.9 | 6 | 20 |

The generated signal program serves north-south cross-street approaches in
phase 0 and east-west arterial approaches in phase 3, with pedestrian greens
for c1/c3 and c0/c2, respectively. Its phase 2/5 duration is 5 s for standard
walking and 8 s for slower walking. The [signal-state note](results/README.md#metric-interpretation)
describes the actual generated states used by the experiments.

### V3 Primary benchmark

| Controller | Vehicle loss (s/veh) | Throughput (veh/h) | Unfinished veh. | Mean ped. wait (s) | Worst crossing p95 (s) | Unfinished ped. |
|---|---:|---:|---:|---:|---:|---:|
| Fixed-Time | 173.69 ± 50.33 | 2379.33 ± 493.56 | 138.33 ± 8.72 | 189.42 ± 104.98 | 974.80 ± 1040.86 | 136.67 ± 66.16 |
| Ped.-Actuated | 178.03 ± 63.63 | 2325.00 ± 818.80 | 134.00 ± 11.38 | 191.48 ± 121.04 | 919.67 ± 1189.92 | 147.00 ± 78.87 |
| IPPO-Vehicle | 191.01 ± 41.01 | 2230.67 ± 325.49 | 137.33 ± 13.83 | 200.93 ± 43.99 | 1024.08 ± 460.71 | 143.00 ± 48.73 |
| IPPO-Combined | 186.11 ± 10.20 | 2269.56 ± 184.01 | 142.11 ± 5.51 | 182.24 ± 17.69 | 966.91 ± 178.76 | 138.22 ± 35.72 |
| MAPPO-Vehicle | 173.69 ± 0.00 | 2379.33 ± 0.00 | 138.33 ± 0.00 | 189.42 ± 0.00 | 974.80 ± 0.00 | 136.67 ± 0.00 |
| MAPPO-Combined | 221.89 ± 118.55 | 2017.56 ± 839.94 | 146.33 ± 16.12 | 192.08 ± 116.80 | 935.48 ± 574.34 | 139.78 ± 103.07 |

### Full results by version

Each full report contains Primary, Low, High, and Slower-Walking tables. The
summary CSVs contain all reported metrics and confidence intervals. Selected
models and their matching source are available alongside each report.

| Version | Full report | Summary data | Selected models | Matching source |
|---|---|---|---|---|
| V1 | [Results](results/v1/FINAL_RESULTS.md) | [CSV](results/v1/final-summary.csv) | [Checkpoints](results/v1/checkpoints) | [Source ZIP](results/v1/source.zip) |
| V2 | [Results](results/v2/FINAL_RESULTS.md) | [CSV](results/v2/final-summary.csv) | [Checkpoints](results/v2/checkpoints) | [Source ZIP](results/v2/source.zip) |
| V3 | [Results](results/v3/FINAL_RESULTS.md) | [CSV](results/v3/final-summary.csv) | [Checkpoints](results/v3/checkpoints) | [Source ZIP](results/v3/source.zip) |
| V4 | [Results](results/v4/FINAL_RESULTS.md) | [CSV](results/v4/final-summary.csv) | [Checkpoints](results/v4/checkpoints) | [Source ZIP](results/v4/source.zip) |
| V5 | [Results](results/v5/FINAL_RESULTS.md) | [CSV](results/v5/final-summary.csv) | [Checkpoints](results/v5/checkpoints) | [Source ZIP](results/v5/source.zip) |

The [results guide](results/README.md) explains the raw evaluation files,
validation records, model hashes, and replay commands. V5 includes the matched
BC reference and paired fine-tuning comparisons. The saved results show
regime-dependent trade-offs across vehicle efficiency and pedestrian waiting.

<details>
<summary>Completed V1-V5 experiment notes</summary>

## Completed V5 BC warm-started experiment

V5 completed six MAPPO vehicle/combined runs with training seeds 100/200/300:
20 expert BC episodes, five critic warm-up episodes and 250 PPO episodes of
3,600 seconds on fixed primary demand. `--frame-stack 3` gives 75 actor and 225
critic inputs. BC uses 15 masked-cross-entropy epochs, Adam 1e-3 and balanced
legal-action minibatches. PPO uses `--learning-rate 1e-4`. Defaults remain frame
stack one and BC zero. Switch cost is 3; V4 reward scales remain unchanged.

All 36 snapshots, 108 validations and six frozen selections are saved. There are
96 main test rows, including 24 exactly reused baselines, plus an 18-row matched
primary BC reference (12 new tests, six reused selected-BC rows). Test seeds
41-43 are reused exploratory cases. Selection uses only validation seeds 21-23.

All three pods and the temporary volume are deleted. Conservative
V5 cost is **$1.22**, including diagnostics and transfers, below the $5 cap.
The project test suite contains 31 passing tests. The audit verified 282 entries, four archive hashes and
**4,841 prior V1-V4 files unchanged**.

V5 does not establish consistent superiority over actuated control or a primary
PPO benefit over BC. Two selections remain BC-only; none meets every crossing
validation threshold. Initial uniform BC collapsed to hold and was excluded.
Balanced sampling learns rare demonstration switches but rollout imitation
remains imperfect. See the [full comparison tables](results/v5/FINAL_RESULTS.md) and
[takeaways](results/v5/KEY_TAKEAWAYS.md).
Do not restart training.

## Completed V4 experiment

V4 trained only MAPPO vehicle/combined objectives with seeds 100/200/300,
400 episodes x3600s, fixed primary training demand (no randomization), and
snapshots every 50 episodes. It adds four approaching-vehicle density features
to the stopped queues, giving 25 local inputs and 75 centralized critic inputs.
Switch cost is 3; V3 pedestrian urgency and scales remain unchanged. Network,
signal phases and two-action semantics are unchanged.

Validation uses 21/22/23. The requested test seeds 41/42/43 are reused from V3,
so this is a matched exploratory comparison. V3's baseline records are copied
exactly rather than rerun. Prior artifacts are protected by a 4,524-file hash
manifest. Outputs go to `results/v4/`.

All six runs, 48 snapshots, 144 validation evaluations, six frozen selections,
and 72 new policy tests are complete (96 final rows including reused baselines).
All three pods were terminated and the temporary checkpoint volume is deleted.
Conservative V4 runtime/storage cost is **$0.95**, below the $5 cap; cumulative project bound is $4.35. Posted
billing lags. The audit verified 214 transferred entries and all 4,524 protected
prior files unchanged. All 25 local tests and each cloud smoke test passed.

V4 does not establish a consistent advantage over actuated control or V3.
Primary vehicle-only reduces mean pedestrian wait 3.1% versus actuated with
similar vehicle loss, but crossing p95 is 3.8% higher. Combined worsens primary
vehicle loss 26.7%, pedestrian wait 7.3%, and crossing p95 16.4%. No selected
checkpoint meets every crossing's validation limit; all use the declared
violation fallback. See the [full comparison tables](results/v4/FINAL_RESULTS.md) and
[takeaways](results/v4/KEY_TAKEAWAYS.md).
Do not restart training.
Older 21-input V3 checkpoints require their preserved V3 source rather than
current 25-input code. V4 changes several settings together, so comparisons
cannot isolate the density feature or the switch penalty's causal effect.

## Completed V3 experiment

V3 trained IPPO/MAPPO with vehicle and combined objectives, two actions, and
21 observations. Pedestrian features distinguish approach waiting counts,
oldest current waits and crossing occupancy. Walking-area queues are attributed
using SUMO's actual next crossing, with a start-edge fallback. The combined cost
uses vehicle waiting / 20.8, pedestrian waiting / 20, crossing urgency
`0.5 * sum((max_wait / 60) ** 2)`, and a switch cost of 8.

Three 16-vCPU pods trained seeds 100/200/300 for 300 episodes of 3,600 seconds,
with randomized demand and validation every 50 episodes on seeds 21/22/23.
Selections froze before testing all four scenarios on fresh seeds 41/42/43.
All 12 runs, 216 validation evaluations, and 168 test evaluations are complete.
All pods and the temporary checkpoint volume are deleted. The conservative V3
runtime/storage bound is $1.69, below the $5 cap. The final local suite passes
23 tests; the audit verified 348 manifest entries and 368 unchanged V2 files.

The [full report](results/v3/FINAL_RESULTS.md),
[summary](results/v3/final-summary.csv), and
[crossing breakdown](results/v3/per-crossing-p95.csv)
are saved under `results/v3/`.
Only 2/12 selected checkpoints were validation-feasible. V3 does not establish
a consistent primary advantage over actuated control. MAPPO vehicle-only
matches fixed-time metrics; its zero-width training-seed intervals do not mean
traffic test seeds have no variability.
V2 checkpoints use 13 inputs and four actions. Its matching implementation is
published in `results/v2/source.zip`.
V2 versus V3 changes several settings together and does not isolate individual
causal effects.

## Completed V2 300-episode experiment

The V2 experiment is complete: 12 runs trained from scratch for 300 episodes of
3,600 seconds, with normalized proposal rewards, a 1.5 switch penalty, and
randomized training demand. Three parallel 16-vCPU pods were used and all are
terminated. The conservative V2 runtime-cost bound is $1.39, below the $15 cap.
All 18 local tests pass; the independent audit verified 216 validation checkpoint
evaluations, 168 held-out evaluations, 12 selected checkpoints, reproducible
demand schedules, and the unchanged 490-file pilot archive.

See the [complete V2 report](results/v2/FINAL_RESULTS.md),
[summary CSV](results/v2/final-summary.csv), and
[pilot comparison](results/v2/pilot-comparison.csv).
The accepted pilot results and selected policies are published under `results/v1/`.

V2 did not establish a consistent primary-benchmark advantage over actuated
control. Low-demand IPPO vehicle-only control had favorable point estimates;
MAPPO vehicle-only reduced vehicle loss under high demand and slower walking,
with worse pedestrian tail waits than actuated control. Only 2 of 12 selected
runs met every crossing's validation limit, versus 1 of 12 in the pilot.
Three-seed intervals are wide and the test seeds were reused from the pilot.

Reproduce one training seed with:

```bash
python -m zipfile -e results/v2/source.zip scenario/generated/review-v2
python scenario/generated/review-v2/scripts/run_marl_sweep.py --algorithms ippo mappo \
  --reward-modes vehicle vehicle_pedestrian --action-sizes 4 \
  --episodes 300 --horizon 3600 --seed 100 --randomize-demand \
  --snapshot-interval 50 --evaluation-duration 3600 \
  --evaluation-seeds 21 22 23 --output-root scenario/generated/v2-300ep-seed100
```

Repeat in separate processes/pods for seeds 200 and 300. Workers cache demand
networks in private temporary directories, and checkpoint validation uses fixed
primary demand. `v2_finalize.py`, `select_v2_checkpoints.py`,
`audit_v2_results.py`, and `report_v2_results.py` implement the remaining stages.

## Completed 60-episode proposal pilot

The four proposal controllers (IPPO/MAPPO × vehicle/vehicle-pedestrian rewards,
four actions) were each trained with seeds 100, 200, and 300 for 60 episodes of
3,600 seconds. Episodes 20, 40, and 60 were evaluated on validation seeds 21–23.
The 12 selected checkpoints and two baselines were tested on seeds 11–13 in
primary, low-demand, high-demand, and slower-walking scenarios: 168 accepted
test evaluations. Test seeds were excluded from this run's selection, although
they had been used in earlier exploratory experiments.

The downloaded [full results](results/v1/FINAL_RESULTS.md) and
[summary CSV](results/v1/final-summary.csv) contain means and 95%
Student-t intervals. Learned-controller intervals use three independent training
seeds after averaging their three test evaluations. On the primary benchmark,
actuated control averaged 168.65 seconds time loss per departed vehicle versus
181.09–208.39 for the learned families. This pilot does not establish a primary
MARL improvement. Only one of 12 selections met every crossing's validation
limit; the others used the lowest normalized violation fallback.

The final code also corrects mixed baseline/MARL CSV output, active unfinished
trip counts, zero-wait pedestrians in p95 populations, and pedestrian arrivals
between policy decisions. The high-demand evaluation was rerun after the arrival
fix, with 3,588 pedestrians injected for every controller. The slow scenario
uses 0.9 m/s pedestrians and eight-second clearance. Final local verification:
15 tests pass. Generated simulation outputs live under `scenario/generated/`.
The accepted final records are published under `results/`.

For a fresh run, with matching SUMO/libsumo versions installed:

```bash
python scripts/run_proposal_pilot.py --output-root scenario/generated/new-pilot --workers 12
python scripts/summarize_proposal_pilot.py --root scenario/generated/new-pilot
```

The caller must supervise cloud costs, download artifacts, and terminate compute;
the Python experiment runner does not provision or terminate cloud resources.

</details>

## Requirements

- SUMO 1.28.0 for recorded experiment replays, with its binaries on `PATH`
- Python 3.10 or newer
- The `traci` package, normally available at `%SUMO_HOME%\tools`
- Python packages listed in [requirements.txt](requirements.txt)

## Run

```powershell
python -m pip install -r requirements.txt
python scripts\build_network.py
python scripts\run_simulation.py --nogui
```

Use `python scripts\run_simulation.py` to launch `sumo-gui`. The generated
files are written to `scenario\generated`.

The controller checks queue lengths every 10 seconds. It keeps a green phase
alive when the queue is high and advances to the next green phase when traffic
has cleared, while respecting minimum and maximum green times.

## Baseline experiment

The pedestrian-actuated baseline injects pedestrians over all 12 crossing
edges and records vehicle and pedestrian waiting seconds independently:

```powershell
python scripts\run_experiment.py --duration 300 --seeds 1 2 3
```

Results are written to `scenario\generated\baseline-results.csv`. The
initial baseline benchmark uses 10 seeds and 600 seconds. Its corrected
pedestrian-actuated controller averaged 55.05 seconds vehicle waiting and
102.43 seconds pedestrian waiting (95% confidence intervals: 49.25-60.86 and
99.32-105.54 seconds). Waiting is network-wide across all three junctions and
12 crossings; per-junction and per-crossing totals are also recorded.

## IPPO and MAPPO

`train_marl.py` provides independent PPO (`--algorithm ippo`) and centralized
critic MAPPO (`--algorithm mappo`) training. Reward objectives, vehicle/pedestrian weights, and switching costs are
configurable. The versioned source snapshots record the completed experiments
for the saved checkpoints.

```powershell
python scripts\train_marl.py --algorithm mappo --episodes 100 --horizon 120
```

Training is not guaranteed to improve monotonically. Use
`--snapshot-dir` and `--snapshot-interval` to save checkpoints and evaluate a
learning curve.

```powershell
python scripts\train_marl.py --algorithm mappo --episodes 50 `
  --snapshot-dir scenario\generated\mappo-snapshots --snapshot-interval 10
python scripts\evaluate_marl.py `
  --checkpoint scenario\generated\mappo-snapshots\episode-20.pt `
  --duration 300 --seeds 1 2 3
```

The main tunable objective parameters are `--pedestrian-weight` and
`--switch-penalty`; the experiment should keep the evaluation seeds and
duration fixed while sweeping them.

Each training episode is 120 simulated seconds by default. Policies act every
5 simulated seconds, giving approximately 24 decisions per episode. On the
development machine, a 50-episode run typically takes about 3-5 minutes when
run alone; running several runs in parallel has similar wall-clock time but
uses more CPU.

The default action space has two actions, hold the current signal behavior
and request a legal phase switch. The optional four-action set also supports
extending the current green by 5 or 10 seconds. Extensions are legal only
during green and remain bounded by the 45-second maximum green. SUMO advances
the generated transition phases.

This objective follows recent pedestrian-aware signal-control work that
evaluates waiting time, queue length, throughput, and pedestrian service
together. Relevant research includes [curiosity-enhanced pedestrian-aware
MAPPO](https://doi.org/10.1049/itr2.70247), the [policy-gradient
pedestrian-vehicle signal model](https://ieeexplore.ieee.org/document/11064433),
and [risk-sensitive pedestrian-aware traffic signal
control](https://link.springer.com/content/pdf/10.1007/978-3-032-44182-9_22.pdf).
These papers support the experiment design, but their reported improvements
cannot be assumed to transfer to this small network; every learned policy
must be evaluated on the same seeds and demand as the baseline.

<details>
<summary>Preliminary experiments and development history</summary>

## Preliminary short-run comparisons

These experiments preceded the V1-V5 final benchmarks and are retained as
development history.

The canonical PPO implementation was tested with identical 120-second
training horizons for 10 episodes, then evaluated on held-out seeds 11-13 for
300 seconds. The matched actuated baseline averaged 31.32 seconds vehicle
waiting and 83.60 seconds pedestrian waiting. Standard MAPPO averaged 35.23
and 76.11 seconds; standard IPPO averaged 45.83 and 77.45 seconds. Therefore
MAPPO reduced pedestrian waiting by 8.97% but increased vehicle waiting by
12.49%; IPPO reduced pedestrian waiting by 7.35% but increased vehicle waiting
by 46.30%. Neither met the no-regression criterion.

Reward ablations were also evaluated on the same held-out seeds. Normalized
MAPPO averaged 35.23/76.11 seconds (vehicle/pedestrian), constrained MAPPO
averaged 38.73/76.83, and queue-aware MAPPO averaged 35.00/76.43. These
10-episode ablations did not yet produce a policy that beats actuated control
on both metrics. Longer training, observation ablations, and multi-seed RL
training are required before claiming a final winner.

### Vehicle-priority weight sweep

The normalized reward now exposes independent `--vehicle-weight` and
`--pedestrian-weight` coefficients. A six-point MAPPO sweep used vehicle
weights 1.0, 1.5, and 2.0 crossed with pedestrian weights 0.5 and 0.75.
Every candidate trained for 10 episodes and was evaluated on held-out seeds
11-13. All six deterministic policies produced the same aggregate result:
35.68 seconds vehicle waiting and 76.27 seconds pedestrian waiting. Thus this
short sweep found no weight achieving the target of vehicle waiting below
31 seconds while keeping pedestrian waiting at or below 80 seconds. The
identical evaluation behavior indicates that longer training and multiple
training seeds are needed to determine whether the objective weights can
change the learned policy.

The next controlled run used the default normalized weights for 50 episodes
with independent training seeds 100, 200, and 300. Final checkpoints were
evaluated on held-out seeds 11-13. The three training-seed aggregates were:

| Training seed | Vehicle waiting | Pedestrian waiting |
|---|---:|---:|
| 100 | 36.74 s | 77.49 s |
| 200 | 44.39 s | 69.86 s |
| 300 | 38.42 s | 70.99 s |

Across all nine held-out evaluations, the mean was 39.85 seconds vehicle
waiting and 72.78 seconds pedestrian waiting. One individual evaluation
achieved both targets, at 28.13/64.04 seconds, but no training seed satisfied
both targets consistently across all held-out seeds. More episodes alone are
therefore not yet sufficient evidence of a reliable improvement below the
31-second vehicle target.

### Reproducing the best observed checkpoint

The final checkpoint from the 50-episode run with training seed 300 was
re-evaluated independently on held-out seed 11 and reproduced exactly:
28.125 seconds vehicle waiting and 64.036 seconds pedestrian waiting. This is
better than the actuated seed-11 result of 32.85/84.12 seconds, with 352
vehicle arrivals versus 351 for actuated control.

The improvement depends strongly on checkpoint timing. On the same held-out
seed, seed-300 snapshots from episodes 10, 20, 30, and 40 produced vehicle
waiting of 42.06, 40.89, 43.47, and 39.74 seconds respectively; episode 50
produced 28.13 seconds. Thus the result is reproducible for the exact
checkpoint, but not yet robust across evaluation seeds: the same checkpoint
produced 41.84/76.50 and 45.30/72.44 seconds on held-out seeds 12 and 13.
Future selection must use validation seeds separate from final test seeds, and
multiple training seeds should be retained rather than assuming the last
checkpoint is best.

## Proposal-aligned long-run sweep

The proposal's four learned controllers are the two algorithms crossed with two
objectives:

- `vehicle`: vehicle waiting only
- `vehicle_pedestrian`: vehicle and pedestrian waiting

Each objective can be evaluated with either the original two-action space
(`hold`, `switch`) or the four-action space (`hold`, `switch`, `extend 5 s`,
`extend 10 s`). The sweep therefore contains eight configurations
(IPPO/MAPPO x two objectives x two action spaces). It saves checkpoints and
evaluates each one at episodes 500, 1000, ..., 3000:

```powershell
python scripts\run_marl_sweep.py --episodes 3000 --horizon 3600 `
  --evaluation-duration 3600 --evaluation-seeds 11 12 13 `
  --output-root scenario\generated\marl-sweep
```

Each configuration writes `checkpoint-metrics.csv`. The reported metrics
include completed and unfinished trips, vehicle time loss, waiting totals and
means, pedestrian 95th-percentile waiting, and waiting by crossing. The same
`--pedestrian-speed` value must be used for training, learned-policy
evaluation, and fixed-time/pedestrian-actuated baselines. Network generation
sets the phase 2/5 durations from the crossing length and walking speed, with
a three-second minimum.

### 100-episode reproduction attempt

To test whether additional training would reproduce the result on new
training seeds, normalized MAPPO was trained for 100 episodes with seeds 400,
500, and 600. Each final checkpoint was evaluated on held-out seeds 11-13:

| Training seed | Vehicle waiting | Pedestrian waiting |
|---|---:|---:|
| 400 | 48.52 s | 80.43 s |
| 500 | 42.57 s | 71.18 s |
| 600 | 44.39 s | 69.86 s |

Across all nine evaluations, the mean was 45.16 seconds vehicle waiting and
73.82 seconds pedestrian waiting. No evaluation met both the vehicle target
below 31 seconds and pedestrian target at or below 80 seconds. Thus more
episodes alone did not reproduce the seed-300 episode-50 result; the next
step should be validation-based checkpoint selection and investigation of
training-seed sensitivity rather than increasing episode count blindly.

### Longer IPPO comparison

IPPO was not conclusively ruled out by the earlier 10-episode result, so it
was retrained for 50 episodes with seeds 400, 500, and 600 using the same
normalized reward, horizons, and held-out evaluation seeds as the MAPPO
comparison. The final-checkpoint aggregates were:

| Training seed | Vehicle waiting | Pedestrian waiting |
|---|---:|---:|
| 400 | 44.38 s | 69.86 s |
| 500 | 44.37 s | 80.44 s |
| 600 | 44.38 s | 69.86 s |

Across all nine IPPO evaluations, the mean was 44.38 seconds vehicle waiting
and 73.38 seconds pedestrian waiting. No evaluation met both the vehicle
target below 31 seconds and pedestrian target at or below 80 seconds. Under
this current environment and training protocol, IPPO is substantially less
competitive than MAPPO for vehicle waiting, but this is not a proof that no
IPPO hyperparameter or architecture can succeed.

### Constrained reward and optimization-stability experiments

Three 50-episode MAPPO runs using the constrained reward were evaluated on
held-out seeds 11-13. Their aggregate results were:

| Training seed | Vehicle waiting | Pedestrian waiting | Seeds meeting both targets |
|---|---:|---:|---:|
| 400 | 44.38 s | 69.86 s | 0/3 |
| 500 | 39.38 s | 76.16 s | 0/3 |
| 600 | 44.92 s | 75.33 s | 0/3 |

Across all nine constrained-reward evaluations, the mean was 42.90 seconds
vehicle waiting and 73.78 seconds pedestrian waiting. The constrained reward
improved neither vehicle waiting nor dual-target reliability relative to the
normalized objective.

To test whether unstable optimization caused the occasional successful
checkpoint, normalized MAPPO was also trained for 50 episodes with learning
rate `1e-4` and 10 PPO epochs, using training seeds 700, 800, and 900. The
three-checkpoint aggregate was 35.64 seconds vehicle waiting and 82.39 seconds
pedestrian waiting; no evaluation met both targets. Lowering the learning rate
therefore improved vehicle waiting relative to the constrained runs but did not
reproduce the low pedestrian waiting result.

### Rich action set and validation-based selection

Three 50-episode four-action MAPPO runs were trained with seeds 1000, 1100,
and 1200. Snapshots at episodes 10, 20, 30, 40, and 50 were evaluated on
validation seeds 21-23. The selected snapshots were episode 40, episode 10,
and episode 50 respectively. Selection minimized normalized threshold
violation, using vehicle target 31 seconds and pedestrian target 80 seconds.

The selected snapshots were then evaluated on untouched test seeds 31-33:

| Training seed | Selected episode | Vehicle waiting | Pedestrian waiting | Seeds meeting both targets |
|---|---:|---:|---:|---:|
| 1000 | 40 | 46.09 s | 86.07 s | 0/3 |
| 1100 | 10 | 33.79 s | 76.78 s | 1/3 |
| 1200 | 50 | 35.70 s | 79.73 s | 1/3 |

Across all nine test evaluations, the mean was 38.53 seconds vehicle waiting
and 80.86 seconds pedestrian waiting. The richer action set produced two
dual-target test results, but did not yet establish robust improvement over
the actuated baseline.

### Signal-timing sensitivity experiment

The original 37-second green duration is generated by SUMO's default
`netconvert` signal program. It is now an explicit scenario parameter, while
37 seconds remains the primary benchmark. Fixed-time and pedestrian-actuated
baselines were evaluated at 20, 25, 30, and 37 seconds on the same held-out
seeds 11-13:

| Green duration | Fixed-time vehicle/pedestrian | Actuated vehicle/pedestrian |
|---:|---:|---:|
| 20 s | 29.06 / 99.75 s | 25.40 / 96.75 s |
| 25 s | 27.31 / 104.22 s | 28.62 / 105.74 s |
| 30 s | 25.21 / 113.27 s | 27.33 / 111.87 s |
| 37 s | 27.32 / 120.23 s | 28.35 / 116.55 s |

Four-action MAPPO was then trained for 50 episodes with three training seeds
at each duration and evaluated on seeds 11-13. The corrected aggregates were:

| Green duration | Vehicle waiting | Pedestrian waiting | Dual-target results |
|---:|---:|---:|---:|
| 20 s | 31.55 s | 103.74 s | 0/9 |
| 25 s | 30.77 s | 109.55 s | 0/9 |
| 30 s | 25.46 s | 114.46 s | 0/9 |
| 37 s | 28.20 s | 122.17 s | 0/9 |

The timing sweep confirms a real tradeoff: shorter greens can bring vehicle
waiting below 31 seconds, but pedestrian waiting worsens substantially. Timing
alone does not solve the joint target. During validation of this experiment we
also corrected the RL environment to reset its internal phase-age state when
SUMO naturally changes phase; earlier preliminary timing-policy results are
not used here.

</details>

## Network layout

The three junctions form an east-west corridor. Each junction has north,
south, east, and west approaches. The outer approaches terminate at boundary
nodes, and every junction has one crossing over each incoming approach.
