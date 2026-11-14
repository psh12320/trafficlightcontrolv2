# SUMO traffic-light control

This project creates a small SUMO scenario with **3 signalized junctions** and
**12 pedestrian crossings** (four at each junction), then runs an adaptive
TraCI controller.

## Requirements

- SUMO 1.18 or newer on `PATH`, or `SUMO_HOME` set to the SUMO installation
- Python 3.10 or newer
- The `traci` package, normally available at `%SUMO_HOME%\tools`

## Run

```powershell
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
authoritative Phase 0 benchmark uses 10 seeds and 600 seconds. Its corrected
pedestrian-actuated controller averaged 55.05 seconds vehicle waiting and
102.43 seconds pedestrian waiting (95% confidence intervals: 49.25-60.86 and
99.32-105.54 seconds). Waiting is network-wide across all three junctions and
12 crossings; per-junction and per-crossing totals are also recorded.

## IPPO and MAPPO

`train_marl.py` provides independent PPO (`--algorithm ippo`) and centralized
critic MAPPO (`--algorithm mappo`) training. The reward penalizes vehicle
waiting plus twice pedestrian waiting, with a small phase-switch penalty:

```powershell
python scripts\train_marl.py --algorithm mappo --episodes 100 --horizon 120
```

Training is not guaranteed to improve monotonically. Use
`--snapshot-dir` to save every episode and evaluate a learning curve:

```powershell
python scripts\train_marl.py --algorithm mappo --episodes 50 `
  --snapshot-dir scenario\generated\mappo-snapshots
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

The original experiments used two actions: hold the current signal behavior,
or request a legal phase switch. The richer action set now also supports
extending the current green by 5 or 10 seconds. Extensions are legal only
during green and remain bounded by the 45-second maximum green. SUMO still
handles yellow and all-red clearance phases.

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

## Current comparison

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
sets the all-red clearance phases from the crossing length and walking speed,
with a three-second minimum.

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

## Network layout

The three junctions form an east-west corridor. Each junction has north,
south, east, and west approaches. The outer approaches terminate at boundary
nodes, and every junction has one crossing over each incoming approach.
