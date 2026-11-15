# V4 Upstream-Lookahead & Calibrated-Switching MAPPO

Six MAPPO runs: vehicle/combined rewards x training seeds 100/200/300, 400 episodes x3600 seconds from scratch.
Training uses fixed primary demand 1.4/6/20, not demand randomization. Observations have 25 features, including four total approach vehicle counts /25 in addition to four stopped queues. Central critic input is 75.
Switch cost 3; vehicle/pedestrian scales 20.8/20; combined urgency 0.5*sum((current crossing max wait/60)^2), once per five-second step. Two actions, minimum/maximum green 10/45; network geometry and signal programs unchanged.
Eight snapshots per run (50..400) were selected only on validation seeds 21/22/23. Test seeds 41/42/43 are reused from V3 for a matched exploratory comparison, not untouched confirmatory testing.
0/6 selections meet every validation crossing limit; others use the predeclared normalized-positive-violation fallback.
72 new learned-policy evaluations plus 24 exactly reused V3 baseline rows, 24 per scenario. Baseline code/network hashes and every reused record are verified.
Mean +/- 95% Student-t CI (n=3, df=2). Learned policies average tests within training seed, then form the interval across training seeds. Baseline intervals use traffic test seeds. A zero learned interval does not imply no traffic-seed variability.
Vehicle loss is total observed loss / departed vehicles; throughput counts completions. Unfinished vehicles exclude demand awaiting insertion. Pedestrian waits include active trips and are right-censored at3600s. Crossing p95 retains the prior route-assigned whole-trip waiting definition.
V4 versus V3 changes observation inputs, switch cost, training demand regime and training length together; it cannot isolate one causal contribution. Percent changes are descriptive, not significance tests.

## primary

| Controller | Loss (s/veh) | Throughput (veh/h) | Active unfinished vehicles | Pedestrian wait (s) | Worst crossing p95 (s) |
|---|---:|---:|---:|---:|---:|
| Baseline fixed_time | 173.69 +/- 50.33 | 2379.33 +/- 493.56 | 138.33 +/- 8.72 | 189.42 +/- 104.98 | 974.80 +/- 1040.86 |
| Baseline pedestrian_actuated | 178.03 +/- 63.63 | 2325.00 +/- 818.80 | 134.00 +/- 11.38 | 191.48 +/- 121.04 | 919.67 +/- 1189.92 |
| V3 mappo-vehicle | 173.69 +/- 0.00 | 2379.33 +/- 0.00 | 138.33 +/- 0.00 | 189.42 +/- 0.00 | 974.80 +/- 0.00 |
| V3 mappo-vehicle_pedestrian | 221.89 +/- 118.55 | 2017.56 +/- 839.94 | 146.33 +/- 16.12 | 192.08 +/- 116.80 | 935.48 +/- 574.34 |
| V4 mappo-vehicle | 178.14 +/- 10.94 | 2326.78 +/- 137.03 | 139.78 +/- 3.73 | 185.47 +/- 41.61 | 954.32 +/- 169.57 |
| V4 mappo-vehicle_pedestrian | 225.66 +/- 20.81 | 1960.00 +/- 114.24 | 146.00 +/- 9.76 | 205.38 +/- 77.23 | 1070.71 +/- 546.83 |

## low

| Controller | Loss (s/veh) | Throughput (veh/h) | Active unfinished vehicles | Pedestrian wait (s) | Worst crossing p95 (s) |
|---|---:|---:|---:|---:|---:|
| Baseline fixed_time | 44.65 +/- 16.11 | 2775.00 +/- 200.91 | 66.33 +/- 86.57 | 104.50 +/- 3.76 | 218.00 +/- 85.31 |
| Baseline pedestrian_actuated | 92.86 +/- 223.53 | 2418.67 +/- 1748.66 | 72.33 +/- 82.29 | 152.58 +/- 211.55 | 598.23 +/- 1748.82 |
| V3 mappo-vehicle | 44.65 +/- 0.00 | 2775.00 +/- 0.00 | 66.33 +/- 0.00 | 104.50 +/- 0.00 | 218.00 +/- 0.00 |
| V3 mappo-vehicle_pedestrian | 82.89 +/- 86.43 | 2549.78 +/- 593.83 | 80.33 +/- 76.49 | 116.07 +/- 58.20 | 483.58 +/- 643.55 |
| V4 mappo-vehicle | 89.11 +/- 126.48 | 2408.00 +/- 1078.63 | 79.11 +/- 63.78 | 146.65 +/- 117.52 | 727.33 +/- 1399.50 |
| V4 mappo-vehicle_pedestrian | 105.70 +/- 120.18 | 2413.78 +/- 903.83 | 95.89 +/- 21.28 | 133.98 +/- 113.26 | 666.42 +/- 1090.86 |

## high

| Controller | Loss (s/veh) | Throughput (veh/h) | Active unfinished vehicles | Pedestrian wait (s) | Worst crossing p95 (s) |
|---|---:|---:|---:|---:|---:|
| Baseline fixed_time | 973.66 +/- 420.06 | 396.67 +/- 245.23 | 150.33 +/- 14.97 | 307.12 +/- 83.03 | 1226.17 +/- 557.24 |
| Baseline pedestrian_actuated | 654.94 +/- 272.67 | 613.33 +/- 350.75 | 147.67 +/- 7.59 | 247.76 +/- 20.27 | 826.03 +/- 100.82 |
| V3 mappo-vehicle | 973.66 +/- 0.00 | 396.67 +/- 0.00 | 150.33 +/- 0.00 | 307.12 +/- 0.00 | 1226.17 +/- 0.00 |
| V3 mappo-vehicle_pedestrian | 989.72 +/- 222.73 | 357.56 +/- 103.79 | 148.00 +/- 10.38 | 273.31 +/- 50.27 | 985.81 +/- 306.54 |
| V4 mappo-vehicle | 877.80 +/- 164.74 | 456.11 +/- 156.86 | 153.89 +/- 21.82 | 292.73 +/- 47.16 | 1100.78 +/- 259.81 |
| V4 mappo-vehicle_pedestrian | 858.18 +/- 427.28 | 483.00 +/- 344.83 | 156.89 +/- 11.18 | 274.75 +/- 40.21 | 1064.41 +/- 291.19 |

## slow

| Controller | Loss (s/veh) | Throughput (veh/h) | Active unfinished vehicles | Pedestrian wait (s) | Worst crossing p95 (s) |
|---|---:|---:|---:|---:|---:|
| Baseline fixed_time | 421.08 +/- 198.16 | 1060.00 +/- 419.02 | 150.00 +/- 9.94 | 254.67 +/- 45.11 | 859.17 +/- 182.92 |
| Baseline pedestrian_actuated | 405.50 +/- 87.21 | 1120.33 +/- 147.64 | 150.33 +/- 7.99 | 262.15 +/- 38.77 | 1057.67 +/- 300.36 |
| V3 mappo-vehicle | 421.08 +/- 0.00 | 1060.00 +/- 0.00 | 150.00 +/- 0.00 | 254.67 +/- 0.00 | 859.17 +/- 0.00 |
| V3 mappo-vehicle_pedestrian | 563.24 +/- 315.52 | 761.22 +/- 438.96 | 149.89 +/- 25.15 | 287.87 +/- 67.01 | 1099.67 +/- 192.71 |
| V4 mappo-vehicle | 466.80 +/- 366.09 | 985.11 +/- 820.83 | 150.11 +/- 2.08 | 267.73 +/- 79.45 | 1002.92 +/- 326.15 |
| V4 mappo-vehicle_pedestrian | 533.03 +/- 304.53 | 827.22 +/- 589.49 | 144.78 +/- 5.06 | 288.56 +/- 24.85 | 1101.67 +/- 177.58 |

## Frozen selections

| Controller | Episode | Validation feasible | Largest crossing excess (s) | Normalized positive violation |
|---|---:|---|---:|---:|
| mappo-vehicle-actions2-seed100 | 200 | False | 0.067 | 0.000374 |
| mappo-vehicle_pedestrian-actions2-seed100 | 150 | False | 14.567 | 0.033675 |
| mappo-vehicle-actions2-seed200 | 50 | False | 15.867 | 0.154868 |
| mappo-vehicle_pedestrian-actions2-seed200 | 400 | False | 3.233 | 0.003805 |
| mappo-vehicle-actions2-seed300 | 100 | False | 14.000 | 0.181572 |
| mappo-vehicle_pedestrian-actions2-seed300 | 400 | False | 1.900 | 0.020191 |
