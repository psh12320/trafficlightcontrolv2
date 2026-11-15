# V5 BC warm-started MAPPO

Six accepted MAPPO runs: vehicle/combined x training seeds 100/200/300. Each run uses 20 actuated-expert episodes, 15 masked-cross-entropy epochs with Adam 1e-3, five critic-only rollout episodes, then 250 PPO episodes with learning rate 1e-4. All episodes last 3600s under fixed primary demand.

Three consecutive 25-feature frames give 75 actor inputs and 225 centralized critic inputs. Reset repeats the initial frame three times; temporal order is oldest to newest. Switch cost 3, vehicle/pedestrian scales 20.8/20, combined urgency 0.5*sum((current max wait/60)^2), two actions and minimum/maximum green 10/45.

The first uniform-minibatch BC attempt collapsed to hold because switches were rare. It is retained only under diagnostics/unbalanced-bc and excluded from all candidates and accepted results. Corrected BC balances legal action classes in each minibatch while retaining the requested loss, learning rate and 15 epochs. The critic initializes its value bias to empirical mean returns and fits discounted returns with MSE. Actor weights remain unchanged during critic fitting.

The expert preserves the recorded actuated request rule, including holding when both phases have requests, at legal five-second MAPPO decisions. The recorded actuated baseline checks every second. Matching the baseline is measured, not guaranteed. Class-specific imitation statistics appear below because high hold accuracy can conceal poor switch recall.

Selections froze using validation seeds 21/22/23 and candidates BC episode 0 plus RL episodes 50/100/150/200/250. 0/6 satisfy every crossing limit; infeasible runs use the declared normalized-positive-violation fallback. 2/6 selected the BC checkpoint. Feasible candidates minimize mean total vehicle time loss; the fallback minimizes normalized violation, then total vehicle time loss.

72 new selected-policy evaluations plus 24 exactly reused V3/V4 baseline measurements produce 96 accepted rows, 24 per scenario. A separate matched primary BC reference has 18 rows, reusing selected BC measurements where applicable and evaluating only the remaining BC policies after selection froze. Test seeds 41/42/43 were reused from prior experiments for a matched exploratory comparison, not untouched confirmatory testing.

Mean +/- 95% Student-t CI, n=3 and df=2. Learned-policy test seeds are averaged within each training seed first; baselines use traffic seeds directly. Zero learned intervals do not mean zero traffic variability. Comparisons are descriptive and unadjusted. Multiple settings differ between versions, so no individual causal effect is isolated.

Vehicle loss divides total observed time loss by departed vehicles. Throughput counts completions. Unfinished vehicles exclude demand waiting for insertion. Pedestrian waits include active trips and are right-censored at 3600s; crossing percentiles retain the prior route-assigned whole-trip definition. Worst-crossing p95 takes the maximum of 12 crossing percentiles within each evaluation, then averages. The per-crossing table averages each crossing separately.

## primary

| Controller | Loss s/veh | Throughput veh/h | Active unfinished vehicles | Pedestrian wait s | Worst crossing p95 s |
|---|---:|---:|---:|---:|---:|
| Baseline fixed_time | 173.69 +/- 50.33 | 2379.33 +/- 493.56 | 138.33 +/- 8.72 | 189.42 +/- 104.98 | 974.80 +/- 1040.86 |
| Baseline pedestrian_actuated | 178.03 +/- 63.63 | 2325.00 +/- 818.80 | 134.00 +/- 11.38 | 191.48 +/- 121.04 | 919.67 +/- 1189.92 |
| V3 mappo-vehicle | 173.69 +/- 0.00 | 2379.33 +/- 0.00 | 138.33 +/- 0.00 | 189.42 +/- 0.00 | 974.80 +/- 0.00 |
| V3 mappo-vehicle_pedestrian | 221.89 +/- 118.55 | 2017.56 +/- 839.94 | 146.33 +/- 16.12 | 192.08 +/- 116.80 | 935.48 +/- 574.34 |
| V4 mappo-vehicle | 178.14 +/- 10.94 | 2326.78 +/- 137.03 | 139.78 +/- 3.73 | 185.47 +/- 41.61 | 954.32 +/- 169.57 |
| V4 mappo-vehicle_pedestrian | 225.66 +/- 20.81 | 1960.00 +/- 114.24 | 146.00 +/- 9.76 | 205.38 +/- 77.23 | 1070.71 +/- 546.83 |
| V5 mappo-vehicle | 181.63 +/- 21.20 | 2289.44 +/- 192.35 | 137.67 +/- 10.95 | 207.68 +/- 50.83 | 1101.88 +/- 624.02 |
| V5 mappo-vehicle_pedestrian | 182.59 +/- 17.36 | 2279.44 +/- 152.30 | 137.00 +/- 12.20 | 212.93 +/- 61.01 | 1160.62 +/- 728.22 |

## low

| Controller | Loss s/veh | Throughput veh/h | Active unfinished vehicles | Pedestrian wait s | Worst crossing p95 s |
|---|---:|---:|---:|---:|---:|
| Baseline fixed_time | 44.65 +/- 16.11 | 2775.00 +/- 200.91 | 66.33 +/- 86.57 | 104.50 +/- 3.76 | 218.00 +/- 85.31 |
| Baseline pedestrian_actuated | 92.86 +/- 223.53 | 2418.67 +/- 1748.66 | 72.33 +/- 82.29 | 152.58 +/- 211.55 | 598.23 +/- 1748.82 |
| V3 mappo-vehicle | 44.65 +/- 0.00 | 2775.00 +/- 0.00 | 66.33 +/- 0.00 | 104.50 +/- 0.00 | 218.00 +/- 0.00 |
| V3 mappo-vehicle_pedestrian | 82.89 +/- 86.43 | 2549.78 +/- 593.83 | 80.33 +/- 76.49 | 116.07 +/- 58.20 | 483.58 +/- 643.55 |
| V4 mappo-vehicle | 89.11 +/- 126.48 | 2408.00 +/- 1078.63 | 79.11 +/- 63.78 | 146.65 +/- 117.52 | 727.33 +/- 1399.50 |
| V4 mappo-vehicle_pedestrian | 105.70 +/- 120.18 | 2413.78 +/- 903.83 | 95.89 +/- 21.28 | 133.98 +/- 113.26 | 666.42 +/- 1090.86 |
| V5 mappo-vehicle | 51.43 +/- 25.99 | 2783.00 +/- 121.20 | 66.67 +/- 28.48 | 103.68 +/- 3.70 | 207.26 +/- 43.10 |
| V5 mappo-vehicle_pedestrian | 91.09 +/- 89.55 | 2392.44 +/- 908.49 | 91.89 +/- 34.11 | 147.00 +/- 114.06 | 703.66 +/- 1349.02 |

## high

| Controller | Loss s/veh | Throughput veh/h | Active unfinished vehicles | Pedestrian wait s | Worst crossing p95 s |
|---|---:|---:|---:|---:|---:|
| Baseline fixed_time | 973.66 +/- 420.06 | 396.67 +/- 245.23 | 150.33 +/- 14.97 | 307.12 +/- 83.03 | 1226.17 +/- 557.24 |
| Baseline pedestrian_actuated | 654.94 +/- 272.67 | 613.33 +/- 350.75 | 147.67 +/- 7.59 | 247.76 +/- 20.27 | 826.03 +/- 100.82 |
| V3 mappo-vehicle | 973.66 +/- 0.00 | 396.67 +/- 0.00 | 150.33 +/- 0.00 | 307.12 +/- 0.00 | 1226.17 +/- 0.00 |
| V3 mappo-vehicle_pedestrian | 989.72 +/- 222.73 | 357.56 +/- 103.79 | 148.00 +/- 10.38 | 273.31 +/- 50.27 | 985.81 +/- 306.54 |
| V4 mappo-vehicle | 877.80 +/- 164.74 | 456.11 +/- 156.86 | 153.89 +/- 21.82 | 292.73 +/- 47.16 | 1100.78 +/- 259.81 |
| V4 mappo-vehicle_pedestrian | 858.18 +/- 427.28 | 483.00 +/- 344.83 | 156.89 +/- 11.18 | 274.75 +/- 40.21 | 1064.41 +/- 291.19 |
| V5 mappo-vehicle | 740.85 +/- 73.53 | 552.67 +/- 44.46 | 154.11 +/- 21.04 | 275.21 +/- 28.60 | 1112.17 +/- 395.56 |
| V5 mappo-vehicle_pedestrian | 740.85 +/- 73.53 | 552.67 +/- 44.46 | 154.11 +/- 21.04 | 275.21 +/- 28.60 | 1112.17 +/- 395.56 |

## slow

| Controller | Loss s/veh | Throughput veh/h | Active unfinished vehicles | Pedestrian wait s | Worst crossing p95 s |
|---|---:|---:|---:|---:|---:|
| Baseline fixed_time | 421.08 +/- 198.16 | 1060.00 +/- 419.02 | 150.00 +/- 9.94 | 254.67 +/- 45.11 | 859.17 +/- 182.92 |
| Baseline pedestrian_actuated | 405.50 +/- 87.21 | 1120.33 +/- 147.64 | 150.33 +/- 7.99 | 262.15 +/- 38.77 | 1057.67 +/- 300.36 |
| V3 mappo-vehicle | 421.08 +/- 0.00 | 1060.00 +/- 0.00 | 150.00 +/- 0.00 | 254.67 +/- 0.00 | 859.17 +/- 0.00 |
| V3 mappo-vehicle_pedestrian | 563.24 +/- 315.52 | 761.22 +/- 438.96 | 149.89 +/- 25.15 | 287.87 +/- 67.01 | 1099.67 +/- 192.71 |
| V4 mappo-vehicle | 466.80 +/- 366.09 | 985.11 +/- 820.83 | 150.11 +/- 2.08 | 267.73 +/- 79.45 | 1002.92 +/- 326.15 |
| V4 mappo-vehicle_pedestrian | 533.03 +/- 304.53 | 827.22 +/- 589.49 | 144.78 +/- 5.06 | 288.56 +/- 24.85 | 1101.67 +/- 177.58 |
| V5 mappo-vehicle | 454.91 +/- 223.27 | 1002.33 +/- 408.34 | 152.67 +/- 18.05 | 277.32 +/- 58.08 | 1065.44 +/- 175.47 |
| V5 mappo-vehicle_pedestrian | 460.68 +/- 213.32 | 985.22 +/- 383.45 | 152.78 +/- 18.34 | 273.25 +/- 61.35 | 1004.16 +/- 186.45 |

## Frozen selections

| Controller | Episode | Largest validation crossing excess s | Normalized violation |
|---|---:|---:|---:|
| mappo-vehicle-actions2-seed100 | 250 | 4.700 | 0.063968 |
| mappo-vehicle_pedestrian-actions2-seed100 | 200 | 4.533 | 0.062549 |
| mappo-vehicle-actions2-seed200 | 0 | 32.300 | 0.103094 |
| mappo-vehicle_pedestrian-actions2-seed200 | 0 | 32.300 | 0.103094 |
| mappo-vehicle-actions2-seed300 | 100 | 65.800 | 0.250203 |
| mappo-vehicle_pedestrian-actions2-seed300 | 100 | 65.800 | 0.250203 |

## BC versus selected validation

| Controller | Selected episode | BC loss s/veh | Selected loss s/veh | Change % | BC ped wait s | Selected ped wait s |
|---|---:|---:|---:|---:|---:|---:|
| mappo-vehicle-actions2-seed100 | 250 | 193.68 | 171.26 | -11.58 | 196.17 | 177.85 |
| mappo-vehicle_pedestrian-actions2-seed100 | 200 | 193.68 | 174.33 | -9.99 | 196.17 | 173.43 |
| mappo-vehicle-actions2-seed200 | 0 | 172.42 | 172.42 | 0.00 | 188.75 | 188.75 |
| mappo-vehicle_pedestrian-actions2-seed200 | 0 | 172.42 | 172.42 | 0.00 | 188.75 | 188.75 |
| mappo-vehicle-actions2-seed300 | 100 | 168.67 | 193.09 | 14.48 | 193.02 | 186.73 |
| mappo-vehicle_pedestrian-actions2-seed300 | 100 | 168.67 | 193.09 | 14.48 | 193.02 | 186.73 |

Validation gains above use the same data used for selection. The following primary comparison uses matched test traffic after frozen selection. The paired CI averages traffic seeds within each training seed, then uses the three training-seed differences.

## BC versus selected primary tests

| Controller | Metric | BC mean | Selected mean | Selected minus BC, 95% CI |
|---|---|---:|---:|---|
| mappo-vehicle | vehicle_mean_time_loss_seconds | 181.69 | 181.63 | -0.06 +/- 17.89 |
| mappo-vehicle | pedestrian_average_waiting_seconds | 192.88 | 207.68 | 14.80 +/- 56.83 |
| mappo-vehicle | worst_crossing_p95_waiting_seconds | 947.24 | 1101.88 | 154.63 +/- 614.15 |
| mappo-vehicle_pedestrian | vehicle_mean_time_loss_seconds | 181.69 | 182.59 | 0.90 +/- 14.44 |
| mappo-vehicle_pedestrian | pedestrian_average_waiting_seconds | 192.88 | 212.93 | 20.05 +/- 51.16 |
| mappo-vehicle_pedestrian | worst_crossing_p95_waiting_seconds | 947.24 | 1160.62 | 213.38 +/- 546.94 |

## Imitation quality

Training recall is measured on demonstrations. Rollout recall and precision use the five different warm-up simulation seeds and the expert labels on the BC policy states. Few positive labels make these estimates unstable. High overall agreement alone is insufficient.

| Controller | Demo switch labels | Training switch recall | Training false switches | Rollout expert switches | Rollout true switches | Rollout false switches |
|---|---:|---:|---:|---:|---:|---:|
| mappo-vehicle-actions2-seed100 | 20 | 1.000 | 304 | 4 | 0 | 15 |
| mappo-vehicle_pedestrian-actions2-seed100 | 20 | 1.000 | 304 | 4 | 0 | 15 |
| mappo-vehicle-actions2-seed200 | 27 | 1.000 | 365 | 4 | 3 | 23 |
| mappo-vehicle_pedestrian-actions2-seed200 | 27 | 1.000 | 365 | 4 | 3 | 23 |
| mappo-vehicle-actions2-seed300 | 33 | 1.000 | 299 | 2 | 0 | 20 |
| mappo-vehicle_pedestrian-actions2-seed300 | 33 | 1.000 | 299 | 2 | 0 | 20 |

## Completion

All pods and the temporary volume are deleted. Last pod termination: 22:49:15 SGT on 14 November, before the 00:30 deadline on 15 November. Conservative V5 runtime/storage bound is $1.22, including every setup and diagnostic attempt. Posted billing may lag. All 31 local tests passed; the audit verified 282 transfer entries, four archive SHA256s and all 4,841 protected prior files unchanged. See COST_ACCOUNTING.json, AUDIT_RESULTS.json and FINAL_RUN_STATUS.json.
