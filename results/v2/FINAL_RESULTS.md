# V2 300-episode results

Twelve runs: IPPO/MAPPO × vehicle/vehicle-pedestrian rewards × training seeds 100/200/300. Each run trained from scratch for 300 episodes of 3,600 seconds.
Vehicle waiting is divided by 20.8 and pedestrian waiting by 81.35 in the proposal objectives. The switch penalty is 1.5. Training demand samples vehicle periods 4/6/8 and pedestrian intervals 12/20/30 with a seeded private RNG and isolated network files.
Snapshots at episodes 50/100/150/200/250/300 were evaluated on fixed validation seeds 21/22/23, primary demand, duration 3,600 seconds. Selections were frozen before V2 testing.
2/12 selected runs met every crossing's mean validation p95 limit. The other 10 used the predeclared lowest normalized positive violation fallback.
Mean ± 95% Student-t CI (n=3, df=2). MARL: average three test seeds within each training seed, then form the CI across training seeds. Baseline: CI across the three test seeds. Intervals are descriptive and unadjusted for multiple comparisons.
Test seeds 11/12/13 were excluded from V2 training and checkpoint selection, but reused from the pilot that informed the V2 changes. They are not a fresh independent confirmatory test set.
Vehicle time loss is measured total loss divided by departed vehicles. Throughput is completed vehicles per simulated hour. Active unfinished counts exclude demand still waiting for insertion. Pedestrian means and percentiles include active unfinished pedestrians.
Waiting metrics are observed to the simulation cutoff; waits for unfinished trips are right-censored.
Worst crossing p95 is the maximum of the 12 per-crossing percentiles per evaluation, averaged across evaluations and training seeds. The summary CSV additionally retains total time loss, pedestrian unfinished counts, and maximum observed values.
Primary: speed 1.4, vehicle period 6, pedestrian interval 20. Low: 1.4/10/30. High: 1.4/4/12. Slow: 0.9/6/20 with eight-second clearance. All tests last 3,600 seconds.
The pilot comparison changes reward scaling, switching penalty, training duration, and demand randomization together. It compares configurations and cannot isolate the causal contribution of each change.

## primary

| Controller | Time loss (s/vehicle) | Throughput (veh/h) | Active unfinished | Mean pedestrian wait (s) | Worst crossing p95 (s) |
|---|---:|---:|---:|---:|---:|
| fixed_time | 176.09 ± 18.03 | 2362.67 ± 159.73 | 135.33 ± 17.45 | 174.29 ± 19.56 | 862.30 ± 394.58 |
| pedestrian_actuated | 168.65 ± 35.73 | 2476.67 ± 367.99 | 137.67 ± 37.29 | 164.87 ± 23.55 | 670.33 ± 417.67 |
| ippo-vehicle | 185.52 ± 92.87 | 2295.11 ± 840.67 | 144.78 ± 12.40 | 188.70 ± 54.26 | 880.30 ± 668.40 |
| ippo-vehicle_pedestrian | 198.95 ± 70.17 | 2157.44 ± 566.01 | 143.56 ± 11.12 | 198.58 ± 54.62 | 998.13 ± 353.75 |
| mappo-vehicle | 184.59 ± 37.87 | 2297.11 ± 320.64 | 144.33 ± 15.73 | 202.38 ± 7.71 | 1032.04 ± 343.13 |
| mappo-vehicle_pedestrian | 204.88 ± 57.73 | 2124.11 ± 447.05 | 141.56 ± 10.65 | 199.02 ± 68.91 | 1017.01 ± 334.27 |

## low

| Controller | Time loss (s/vehicle) | Throughput (veh/h) | Active unfinished | Mean pedestrian wait (s) | Worst crossing p95 (s) |
|---|---:|---:|---:|---:|---:|
| fixed_time | 81.06 ± 160.02 | 2464.33 ± 1366.13 | 79.33 ± 69.39 | 142.96 ± 165.08 | 617.73 ± 1724.67 |
| pedestrian_actuated | 94.21 ± 210.63 | 2430.00 ± 1723.31 | 70.67 ± 94.83 | 151.04 ± 202.19 | 637.00 ± 1893.04 |
| ippo-vehicle | 48.03 ± 19.27 | 2770.44 ± 225.13 | 62.44 ± 38.08 | 107.00 ± 9.50 | 278.48 ± 222.97 |
| ippo-vehicle_pedestrian | 64.41 ± 72.94 | 2642.33 ± 568.29 | 75.78 ± 41.11 | 115.34 ± 27.68 | 362.41 ± 540.50 |
| mappo-vehicle | 61.52 ± 41.95 | 2704.78 ± 494.17 | 62.22 ± 35.37 | 120.83 ± 37.40 | 349.81 ± 564.91 |
| mappo-vehicle_pedestrian | 100.76 ± 79.84 | 2353.11 ± 540.35 | 93.89 ± 15.05 | 138.08 ± 57.62 | 645.18 ± 389.75 |

## high

| Controller | Time loss (s/vehicle) | Throughput (veh/h) | Active unfinished | Mean pedestrian wait (s) | Worst crossing p95 (s) |
|---|---:|---:|---:|---:|---:|
| fixed_time | 932.06 ± 821.73 | 478.67 ± 629.18 | 159.00 ± 22.36 | 288.40 ± 27.39 | 1165.83 ± 245.66 |
| pedestrian_actuated | 691.22 ± 232.44 | 590.67 ± 245.97 | 152.33 ± 21.70 | 262.62 ± 39.99 | 935.27 ± 263.04 |
| ippo-vehicle | 718.67 ± 248.89 | 555.44 ± 228.87 | 149.56 ± 13.10 | 285.73 ± 48.34 | 1116.94 ± 334.36 |
| ippo-vehicle_pedestrian | 739.52 ± 393.24 | 605.11 ± 413.86 | 159.78 ± 6.69 | 269.35 ± 42.55 | 1079.06 ± 444.08 |
| mappo-vehicle | 682.32 ± 144.43 | 610.78 ± 136.24 | 153.89 ± 27.74 | 269.28 ± 42.94 | 1004.83 ± 280.06 |
| mappo-vehicle_pedestrian | 776.95 ± 315.51 | 517.67 ± 254.08 | 150.11 ± 23.79 | 275.96 ± 53.16 | 979.27 ± 226.09 |

## slow

| Controller | Time loss (s/vehicle) | Throughput (veh/h) | Active unfinished | Mean pedestrian wait (s) | Worst crossing p95 (s) |
|---|---:|---:|---:|---:|---:|
| fixed_time | 395.23 ± 213.78 | 1124.33 ± 718.57 | 153.33 ± 18.64 | 268.19 ± 9.04 | 1152.17 ± 643.28 |
| pedestrian_actuated | 398.62 ± 172.16 | 1101.33 ± 385.34 | 145.00 ± 8.96 | 241.73 ± 14.91 | 864.40 ± 80.80 |
| ippo-vehicle | 444.42 ± 119.33 | 980.56 ± 358.11 | 147.00 ± 9.22 | 296.93 ± 10.55 | 1256.51 ± 382.66 |
| ippo-vehicle_pedestrian | 478.11 ± 442.57 | 960.78 ± 915.30 | 150.11 ± 10.35 | 269.30 ± 43.67 | 903.57 ± 102.23 |
| mappo-vehicle | 353.79 ± 64.47 | 1264.78 ± 161.03 | 149.56 ± 17.02 | 259.13 ± 23.31 | 1092.16 ± 51.22 |
| mappo-vehicle_pedestrian | 449.28 ± 297.28 | 1003.89 ± 507.78 | 146.22 ± 7.47 | 291.67 ± 84.70 | 1152.49 ± 244.73 |

## V2 versus archived pilot (point-estimate changes)

Negative changes favor V2 for time loss and waiting; positive changes favor V2 for throughput. These percentage changes are not significance tests.

| Scenario | Controller | Vehicle loss change | Throughput change | Pedestrian wait change | Worst crossing p95 change |
|---|---|---:|---:|---:|---:|
| primary | fixed_time | +0.00% | +0.00% | +0.00% | +0.00% |
| primary | pedestrian_actuated | +0.00% | +0.00% | +0.00% | +0.00% |
| primary | ippo-vehicle | +2.44% | -0.95% | +3.36% | -1.94% |
| primary | ippo-vehicle_pedestrian | -4.53% | +4.19% | -7.84% | -16.42% |
| primary | mappo-vehicle | -2.17% | +3.18% | +1.74% | -1.43% |
| primary | mappo-vehicle_pedestrian | +1.43% | +1.30% | +6.92% | +8.75% |
| low | fixed_time | +0.00% | +0.00% | +0.00% | +0.00% |
| low | pedestrian_actuated | +0.00% | +0.00% | +0.00% | +0.00% |
| low | ippo-vehicle | -2.62% | -1.71% | -1.30% | +28.04% |
| low | ippo-vehicle_pedestrian | -19.36% | +4.27% | -8.60% | -30.18% |
| low | mappo-vehicle | -30.85% | +9.61% | -12.44% | -37.18% |
| low | mappo-vehicle_pedestrian | -6.47% | -1.20% | -1.85% | +2.99% |
| high | fixed_time | +0.00% | +0.00% | +0.00% | +0.00% |
| high | pedestrian_actuated | +0.00% | +0.00% | +0.00% | +0.00% |
| high | ippo-vehicle | -4.33% | -2.74% | +2.36% | -1.12% |
| high | ippo-vehicle_pedestrian | -8.38% | +12.64% | -2.10% | +2.79% |
| high | mappo-vehicle | -9.29% | +7.11% | -0.32% | +6.86% |
| high | mappo-vehicle_pedestrian | -1.16% | -1.38% | +1.06% | -1.98% |
| slow | fixed_time | +0.00% | +0.00% | +0.00% | +0.00% |
| slow | pedestrian_actuated | +0.00% | +0.00% | +0.00% | +0.00% |
| slow | ippo-vehicle | +12.26% | -11.66% | +13.28% | +21.79% |
| slow | ippo-vehicle_pedestrian | -5.16% | +10.00% | -8.33% | -18.95% |
| slow | mappo-vehicle | -24.96% | +32.99% | -4.79% | +8.87% |
| slow | mappo-vehicle_pedestrian | -22.20% | +30.51% | -2.76% | -0.52% |
