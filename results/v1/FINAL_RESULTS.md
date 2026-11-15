# Proposal pilot results

Mean ± 95% Student-t CI (n=3). MARL: test seeds are averaged within each training seed; baseline: CI across test seeds.
Vehicle time loss is total measured loss divided by departed vehicles; throughput is completed vehicles per simulated hour.
Worst crossing p95 is the maximum of 12 crossing percentiles per evaluation, then averaged across evaluations/seeds.
Intervals are descriptive and unadjusted for comparisons. Unfinished pedestrians remain in waiting metrics.

Active unfinished vehicles exclude vehicles still queued for insertion. Time loss per vehicle uses departed vehicles as its denominator.
Only 1 of 12 selected runs met every crossing's validation limit; 11 used the predeclared lowest normalized violation fallback.
The high-demand evaluation was rerun with pedestrian injection every simulated second. Accepted high-demand rows each contain 3,588 injected pedestrians; initial high-demand results are diagnostic only.
The selected checkpoints were frozen before held-out testing. Test seeds were held out from this run's training/selection, but were used in earlier exploratory work.
Raw CSVs retain total vehicle time loss; final-summary.csv also reports pedestrian unfinished counts and the maximum observed value for each metric.

## primary

| Controller | Time loss (s/vehicle) | Throughput (veh/h) | Active unfinished vehicles | Mean pedestrian wait (s) | Worst crossing p95 (s) |
|---|---:|---:|---:|---:|---:|
| fixed_time | 176.09 ± 18.03 | 2362.67 ± 159.73 | 135.33 ± 17.45 | 174.29 ± 19.56 | 862.30 ± 394.58 |
| pedestrian_actuated | 168.65 ± 35.73 | 2476.67 ± 367.99 | 137.67 ± 37.29 | 164.87 ± 23.55 | 670.33 ± 417.67 |
| ippo-vehicle | 181.09 ± 48.01 | 2317.11 ± 455.19 | 142.11 ± 5.87 | 182.58 ± 40.48 | 897.72 ± 400.71 |
| ippo-vehicle_pedestrian | 208.39 ± 37.16 | 2070.67 ± 379.48 | 143.22 ± 6.27 | 215.48 ± 32.96 | 1194.16 ± 262.14 |
| mappo-vehicle | 188.69 ± 44.91 | 2226.33 ± 457.83 | 139.11 ± 12.54 | 198.91 ± 65.30 | 1047.06 ± 696.83 |
| mappo-vehicle_pedestrian | 201.98 ± 41.58 | 2096.78 ± 376.93 | 140.44 ± 4.08 | 186.14 ± 70.02 | 935.21 ± 932.24 |

## low

| Controller | Time loss (s/vehicle) | Throughput (veh/h) | Active unfinished vehicles | Mean pedestrian wait (s) | Worst crossing p95 (s) |
|---|---:|---:|---:|---:|---:|
| fixed_time | 81.06 ± 160.02 | 2464.33 ± 1366.13 | 79.33 ± 69.39 | 142.96 ± 165.08 | 617.73 ± 1724.67 |
| pedestrian_actuated | 94.21 ± 210.63 | 2430.00 ± 1723.31 | 70.67 ± 94.83 | 151.04 ± 202.19 | 637.00 ± 1893.04 |
| ippo-vehicle | 49.32 ± 14.36 | 2818.56 ± 33.99 | 54.56 ± 18.89 | 108.41 ± 12.55 | 217.49 ± 0.51 |
| ippo-vehicle_pedestrian | 79.87 ± 57.80 | 2534.22 ± 407.95 | 80.89 ± 34.11 | 126.19 ± 45.77 | 519.09 ± 326.25 |
| mappo-vehicle | 88.96 ± 51.55 | 2467.67 ± 476.26 | 88.22 ± 15.54 | 137.99 ± 58.84 | 556.83 ± 545.41 |
| mappo-vehicle_pedestrian | 107.74 ± 13.86 | 2381.78 ± 63.42 | 87.78 ± 44.56 | 140.68 ± 41.14 | 626.43 ± 115.76 |

## high

| Controller | Time loss (s/vehicle) | Throughput (veh/h) | Active unfinished vehicles | Mean pedestrian wait (s) | Worst crossing p95 (s) |
|---|---:|---:|---:|---:|---:|
| fixed_time | 932.06 ± 821.73 | 478.67 ± 629.18 | 159.00 ± 22.36 | 288.40 ± 27.39 | 1165.83 ± 245.66 |
| pedestrian_actuated | 691.22 ± 232.44 | 590.67 ± 245.97 | 152.33 ± 21.70 | 262.62 ± 39.99 | 935.27 ± 263.04 |
| ippo-vehicle | 751.19 ± 269.82 | 571.11 ± 225.96 | 157.22 ± 7.69 | 279.15 ± 42.78 | 1129.61 ± 579.77 |
| ippo-vehicle_pedestrian | 807.12 ± 335.60 | 537.22 ± 90.09 | 157.00 ± 33.03 | 275.14 ± 19.46 | 1049.72 ± 511.96 |
| mappo-vehicle | 752.20 ± 250.83 | 570.22 ± 191.46 | 162.00 ± 12.50 | 270.14 ± 25.38 | 940.33 ± 57.14 |
| mappo-vehicle_pedestrian | 786.04 ± 214.74 | 524.89 ± 177.49 | 155.00 ± 1.43 | 273.06 ± 38.20 | 999.08 ± 136.06 |

## slow

| Controller | Time loss (s/vehicle) | Throughput (veh/h) | Active unfinished vehicles | Mean pedestrian wait (s) | Worst crossing p95 (s) |
|---|---:|---:|---:|---:|---:|
| fixed_time | 395.23 ± 213.78 | 1124.33 ± 718.57 | 153.33 ± 18.64 | 268.19 ± 9.04 | 1152.17 ± 643.28 |
| pedestrian_actuated | 398.62 ± 172.16 | 1101.33 ± 385.34 | 145.00 ± 8.96 | 241.73 ± 14.91 | 864.40 ± 80.80 |
| ippo-vehicle | 395.90 ± 182.82 | 1110.00 ± 550.12 | 154.44 ± 19.12 | 262.11 ± 32.93 | 1031.72 ± 45.77 |
| ippo-vehicle_pedestrian | 504.13 ± 276.76 | 873.44 ± 532.04 | 150.78 ± 10.78 | 293.76 ± 49.78 | 1114.89 ± 252.42 |
| mappo-vehicle | 471.49 ± 252.94 | 951.00 ± 379.73 | 153.11 ± 3.73 | 272.16 ± 32.30 | 1003.14 ± 207.28 |
| mappo-vehicle_pedestrian | 577.45 ± 70.13 | 769.22 ± 115.19 | 154.33 ± 11.68 | 299.94 ± 43.22 | 1158.56 ± 230.90 |
