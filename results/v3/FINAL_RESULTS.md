# V3 Observation-Fixed & Fairness-Calibrated MARL

Twelve runs from scratch, 300 episodes x 3600s, randomized training demand, training seeds 100/200/300.
21 features: vehicle queues, approach waiting counts, current oldest waits, crossing occupancy, phase indicators/age, neighbor queues. Walking-area queues use the actual next crossing, with assigned start-edge fallback (explicitly approved after a local SUMO probe).
Two actions: hold or safe switch. Existing minimum-green and clearance masks remain active. Hold lets the programmed phase timer advance; it does not issue extension actions.
Vehicle scale 20.8; pedestrian scale 20; switch penalty 8. Combined reward additionally penalizes 0.5*sum((current crossing max wait/60)^2), summed across 12 crossings once per five-second step. Vehicle-only reward omits pedestrian and urgency terms.
The 8.0 switch cost is a reward coefficient, not a measurement of physical clearance duration.
Six snapshots at episodes 50/100/150/200/250/300 were validated on seeds 21/22/23 under primary demand. Frozen checkpoint selection precedes all final test evaluations.
Final test seeds: 41/42/43; verified absent from prior saved evaluation CSVs.
2/12 selected runs satisfy every crossing's mean validation p95 threshold. Other selections use the predeclared normalized positive violation fallback.
Means +/- 95% Student-t CI, n=3, df=2. Learned policies average test seeds within each training seed before constructing intervals. Baseline intervals use the three test seeds. Intervals are descriptive and unadjusted for multiple comparisons.
A zero-width learned-policy interval means the three training-seed averages are identical on this fixed test set. It does not mean there is no variation between traffic test seeds. MAPPO-vehicle's recorded metrics match fixed-time control in every evaluated scenario and test seed.
Vehicle loss includes active trips and is divided by departed vehicles. Throughput is completed trips per hour. Unfinished counts exclude demand not inserted. Pedestrian waits include active pedestrians and are observed to the cutoff; active waits are right-censored.
Crossing p95 retains the V2 route-assigned metric definition, including stopped time over that pedestrian's observed trip. It is not a measurement exclusively of red-signal waiting at one physical crossing.
Primary 1.4/6/20, low 1.4/10/30, high 1.4/4/12, slow 0.9/6/20 (speed/vehicle period/pedestrian interval); slow all-red is 8 seconds. All tests last 3600 seconds.
V2 and V3 change observation, action count, reward scaling, urgency and switching cost together, and use different test seeds. Their comparison is an ablation between configurations, not a causal estimate of any individual change.

## primary

| Controller | Vehicle loss (s/veh) | Throughput (veh/h) | Unfinished vehicles | Mean pedestrian wait (s) | Worst crossing p95 (s) | Unfinished pedestrians |
|---|---:|---:|---:|---:|---:|---:|
| fixed_time | 173.69 +/- 50.33 | 2379.33 +/- 493.56 | 138.33 +/- 8.72 | 189.42 +/- 104.98 | 974.80 +/- 1040.86 | 136.67 +/- 66.16 |
| pedestrian_actuated | 178.03 +/- 63.63 | 2325.00 +/- 818.80 | 134.00 +/- 11.38 | 191.48 +/- 121.04 | 919.67 +/- 1189.92 | 147.00 +/- 78.87 |
| ippo-vehicle | 191.01 +/- 41.01 | 2230.67 +/- 325.49 | 137.33 +/- 13.83 | 200.93 +/- 43.99 | 1024.08 +/- 460.71 | 143.00 +/- 48.73 |
| ippo-vehicle_pedestrian | 186.11 +/- 10.20 | 2269.56 +/- 184.01 | 142.11 +/- 5.51 | 182.24 +/- 17.69 | 966.91 +/- 178.76 | 138.22 +/- 35.72 |
| mappo-vehicle | 173.69 +/- 0.00 | 2379.33 +/- 0.00 | 138.33 +/- 0.00 | 189.42 +/- 0.00 | 974.80 +/- 0.00 | 136.67 +/- 0.00 |
| mappo-vehicle_pedestrian | 221.89 +/- 118.55 | 2017.56 +/- 839.94 | 146.33 +/- 16.12 | 192.08 +/- 116.80 | 935.48 +/- 574.34 | 139.78 +/- 103.07 |

## low

| Controller | Vehicle loss (s/veh) | Throughput (veh/h) | Unfinished vehicles | Mean pedestrian wait (s) | Worst crossing p95 (s) | Unfinished pedestrians |
|---|---:|---:|---:|---:|---:|---:|
| fixed_time | 44.65 +/- 16.11 | 2775.00 +/- 200.91 | 66.33 +/- 86.57 | 104.50 +/- 3.76 | 218.00 +/- 85.31 | 53.33 +/- 40.46 |
| pedestrian_actuated | 92.86 +/- 223.53 | 2418.67 +/- 1748.66 | 72.33 +/- 82.29 | 152.58 +/- 211.55 | 598.23 +/- 1748.82 | 73.00 +/- 131.24 |
| ippo-vehicle | 65.90 +/- 46.70 | 2652.78 +/- 401.86 | 71.78 +/- 12.29 | 120.48 +/- 59.35 | 416.80 +/- 619.05 | 67.11 +/- 31.01 |
| ippo-vehicle_pedestrian | 84.16 +/- 155.52 | 2490.22 +/- 1277.06 | 81.44 +/- 84.60 | 134.34 +/- 170.81 | 549.89 +/- 1439.22 | 74.33 +/- 117.72 |
| mappo-vehicle | 44.65 +/- 0.00 | 2775.00 +/- 0.00 | 66.33 +/- 0.00 | 104.50 +/- 0.00 | 218.00 +/- 0.00 | 53.33 +/- 0.00 |
| mappo-vehicle_pedestrian | 82.89 +/- 86.43 | 2549.78 +/- 593.83 | 80.33 +/- 76.49 | 116.07 +/- 58.20 | 483.58 +/- 643.55 | 58.33 +/- 40.19 |

## high

| Controller | Vehicle loss (s/veh) | Throughput (veh/h) | Unfinished vehicles | Mean pedestrian wait (s) | Worst crossing p95 (s) | Unfinished pedestrians |
|---|---:|---:|---:|---:|---:|---:|
| fixed_time | 973.66 +/- 420.06 | 396.67 +/- 245.23 | 150.33 +/- 14.97 | 307.12 +/- 83.03 | 1226.17 +/- 557.24 | 345.00 +/- 86.38 |
| pedestrian_actuated | 654.94 +/- 272.67 | 613.33 +/- 350.75 | 147.67 +/- 7.59 | 247.76 +/- 20.27 | 826.03 +/- 100.82 | 280.67 +/- 117.05 |
| ippo-vehicle | 928.42 +/- 194.64 | 417.67 +/- 90.36 | 150.67 +/- 1.43 | 298.78 +/- 35.87 | 1145.20 +/- 348.37 | 336.56 +/- 36.33 |
| ippo-vehicle_pedestrian | 825.74 +/- 327.37 | 477.22 +/- 292.54 | 150.00 +/- 15.73 | 264.93 +/- 20.25 | 970.02 +/- 274.76 | 333.78 +/- 45.70 |
| mappo-vehicle | 973.66 +/- 0.00 | 396.67 +/- 0.00 | 150.33 +/- 0.00 | 307.12 +/- 0.00 | 1226.17 +/- 0.00 | 345.00 +/- 0.00 |
| mappo-vehicle_pedestrian | 989.72 +/- 222.73 | 357.56 +/- 103.79 | 148.00 +/- 10.38 | 273.31 +/- 50.27 | 985.81 +/- 306.54 | 321.78 +/- 60.24 |

## slow

| Controller | Vehicle loss (s/veh) | Throughput (veh/h) | Unfinished vehicles | Mean pedestrian wait (s) | Worst crossing p95 (s) | Unfinished pedestrians |
|---|---:|---:|---:|---:|---:|---:|
| fixed_time | 421.08 +/- 198.16 | 1060.00 +/- 419.02 | 150.00 +/- 9.94 | 254.67 +/- 45.11 | 859.17 +/- 182.92 | 179.00 +/- 32.48 |
| pedestrian_actuated | 405.50 +/- 87.21 | 1120.33 +/- 147.64 | 150.33 +/- 7.99 | 262.15 +/- 38.77 | 1057.67 +/- 300.36 | 164.00 +/- 16.29 |
| ippo-vehicle | 408.72 +/- 53.21 | 1059.44 +/- 2.39 | 149.44 +/- 2.39 | 254.88 +/- 0.90 | 882.31 +/- 99.58 | 177.44 +/- 6.69 |
| ippo-vehicle_pedestrian | 520.59 +/- 188.08 | 838.56 +/- 357.49 | 154.56 +/- 28.21 | 290.96 +/- 57.23 | 1130.46 +/- 146.95 | 200.22 +/- 49.83 |
| mappo-vehicle | 421.08 +/- 0.00 | 1060.00 +/- 0.00 | 150.00 +/- 0.00 | 254.67 +/- 0.00 | 859.17 +/- 0.00 | 179.00 +/- 0.00 |
| mappo-vehicle_pedestrian | 563.24 +/- 315.52 | 761.22 +/- 438.96 | 149.89 +/- 25.15 | 287.87 +/- 67.01 | 1099.67 +/- 192.71 | 187.33 +/- 52.06 |

## Frozen selections

| Controller | Episode | Validation feasible |
|---|---:|---|
| ippo-vehicle-actions2-seed100 | 50 | False |
| ippo-vehicle_pedestrian-actions2-seed100 | 150 | False |
| mappo-vehicle-actions2-seed100 | 100 | False |
| mappo-vehicle_pedestrian-actions2-seed100 | 50 | False |
| ippo-vehicle-actions2-seed200 | 50 | False |
| ippo-vehicle_pedestrian-actions2-seed200 | 300 | False |
| mappo-vehicle-actions2-seed200 | 150 | False |
| mappo-vehicle_pedestrian-actions2-seed200 | 250 | True |
| ippo-vehicle-actions2-seed300 | 100 | False |
| ippo-vehicle_pedestrian-actions2-seed300 | 50 | False |
| mappo-vehicle-actions2-seed300 | 100 | False |
| mappo-vehicle_pedestrian-actions2-seed300 | 100 | True |

## Primary crossing p95 (seconds)

| Crossing | fixed_time | pedestrian_actuated | ippo-vehicle | ippo-vehicle_pedestrian | mappo-vehicle | mappo-vehicle_pedestrian |
|---|---:|---:|---:|---:|---:|---:|
| :J0_c0 | 162.4 | 160.7 | 162.4 | 148.6 | 162.4 | 116.8 |
| :J0_c1 | 94.7 | 95.4 | 94.1 | 72.2 | 94.7 | 55.9 |
| :J0_c2 | 181.1 | 172.5 | 178.8 | 179.8 | 181.1 | 143.3 |
| :J0_c3 | 163.2 | 162.4 | 161.4 | 157.1 | 163.2 | 133.7 |
| :J1_c0 | 177.6 | 233.6 | 421.4 | 453.7 | 177.6 | 479.6 |
| :J1_c1 | 96.7 | 120.2 | 220.9 | 217.3 | 96.7 | 228.6 |
| :J1_c2 | 228.4 | 301.9 | 454.8 | 517.2 | 228.4 | 474.6 |
| :J1_c3 | 267.9 | 327.3 | 460.7 | 491.8 | 267.9 | 455.2 |
| :J2_c0 | 856.9 | 880.3 | 752.8 | 634.0 | 856.9 | 807.6 |
| :J2_c1 | 871.3 | 822.0 | 783.3 | 639.1 | 871.3 | 828.5 |
| :J2_c2 | 964.9 | 852.2 | 830.2 | 631.9 | 964.9 | 824.4 |
| :J2_c3 | 838.8 | 787.1 | 788.0 | 631.7 | 838.8 | 732.1 |
