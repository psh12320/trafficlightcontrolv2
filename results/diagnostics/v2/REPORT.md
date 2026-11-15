# Local V2 diagnostics

Primary saved means: 3 test seeds per training seed, 3 training seeds per learned family; baselines have 3 test seeds.
Bold crossing values exceed the same crossing's actuated test mean. These are descriptive comparisons.

| Crossing | Approach | Actuated | IPPO vehicle | IPPO combined | MAPPO vehicle | MAPPO combined |
|---|---|---:|---:|---:|---:|---:|
| :J0_c0 | North | 166.1 | **175.6** | **170.4** | **175.1** | **439.5** |
| :J0_c1 | East | 95.7 | 86.7 | 88.1 | **99.1** | **214.3** |
| :J0_c2 | South | 186.8 | **196.3** | **210.2** | **197.2** | **489.8** |
| :J0_c3 | West | 165.5 | **178.4** | **189.5** | **180.6** | **461.5** |
| :J1_c0 | North | 213.4 | **382.4** | **450.6** | **576.8** | 188.0 |
| :J1_c1 | East | 98.8 | **187.5** | **197.9** | **343.8** | **99.0** |
| :J1_c2 | South | 252.9 | **455.6** | **457.8** | **674.6** | 250.3 |
| :J1_c3 | West | 296.3 | **433.3** | **439.9** | **624.3** | 280.6 |
| :J2_c0 | North | 572.2 | **649.1** | **790.1** | **577.7** | **767.5** |
| :J2_c1 | East | 634.0 | **721.6** | **805.9** | 605.7 | **675.3** |
| :J2_c2 | South | 597.8 | **671.8** | **836.0** | **623.4** | **737.6** |
| :J2_c3 | West | 670.3 | **694.2** | **767.1** | 618.5 | **695.0** |

## Deterministic local primary rollouts

Seed 11, 3600 seconds; selected MAPPO training-seed-100 checkpoints: vehicle episode 50, combined episode 250.
Phase transitions include green/yellow/all-red changes. Green terminations count exits from phase 0 or 3.
Mean green duration excludes the unfinished final green interval. Action shares include all 720 decisions per junction.

| Controller | Junction | Phase transitions | Green terminations | Mean green (s) | Hold % | Switch % | +5s % | +10s % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pedestrian_actuated | J0 | 229 | 77 | 36.87 | n.a. | n.a. | n.a. | n.a. |
| pedestrian_actuated | J1 | 229 | 77 | 36.86 | n.a. | n.a. | n.a. | n.a. |
| pedestrian_actuated | J2 | 228 | 76 | 36.93 | n.a. | n.a. | n.a. | n.a. |
| mappo-vehicle | J0 | 228 | 76 | 37.01 | 100.00 | 0.00 | 0.00 | 0.00 |
| mappo-vehicle | J1 | 195 | 65 | 44.94 | 18.06 | 0.00 | 81.94 | 0.00 |
| mappo-vehicle | J2 | 225 | 75 | 37.95 | 84.17 | 1.67 | 0.42 | 13.75 |
| mappo-vehicle_pedestrian | J0 | 384 | 128 | 18.05 | 82.36 | 17.64 | 0.00 | 0.00 |
| mappo-vehicle_pedestrian | J1 | 228 | 76 | 37.12 | 99.44 | 0.00 | 0.56 | 0.00 |
| mappo-vehicle_pedestrian | J2 | 195 | 65 | 44.94 | 18.06 | 0.00 | 81.94 | 0.00 |

## Saved unfinished counts and tail extremes

Individual maximum waits were not recorded in saved V2 CSVs or JSONs and cannot be recovered from p95 summaries.
The final column is the maximum CROSSING P95 across saved evaluations, not an individual maximum.
Unfinished includes moving pedestrians. Counts are mean +/- 95% CI, using the existing three-seed summary.

| Scenario | Controller | Unfinished mean +/- CI | Max unfinished | Max crossing p95 (s) |
|---|---|---:|---:|---:|
| primary | fixed_time | 107.00 +/- 22.77 | 115 | 1006.6 |
| primary | pedestrian_actuated | 124.00 +/- 110.65 | 172 | 863.0 |
| primary | ippo-vehicle | 122.11 +/- 33.58 | 167 | 1562.5 |
| primary | ippo-vehicle_pedestrian | 144.56 +/- 60.17 | 181 | 1512.7 |
| primary | mappo-vehicle | 144.00 +/- 17.62 | 177 | 1528.6 |
| primary | mappo-vehicle_pedestrian | 128.89 +/- 21.94 | 178 | 1516.3 |
| high | fixed_time | 329.00 +/- 61.20 | 355 | 1258.9 |
| high | pedestrian_actuated | 316.00 +/- 146.06 | 383 | 1039.5 |
| high | ippo-vehicle | 343.22 +/- 129.66 | 436 | 1483.1 |
| high | ippo-vehicle_pedestrian | 320.67 +/- 80.97 | 399 | 1652.3 |
| high | mappo-vehicle | 303.89 +/- 36.87 | 337 | 1203.6 |
| high | mappo-vehicle_pedestrian | 309.22 +/- 101.09 | 348 | 1094.2 |
| slow | fixed_time | 189.67 +/- 35.16 | 206 | 1412.2 |
| slow | pedestrian_actuated | 148.67 +/- 40.39 | 163 | 886.1 |
| slow | ippo-vehicle | 194.00 +/- 12.93 | 231 | 1506.0 |
| slow | ippo-vehicle_pedestrian | 194.33 +/- 19.10 | 226 | 1098.3 |
| slow | mappo-vehicle | 188.00 +/- 0.83 | 211 | 1341.2 |
| slow | mappo-vehicle_pedestrian | 203.33 +/- 88.93 | 259 | 1565.5 |

Exact observed individual maximum waits from the three new local rollouts:

| Controller | Max individual wait (s) | Unfinished | Replay matched saved metrics |
|---|---:|---:|---|
| pedestrian_actuated | 979 | 116 | True |
| mappo-vehicle | 2327 | 177 | True |
| mappo-vehicle_pedestrian | 1236 | 126 | True |

The maximum wait includes completed and still-active pedestrians' observed cumulative stopped time through 3600 seconds.
Full action commands, reward components, observation coverage, checksums and replay differences are in diagnostics.json.
No accepted V2 artifact or checkpoint was modified. No cloud resources were used.
