# V4 convergence diagnostics

Local saved-data analysis only. No cloud operations or simulations were performed. All six logs contain consecutive episodes 1..400. All six checkpoint-metrics files contain eight snapshots x three validation seeds = 144 rows.

**Conclusion:** convergence by episode 400 is not established. Combined pedestrian validation metrics and crossing fairness improved across 300/350/400, but vehicle loss was not monotonic and the late training reward slope is not consistently positive across seeds. Further improvement after 400 is unobserved.

## Validation trajectory

For each snapshot, average validation seeds 21/22/23 within each training seed, then average training seeds 100/200/300. Vehicle loss divides total loss by departed vehicles separately in each evaluation. Worst crossing p95 is the maximum of 12 crossing p95s within each evaluation, then averaged.

For each training seed, crossing p95s are first averaged over validation seeds. Relative to the corresponding actuated crossing means b_c, normalized violation = sum_c max(p95_c-b_c,0)/max(b_c,1s). Largest excess = max_c(p95_c-b_c). The table averages these two quantities across training seeds, preserving the original selector order of operations. They are not calculated after pooling policies. Lower is better for every column.

### mappo-vehicle

| Episode | Loss s/veh | Mean ped wait s | Worst crossing p95 s | Normalized violation | Mean largest crossing excess s |
|---:|---:|---:|---:|---:|---:|
| 50 | 203.7217 | 190.9147 | 958.1444 | 0.4614 | 93.3556 |
| 100 | 185.8570 | 182.4357 | 858.4667 | 0.3996 | 53.7000 |
| 150 | 196.7099 | 199.8161 | 979.6667 | 2.7594 | 161.1222 |
| 200 | 183.1698 | 184.7181 | 919.1889 | 0.3211 | 96.9222 |
| 250 | 185.0172 | 173.2509 | 741.9889 | 0.1785 | 17.6667 |
| 300 | 180.8603 | 187.4981 | 933.6222 | 1.0915 | 100.9444 |
| 350 | 181.5459 | 187.8994 | 913.2222 | 0.6706 | 86.4556 |
| 400 | 193.5128 | 208.0982 | 1004.3222 | 1.2595 | 120.5333 |

### mappo-vehicle_pedestrian

| Episode | Loss s/veh | Mean ped wait s | Worst crossing p95 s | Normalized violation | Mean largest crossing excess s |
|---:|---:|---:|---:|---:|---:|
| 50 | 240.3791 | 198.9981 | 996.7333 | 1.0599 | 137.6889 |
| 100 | 225.8536 | 196.7049 | 1083.1778 | 1.1873 | 197.5556 |
| 150 | 211.3204 | 175.3714 | 818.9222 | 2.7009 | 133.9667 |
| 200 | 197.4693 | 181.4655 | 955.3111 | 0.1793 | 24.2333 |
| 250 | 228.1653 | 199.4784 | 1085.8889 | 5.0234 | 287.4667 |
| 300 | 221.6092 | 215.5326 | 1134.4333 | 2.5734 | 157.6111 |
| 350 | 230.2156 | 186.7926 | 1000.8000 | 0.3842 | 33.8222 |
| 400 | 210.8209 | 163.9149 | 897.1000 | 0.0826 | 31.7000 |

## Trailing 25-episode rewards

Each cell is mean +/- sample standard deviation of the last 25 raw episode rewards, within one training seed. These are SDs, not confidence intervals. Higher (less negative) reward is better. Reward magnitudes differ between objectives and should not be compared across them.

### mappo-vehicle

| Episode | Training seed 100 | Training seed 200 | Training seed 300 |
|---:|---:|---:|---:|
| 50 | -20557.73 +/- 1008.58 | -20538.14 +/- 1126.70 | -20297.33 +/- 928.63 |
| 100 | -19943.09 +/- 776.26 | -19641.11 +/- 576.88 | -19626.60 +/- 764.82 |
| 150 | -19961.83 +/- 745.14 | -19051.20 +/- 1055.15 | -19614.98 +/- 963.77 |
| 200 | -19508.34 +/- 851.03 | -19425.82 +/- 884.13 | -19826.65 +/- 1036.97 |
| 250 | -19251.48 +/- 620.00 | -19195.99 +/- 934.92 | -19437.82 +/- 1224.03 |
| 300 | -19658.17 +/- 1349.39 | -19070.92 +/- 1125.14 | -19466.64 +/- 971.63 |
| 350 | -19468.21 +/- 1373.05 | -18752.38 +/- 701.02 | -19362.66 +/- 1163.36 |
| 400 | -18938.43 +/- 827.39 | -19022.65 +/- 1008.97 | -19473.28 +/- 1029.99 |

### mappo-vehicle_pedestrian

| Episode | Training seed 100 | Training seed 200 | Training seed 300 |
|---:|---:|---:|---:|
| 50 | -56033.78 +/- 12583.42 | -54272.93 +/- 9412.50 | -52527.80 +/- 10936.49 |
| 100 | -60730.93 +/- 15289.79 | -57154.06 +/- 11012.96 | -55437.18 +/- 19592.91 |
| 150 | -56923.26 +/- 13745.22 | -52262.77 +/- 9494.91 | -54846.17 +/- 10363.49 |
| 200 | -57559.31 +/- 11539.37 | -58909.50 +/- 13331.31 | -55337.80 +/- 15689.64 |
| 250 | -56829.48 +/- 11649.43 | -58073.04 +/- 15209.06 | -57474.92 +/- 12704.25 |
| 300 | -61319.18 +/- 13889.94 | -53215.50 +/- 10701.03 | -60311.92 +/- 15863.74 |
| 350 | -57673.40 +/- 14744.36 | -60999.74 +/- 17526.70 | -57950.81 +/- 13845.32 |
| 400 | -55916.42 +/- 11277.34 | -60320.71 +/- 12536.15 | -62127.54 +/- 14366.54 |

## Reward slopes

Fit raw reward against episode with an intercept independently for each training seed. The primary window 250..400 includes 151 observations per fit. Form the 95% Student-t interval from the three seed slopes: mean +/- 4.3026527299*sample_SD/sqrt(3), df=2. Episodes are not treated as 151 independent training replicates. The CI describes variation in fitted slopes across training seeds. Serial dependence within a run and nonlinear changes remain possible.

| Controller | Window | Seed 100 | Seed 200 | Seed 300 | Mean slope | 95% CI |
|---|---|---:|---:|---:|---:|---|
| mappo-vehicle | 250..400 | 2.7047 | -0.4300 | 1.1912 | 1.1553 | [-2.7390, 5.0496] |
| mappo-vehicle | 300..400 | 1.5140 | 0.4847 | -4.5655 | -0.8556 | [-8.9385, 7.2273] |
| mappo-vehicle | 350..400 | 9.2334 | -6.5198 | -6.6587 | -1.3150 | [-24.0088, 21.3788] |
| mappo-vehicle_pedestrian | 250..400 | 9.4409 | -44.3052 | -4.0511 | -12.9718 | [-82.4321, 56.4885] |
| mappo-vehicle_pedestrian | 300..400 | -4.3012 | -14.3315 | 33.4402 | 4.9358 | [-57.6391, 67.5107] |
| mappo-vehicle_pedestrian | 350..400 | 15.5127 | -89.1214 | -166.8249 | -80.1445 | [-307.4424, 147.1533] |

Slope units are reward units per episode. Positive means improvement. Both primary intervals span zero, so neither a persistent upward reward trend nor equivalence to a flat plateau is established with three seeds. The 300..400 and 350..400 fits are descriptive window sensitivity checks, not additional independent tests.

## Primary vehicle loss by traffic seed

| Traffic seed | V4 train 100 | V4 train 200 | V4 train 300 | V4 policy average | Actuated | Paired V4-actuated |
|---:|---:|---:|---:|---:|---:|---:|
| 41 | 170.5692 | 180.8862 | 166.4982 | 172.6512 | 197.3443 | -24.6931 |
| 42 | 187.2221 | 175.3814 | 158.0033 | 173.5356 | 148.9770 | 24.5586 |
| 43 | 189.6906 | 178.4664 | 196.5628 | 188.2399 | 187.7817 | 0.4583 |

All cells are seconds per departed vehicle. V4 policy average averages the three learned training seeds for that same traffic seed. Test seeds 41/42/43 were reused from V3 and held out of V4 training and selection.

| Estimate | Uncertainty unit | Mean | 95% CI half-width |
|---|---|---:|---:|
| V4 vehicle | training_seed_average_of_traffic_seeds | 178.1422 | 10.9397 |
| V4 vehicle | traffic_seed_average_of_training_seeds | 178.1422 | 21.7512 |
| pedestrian_actuated | traffic_seed | 178.0343 | 63.6302 |
| V4 minus actuated | paired_traffic_seed_average_of_training_seeds | 0.1079 | 61.1786 |

The original +/- 10.94 summarizes three training-seed means, each averaging three traffic seeds. The actuated +/- 63.63 summarizes three traffic seeds directly. Their widths therefore measure different sources of variability. For a shared traffic-seed unit, V4 is 178.14 +/- 21.75 versus actuated 178.03 +/- 63.63. Averaging three V4 policies can smooth policy-specific traffic responses. This does not prove a population variance advantage. The paired mean difference is 0.11 +/- 61.18 seconds per vehicle.

## Interpretation

Combined 300/350/400 mean pedestrian wait, worst-crossing p95, normalized crossing violation and mean largest crossing excess all decrease. Vehicle loss first rises at 350 before falling at 400. Normalized violation decreases for training seeds 100/200 and is flat then improves for 300. Largest crossing excess rises from 350 to 400 for seed 100 (67.83 to 89.97s), despite improvement in its summed normalized violation. Thus aggregate progress does not mean every crossing or every metric improved in every run.

Training samples stochastic actions and changes simulation seed each episode (training_seed + zero_based_episode). Validation uses deterministic argmax actions on fixed seeds 21/22/23. Training reward also measures waiting costs, urgency and switching, while validation separately reports vehicle time loss and pedestrian tails. These differences can produce divergent reward and validation trends.

The six frozen choices remain vehicle 200/50/100 and combined 150/400/400 for training seeds 100/200/300. Selection uses a fairness constraint/fallback, so a late selection is not by itself proof of an unconverged reward objective. Vehicle validation is oscillatory. Combined shows substantial late pedestrian/fairness validation improvement, but training reward is noisy and has no consistently positive late slope. The saved data support possible remaining headroom, not a claim that longer training will reliably help or that the objective has converged.

## Sources and verification

All 271 files in the frozen V4 local artifact manifest were hash-verified before and after this analysis. Original logs, metrics, checkpoints, frozen selections, reports and result bundles were not overwritten. Input SHA256s and unrounded calculations are in diagnostics.json.

- v4-source-upload.zip
- v4-source-upload.zip::scripts/train_marl.py
- v4-source-upload.zip::scripts/evaluate_marl.py
- validation-baseline/validation-baseline.csv
- training/v4-400ep-seed100/mappo-vehicle-actions2-seed100/checkpoint-metrics.csv
- training/v4-400ep-seed100/mappo-vehicle-actions2-seed100/training.log
- training/v4-400ep-seed100/mappo-vehicle_pedestrian-actions2-seed100/checkpoint-metrics.csv
- training/v4-400ep-seed100/mappo-vehicle_pedestrian-actions2-seed100/training.log
- training/v4-400ep-seed200/mappo-vehicle-actions2-seed200/checkpoint-metrics.csv
- training/v4-400ep-seed200/mappo-vehicle-actions2-seed200/training.log
- training/v4-400ep-seed200/mappo-vehicle_pedestrian-actions2-seed200/checkpoint-metrics.csv
- training/v4-400ep-seed200/mappo-vehicle_pedestrian-actions2-seed200/training.log
- training/v4-400ep-seed300/mappo-vehicle-actions2-seed300/checkpoint-metrics.csv
- training/v4-400ep-seed300/mappo-vehicle-actions2-seed300/training.log
- training/v4-400ep-seed300/mappo-vehicle_pedestrian-actions2-seed300/checkpoint-metrics.csv
- training/v4-400ep-seed300/mappo-vehicle_pedestrian-actions2-seed300/training.log
- selected-checkpoints.json
- validation-trajectory.csv
- final-primary-results.csv
