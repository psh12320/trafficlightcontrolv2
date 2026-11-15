# Report figures

Five PNG figures rendered locally at **300 DPI**. No RunPod resources were used.
The original experiment results are read-only inputs; their SHA-256 hashes are
checked against `_support/prior_artifact_manifest.json` after rendering.
Supporting captures, extracted source snapshots, and working files under
`_support/` are kept locally and gitignored.

| Figure | File | Pixels | Size at 300 DPI |
| --- | --- | --- | --- |
| 1 | `fig1_sumo_corridor_overview.png` | 4350 × 1680 | 14.5 × 5.6 in |
| 2 | `fig2_v1_to_v5_progression.png` | 4800 × 1350 | 16.0 × 4.5 in |
| 3 | `fig3_scenario_comparison.png` | 4500 × 3150 | 15.0 × 10.5 in |
| 4 | `fig4_per_crossing_p95.png` | 3540 × 2400 | 11.8 × 8.0 in |
| 5 | `fig5_ablation_progression.png` | 4200 × 1950 | 14.0 × 6.5 in |

## Captions and interpretation

**Figure 1. SUMO corridor and pedestrian infrastructure.** Local SUMO GUI views
show the three-junction corridor at t = 120 s and junction J1 at t = 121 s under
Primary demand, traffic seed 41. Junction spacing is 200 m. The detail identifies
the four crosswalks and outlines internal walking areas using the network's
actual polygons. Crosswalk indices c0/c1/c2/c3 denote north/east/south/west.
Vehicles are blue and pedestrians orange; pedestrian symbols are enlarged by
1.6 for visibility. The dashed box marks the J1 detail's 100 m × 100 m extent.

**Figure 2. Progressive controller designs, with local seed-41 illustrations.**
(a) V2 MAPPO-VehPed, selected episode 250, shows J0 in genuine yellow clearance
at t = 117 s, with 14 stopped pedestrians on internal walking areas. This V2
frame illustrates the observation limitation shared with the earlier design;
it is not a separate V1 replay. (b) V3 IPPO-VehPed, episode 150, shows J2 at
t = 195 s, with pedestrians on all four crossing edges and an active crossing
pedestrian. (c) V4 MAPPO-Veh, episode 200, shows the corridor at t = 81 and 91 s;
magenta colors and boxes identify the same moving platoon as it passes J2.
(d) V5 MAPPO-Veh, episode 250 after BC initialization and PPO, shows Low demand
at t = 123 s: the stopped queue on lane `N1_J1_1` falls from five at t = 118 s
to zero five seconds later. All pictured policies use training seed 100; all
traffic replays use seed 41. Primary demand is used in (a–c), Low in (d).
The images illustrate behavior, while the metric annotations summarize the
separate saved diagnostics or aggregate results. They do not isolate causal
effects of individual design changes.

The annotation scopes are deliberately explicit:

- **50.1%** is V2 MAPPO-Veh's fraction of crossing observation slots with zero
  observed count conditional on pedestrians waiting outside a crossing, from
  the prior seed-11 diagnostic: 4274/8536. It is not a fraction of all simulation
  seconds, nor the V2 combined policy's corresponding fraction.
- **384** counts all J0 phase transitions for V2 MAPPO-VehPed in that seed-11
  diagnostic, including **128 green terminations**. **35.6%** is time in
  phases 1/2/4/5. Some of these phases still permit vehicle green, so it is
  labelled clearance time, **not vehicle dead time or all-red time**. The
  genuinely yellow phases occupy 17.8% of that diagnostic rollout. No phases
  were changed for the illustration.
- **24.1%** is the relative reduction in the mean of the four J2 crossing p95s
  for V3 IPPO-VehPed against Pedestrian-Actuated, averaged over the saved three
  training seeds and test seeds 41–43. It is not a seed-41-only effect. V3
  MAPPO-Veh separately matches fixed-time metrics in the saved evaluation set
  with switch penalty 8.0; it is not the IPPO policy pictured in (b).
- **178.1 s** is V4 MAPPO-Veh's Primary aggregate loss. **2.9×** compares the
  traffic-seed CI half-widths: actuated 63.63 s / V4 21.75 s. For V4, each
  traffic-seed value averages the three learned policies. This smoothing and
  the three-seed sample preclude a claim of a proven population variance
  advantage. It does not compare the unlike 10.94 s training-seed CI.
- **51.4 s loss / 207.3 s worst-crossing p95** are V5 MAPPO-Veh's Low-demand
  aggregate test means, not metrics measured from the five-second image sequence.

Figure 2 is visually proofed at a 6.5-inch text-block width in a report using
12-point body text. At this placement the main figure annotations are 8.33 pt;
the image remains a compact 16:4.5 composite. The old Figure 2 is retained under
`_support/superseded/fig2_sumo_failure_modes.png`.

**Figure 3. V3 performance across demand and walking-speed scenarios.** Each
scenario panel contains paired plots of vehicle time loss and worst-crossing
p95 pedestrian waiting time for the six controller families. Points show means
and whiskers show 95% Student-t confidence intervals. Primary/Low/High use
(pedestrian speed, vehicle period, pedestrian interval) of (1.4, 6, 20),
(1.4, 10, 30), and (1.4, 4, 12); Slow uses (0.9, 6, 20). Evaluations last
3600 s and use test seeds 41–43. Panel scales differ. Confidence intervals are
not clipped at zero, so a lower bound can be negative despite a nonnegative
metric.

**Figure 4. Spatial pattern of V3 Primary pedestrian waiting.** Cells show
averaged per-crossing p95 waiting times, rather than percentiles of pooled
pedestrian observations. J2 is outlined in blue. Relative to Pedestrian-Actuated,
IPPO-VehPed reduces the mean of the four J2 crossing values from 835.4 s to
634.2 s, a **24.1% reduction**. The individual J2 reductions are 28.0%, 22.3%,
25.8%, and 19.7% for c0–c3. J1 increases are displayed as well. The local J2
improvement does not imply a lower corridor-wide worst-crossing metric.

**Figure 5. Validation evolution and configuration progression.** Left: V4
validation means at episodes 50–400 for both MAPPO reward modes, showing vehicle
time loss and normalized positive crossing violation. Right: Primary vehicle
time loss across V2–V5 with the corresponding version's Fixed-Time and
Pedestrian-Actuated records. V2 uses test seeds 11–13; V3–V5 use 41–43, separated
by a dotted line. Multiple observation, reward, action, initialization, and
training choices change together across versions. The comparison is descriptive
and does not isolate a causal effect of any one change.

## Statistical conventions

- Learned controllers: average the three traffic test seeds within each
  training seed (100, 200, 300), then calculate the mean and CI across those
  three training-seed averages. These intervals quantify training-seed
  variation conditional on the shared traffic set.
- Baselines: calculate the mean and CI directly across the three traffic seeds.
  The two uncertainty units differ and should not be compared as if they
  represented the same source of variation.
- Two-sided 95% CI: mean ± `4.302652729911275 × sample_SD / sqrt(3)` (df = 2).
  The intervals are descriptive and unadjusted for multiple comparisons. A zero
  half-width for V3 MAPPO-Veh reflects identical training-seed averages in the
  saved records, not absence of traffic variability.
- Vehicle time loss is the saved total time loss divided by departed vehicles.
  Worst-crossing p95 is computed per rollout before the saved aggregation;
  Figure 4 instead shows the mean p95 of each named crossing.
- V4 crossing violation is calculated separately for each training run:
  `sum_c max(mean_validation_p95_c - actuated_threshold_c, 0) /
  max(actuated_threshold_c, 1)`, then averaged over the three training seeds.
  Each threshold is the Pedestrian-Actuated crossing mean on validation seeds
  21–23. This is a sum of normalized positive excesses, not a percentage or a
  count of failed crossings.
- Seeds 41–43 were fresh for the V3 evaluation, then reused for V4 and V5.
  Later comparisons therefore are not independent fresh confirmatory tests.

## Sources and reproduction

All paths below are relative to the repository root:

- Figure 3: `scenario/generated/final-v3-results/final-summary.csv`.
- Figure 2: each V2–V5 selected checkpoint with its corresponding deployed
  source ZIP, plus `scenario/generated/v2-local-diagnostics/diagnostics.json`, V3 crossing means,
  V4 `convergence-diagnostics/primary-uncertainty-by-unit.csv`, and V5 summary.
  Detailed capture records and claim values are in `_support/progression/`.
- Figure 4: `scenario/generated/final-v3-results/per-crossing-p95.csv`.
- Figure 5 left:
  `scenario/generated/final-v4-results/convergence-diagnostics/validation-trajectory.csv`.
- Figure 5 right: `scenario/generated/final-v{2,3,4,5}-results/final-summary.csv`.
- GUI replay source: `scenario/generated/final-v2-results/reproducibility-source.zip`.
- GUI replay checkpoint:
  `scenario/generated/final-v2-results/checkpoints/mappo-vehicle-actions4-seed100/episode-50.pt`;
  SHA-256 `b5297ac97ecb6fa2625ba0c38746f9bc9f9fde4475643b68c8fcdbc7dee27738`.

Figure 1 uses deterministic actions from the archived V2 MAPPO-Veh controller,
training seed 100, selected episode 50. It is explicitly labeled a V2 replay
illustration and is not claimed to be a V3 evaluation frame. SUMO 1.28.0 runs
locally through TraCI. The archived code is extracted under `_support/v2-replay/`
and builds its network only under `_support/capture-network/`. Rendering uses
white backgrounds, a north-up camera, and changed display colors; simulation
parameters and controller actions are preserved.

Raw GUI screenshots, per-second queue traces, camera bounds, actual signal
states, and network geometry are retained under `_support/`. Exact input and
script hashes are in `_support/figure_provenance.json`. Final PNG dimensions,
DPI metadata, hashes, and archive-preservation checks are recorded in
`figure_verification.json`.

Install the figure-rendering dependencies alongside the repository requirements.

```powershell
python -m pip install matplotlib Pillow
```

Generate the captures and figures with the saved experiment artifacts available
under `scenario/generated/`.

```powershell
python -X utf8 scripts/capture_report_sumo.py
foreach ($version in 2..5) {
    python -X utf8 scripts/capture_progression_sumo.py --version $version
}
python -X utf8 scripts/generate_report_figures.py
```

Rendering from the saved raw captures does not require another simulation.
Use `--statistics-only` or `--screenshots-only` to regenerate a subset of plots.
To render only the replacement Figure 2 from its saved captures, run
`python -X utf8 scripts/compose_progression_figure.py`.
