# V4 findings

The focused V4 experiment is complete. It does not establish a consistent
advantage over pedestrian-actuated control or V3. Keep it as an upstream-density
and switching-calibration experiment in the report; do not present it as the
best-performing configuration. No further cloud training was launched.

- Primary, V4 vehicle versus actuated: vehicle loss +0.06%, mean pedestrian
  wait -3.14%, worst-crossing p95 +3.77%. Versus V3 MAPPO vehicle: vehicle loss
  +2.56%, mean pedestrian wait -2.08%, worst-crossing p95 -2.10%.
- Primary, V4 combined versus actuated: vehicle loss +26.75%, pedestrian wait
  +7.25%, worst-crossing p95 +16.42%. Versus V3 MAPPO combined, the corresponding
  increases are +1.70%, +6.92%, and +14.46%.
- High demand, V4 vehicle/combined vehicle loss improves over V3 by 9.84%/13.29%,
  but remains 34.03%/31.03% above actuated control. This is partial improvement
  within the MARL configurations, with no robust baseline win.
- Low demand, vehicle loss worsens versus V3 by 99.57%/27.52% for vehicle/combined.
  Slow pedestrians produce mixed changes versus V3; neither V4 objective beats
  actuated vehicle loss.
- No selected run satisfies every validation crossing limit. The closest miss
  is 0.067 seconds; selection used the declared normalized-positive-violation
  fallback without changing its tolerance. Episodes selected for training seeds
  100/200/300 are vehicle: 200/50/100, combined: 150/400/400.
- The largest primary crossing excess over actuated is at J2 East for vehicle
  (+72.82 s), and J1 South/North for combined (+198.41/+192.10 s). These are
  crossing-specific mean p95s; the main table separately averages each run's
  maximum crossing p95.

The experiment changes observations, switch penalty, demand regime, and training
length together. It cannot attribute a causal effect to density lookahead or
establish switch cost 3 as optimal. Test seeds 41-43 were reused from V3, so this
is a matched exploratory comparison. Selections used only validation seeds
21-23 and stayed frozen throughout testing. Three training seeds give wide
descriptive Student-t intervals. Zero V3 vehicle intervals arise from identical
training-seed averages, not absence of traffic-seed variability.

Training-only diagnostics are saved as validation-trajectory.csv,
validation-trajectory-by-run.csv, and reward-rolling25-by-run.csv. They do not
alter checkpoint selection. The combined validation trajectory is non-monotonic;
two of its three choices are at episode 400, while the vehicle choices are early.
These observations alone do not justify a large continuation of the same setup.

All 2,400 training episodes, 48 snapshots, 144 validation evaluations, six selected
checkpoints, and 96 final records are present locally. The final set consists of
72 new policy evaluations and 24 exactly reused baseline measurements. The audit
verified 214 transfer-manifest entries and all 4,524 protected prior artifacts.
V1, V2, and V3 outputs and archives are unchanged. All three pods and the temporary
checkpoint volume are deleted.
