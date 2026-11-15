# V2 findings

All comparisons below describe observed means. They are not significance tests.

- The primary benchmark still favors pedestrian-actuated control. The lowest
  learned vehicle time loss is MAPPO vehicle-only at 184.6 seconds per departed
  vehicle, versus 168.7 for actuated control (+9.4%). Every learned family also
  has higher mean pedestrian wait and worst-crossing p95 than actuated control.
- Low-demand IPPO vehicle-only has favorable point estimates: vehicle loss is
  49.0% lower, mean pedestrian wait 29.2% lower, and worst-crossing p95 56.3%
  lower than actuated control. The three-seed intervals are wide.
- MAPPO vehicle-only reduces mean vehicle loss relative to actuated control by
  1.3% under high demand and 11.2% for slow pedestrians. Its worst-crossing p95
  is higher by 7.4% and 26.4%, respectively, so these are vehicle/pedestrian
  tradeoffs rather than improvements on every objective.
- Against its own 60-episode pilot, primary IPPO vehicle-pedestrian improves
  vehicle loss by 4.5%, mean pedestrian wait by 7.8%, and worst-crossing p95 by
  16.4%. It still trails actuated control. Slow-scenario MAPPO vehicle-only
  improves vehicle loss by 25.0% against its pilot, but its p95 increases 8.9%.
- Only 2/12 selected runs met every crossing's validation constraint, versus
  1/12 in the pilot. The remaining ten used the predeclared lowest-violation
  fallback. The experiment does not demonstrate a consistent primary MARL
  advantage or reliable satisfaction of the pedestrian constraint.

V2 bundles normalization, switch penalty, demand randomization, and longer
training. The comparison cannot attribute effects to one change. Test seeds
11–13 were reused from the pilot that informed V2, so the results are held out
from V2 training/selection but are not independent confirmatory evidence.

All pods were terminated and an empty API pod list confirmed at 17:09 SGT on
14 November 2026. The conservative runtime-cost bound is $1.39 for V2 and
$1.71 including the earlier pilot; posted billing is delayed.
