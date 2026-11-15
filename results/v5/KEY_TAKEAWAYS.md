# V5 findings

V5 completed, but does not establish a consistent advantage over actuated control
or a reliable primary-scenario gain from PPO after behavior cloning.

- Primary, vehicle versus actuated: vehicle loss +2.02%, mean pedestrian wait
  +8.46%, worst-crossing p95 +19.81%. Combined: +2.56%, +11.20%, +26.20%.
- Low demand is a partial success for vehicle-only: its point means improve over
  actuated for loss, pedestrian mean and crossing p95. Fixed time still has lower
  vehicle loss. Combined has substantially worse low-demand tails than vehicle-only.
- High demand: both objectives have identical reported aggregate metrics. Vehicle
  loss improves over V4 by 15.60%/13.67% for vehicle/combined, but remains 13.12%
  above actuated. Slow-pedestrian results are mixed and do not establish dominance.
- None of six selections meets every validation crossing threshold. For training
  seeds 100/200/300, selected episodes are vehicle 250/0/100 and combined 200/0/100.
  Episode zero is BC plus critic warm-up. All choices use the declared normalized
  violation fallback, frozen before test evaluation.

The matched primary BC reference directly checks fine-tuning gain. Selected minus
BC means and 95% paired training-seed CIs are:

| Objective | Vehicle loss s/veh | Pedestrian wait s | Worst crossing p95 s |
|---|---:|---:|---:|
| Vehicle | -0.06 +/- 17.89 | +14.80 +/- 56.83 | +154.63 +/- 614.15 |
| Combined | +0.90 +/- 14.44 | +20.05 +/- 51.16 | +213.38 +/- 546.94 |

All intervals span zero. These results do not demonstrate a primary fine-tuning
benefit. The BC reference itself averages 181.69s vehicle loss, 192.88s pedestrian
wait and 947.24s worst-crossing p95. Its selected PPO successors do not show a
consistent improvement on those held-out traffic cases.

The expert supplied only 20/27/33 switch labels across 43,200 actor decisions per
training seed, per reward objective. Uniform BC initially learned only hold. That
attempt was stopped and retained solely as an excluded diagnostic. Accepted BC
uses balanced legal-action minibatches, the same masked CE, 15 epochs and Adam
1e-3. It recalls every demonstration switch, but generalizes imperfectly: for one
objective across the three initializations, warm-up rollouts match only 3 of 10
expert switch opportunities and produce 58 false switches. The corresponding BC
results repeat for the other objective. High overall hold accuracy is misleading.
The teacher also acts at legal five-second decisions while the recorded baseline
checks every second. Exact imitation was not achieved.

Six accepted runs completed 120 BC episodes, 30 critic warm-up episodes and 1,500
RL episodes. There are 36 snapshots including six BC checkpoints, 108 validation
records, six selections and 96 main test records. A separate 18-row BC primary
reference required 12 additional evaluations and reused six selected-BC rows.
Do not treat reused rows or traffic cases as independent training replicates.

The 95% intervals use Student-t with three training-seed averages for learned
policies and three traffic seeds for baselines. Paired BC gains use three
training-seed differences after averaging traffic cases. Seeds 41-43 were reused
from earlier versions, so comparisons are exploratory. V5 changes initialization,
frame history, learning rate and training length together; cross-version results
cannot isolate any individual causal contribution.

All 31 local tests and corrected cloud smoke tests passed. The audit verified 282
transfer-manifest entries, four archive SHA256s, unchanged frozen selections,
exact baseline reuse and all 4,841 protected V1-V4 artifacts unchanged. Every pod
is deleted (last: 22:49:15 SGT on 14 November, before 00:30 on 15 November), and the
temporary volume is deleted. Conservative V5 runtime/storage bound is $1.22,
including setup, discarded diagnostics, slow-host time and transfers. Posted
billing may lag. Keep V5 as the final focused ablation; the data do not support
presenting it as a consistent improvement over actuated control.
