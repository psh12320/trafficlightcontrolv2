"""Report selected V5 policies, BC quality, validation gain and prior comparisons."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import statistics

if __package__:
    from .summarize_proposal_pilot import summarize
    from .report_v3_results import read_csv, write_csv
else:
    from summarize_proposal_pilot import summarize
    from report_v3_results import read_csv, write_csv

SCENARIOS = ['primary', 'low', 'high', 'slow']
FAMILIES = ['fixed_time', 'pedestrian_actuated', 'mappo-vehicle', 'mappo-vehicle_pedestrian']
METRICS = ['vehicle_mean_time_loss_seconds', 'vehicle_throughput_per_hour',
           'vehicle_unfinished', 'pedestrian_average_waiting_seconds', 'worst_crossing_p95_waiting_seconds']


def main(root, v3, v4):
    paths = [root/n for n in ['final-primary-results.csv', 'final-demand-sweep.csv', 'final-slow-pedestrian-results.csv']]
    summary = summarize(paths)
    write_csv(root/'final-summary.csv', summary)
    raw = [r for p in paths for r in read_csv(p)]
    assert len(raw) == 96
    old = {'V3': read_csv(v3/'final-summary.csv'), 'V4': read_csv(v4/'final-summary.csv')}
    comparisons = []
    for version, rows in old.items():
        lookup = {(r['scenario'], r['controller'], r['metric']): float(r['mean']) for r in rows}
        for r in summary:
            before = lookup[(r['scenario'], r['controller'], r['metric'])]
            comparisons.append({'reference_version': version, 'scenario': r['scenario'], 'controller': r['controller'],
                'metric': r['metric'], 'reference_mean': before, 'v5_mean': r['mean'],
                'absolute_change': r['mean']-before, 'percent_change': 100*(r['mean']/before-1) if before else ''})
    write_csv(root/'prior-comparison.csv', comparisons)
    actuated = {(r['scenario'], r['metric']): r['mean'] for r in summary if r['controller'] == 'pedestrian_actuated'}
    write_csv(root/'actuated-comparison.csv', [{'scenario': r['scenario'], 'controller': r['controller'], 'metric': r['metric'],
        'actuated_mean': actuated[(r['scenario'], r['metric'])], 'v5_mean': r['mean'],
        'percent_change': 100*(r['mean']/actuated[(r['scenario'], r['metric'])]-1) if actuated[(r['scenario'], r['metric'])] else ''} for r in summary])
    groups = defaultdict(lambda: defaultdict(list))
    for r in raw:
        match = re.fullmatch(r'(mappo-(?:vehicle|vehicle_pedestrian))-actions2-seed(\d+)', r['scenario_controller'])
        family, unit = (match[1], int(match[2])) if match else (r['scenario_controller'], int(r['seed']))
        groups[(r['scenario'], family)][unit].append(r)
    crossing = []
    for (scenario, family), units in sorted(groups.items()):
        for j in range(3):
            for c, approach in enumerate(['North', 'East', 'South', 'West']):
                name = f':J{j}_c{c}'
                values = [statistics.mean(json.loads(r['pedestrian_p95_by_crossing'])[name] for r in rows) for rows in units.values()]
                crossing.append({'scenario': scenario, 'controller': family, 'crossing': name, 'approach': approach,
                    'mean_p95_seconds': statistics.mean(values),
                    'ci95_half_width': 4.302652729911275*statistics.stdev(values)/math.sqrt(3),
                    'maximum_observed_p95_seconds': max(json.loads(r['pedestrian_p95_by_crossing'])[name] for rows in units.values() for r in rows)})
    baseline = {(r['scenario'], r['crossing']): r['mean_p95_seconds'] for r in crossing if r['controller'] == 'pedestrian_actuated'}
    for r in crossing:
        r['excess_over_actuated_seconds'] = r['mean_p95_seconds']-baseline[(r['scenario'], r['crossing'])]
    write_csv(root/'per-crossing-p95.csv', crossing)
    previous_crossing = {(r['scenario'], r['controller'], r['crossing']): float(r['mean_p95_seconds']) for r in read_csv(v4/'per-crossing-p95.csv')}
    write_csv(root/'per-crossing-v4-comparison.csv', [{**r, 'v4_mean_p95_seconds': previous_crossing[(r['scenario'], r['controller'], r['crossing'])],
        'change_from_v4_seconds': r['mean_p95_seconds']-previous_crossing[(r['scenario'], r['controller'], r['crossing'])]} for r in crossing])

    selected = json.loads((root/'selected-checkpoints.json').read_text())
    quality = []
    gains = []
    for path in sorted(root.glob('training/v5-250ep-seed*/*/bc-metrics.json')):
        report = json.loads(path.read_text())
        fits = report['actor_fits']
        cm = report['bc_policy_rollout_confusion_by_agent']
        tp = sum(a[1][1] for a in cm)
        fn = sum(a[1][0] for a in cm)
        fp = sum(a[0][1] for a in cm)
        quality.append({'controller': path.parent.name,
            'demonstration_switch_labels': sum(a['expert_switches'] for a in fits),
            'training_switch_recall': sum(a['confusion_true_pred'][1][1] for a in fits)/sum(a['expert_switches'] for a in fits),
            'training_false_switches': sum(a['confusion_true_pred'][0][1] for a in fits),
            'rollout_expert_switches': tp+fn, 'rollout_policy_switches': tp+fp,
            'rollout_true_switches': tp, 'rollout_false_switches': fp,
            'rollout_switch_recall': tp/(tp+fn) if tp+fn else None,
            'rollout_switch_precision': tp/(tp+fp) if tp+fp else None,
            'rollout_legal_choice_agreement': report['bc_policy_rollout_legal_choice_expert_agreement'],
            'critic_mse_before': report['critic_fits'][0]['mse_before_warmup'],
            'critic_mse_after': report['critic_fits'][0]['mse_after_warmup']})
        chosen = next(r for r in selected if r['controller'] == path.parent.name)
        rows = read_csv(path.parent/'checkpoint-metrics.csv')
        pair = {}
        for tag, episode in [('bc', 0), ('selected', chosen['episode'])]:
            subset = [r for r in rows if int(r['training_episode']) == episode]
            pair[tag+'_vehicle_loss_s_per_vehicle'] = statistics.mean(float(r['vehicle_time_loss_seconds'])/float(r.get('vehicle_departed') or r['vehicle_arrivals']) for r in subset)
            pair[tag+'_pedestrian_mean_wait_s'] = statistics.mean(float(r['pedestrian_average_waiting_seconds']) for r in subset)
            pair[tag+'_worst_crossing_p95_s'] = statistics.mean(max(json.loads(r['pedestrian_p95_by_crossing']).values()) for r in subset)
        gains.append({'controller': path.parent.name, 'selected_episode': chosen['episode'], **pair,
            'vehicle_loss_change_percent': 100*(pair['selected_vehicle_loss_s_per_vehicle']/pair['bc_vehicle_loss_s_per_vehicle']-1)})
    write_csv(root/'bc-quality.csv', quality)
    write_csv(root/'bc-vs-selected-validation.csv', gains)
    bc_reference = read_csv(root/'bc-primary-results.csv')
    bc_summary = summarize([root/'bc-primary-results.csv'])
    write_csv(root/'bc-primary-summary.csv', bc_summary)
    def values(row):
        return [float(row['vehicle_time_loss_seconds'])/float(row.get('vehicle_departed') or row['vehicle_arrivals']),
            float(row['vehicle_completed'])*3600/float(row['duration_seconds']), float(row['vehicle_unfinished']),
            float(row['pedestrian_average_waiting_seconds']), max(json.loads(row['pedestrian_p95_by_crossing']).values())]
    primary_gain = []
    for family in FAMILIES[2:]:
        for index, metric in enumerate(METRICS):
            bc_values, selected_values = [], []
            for seed in [100,200,300]:
                controller = f'{family}-actions2-seed{seed}'
                initial = [r for r in bc_reference if r['scenario_controller'] == controller]
                final = [r for r in raw if r['scenario'] == 'primary' and r['scenario_controller'] == controller]
                assert len(initial) == len(final) == 3
                bc_values.append(statistics.mean(values(r)[index] for r in initial))
                selected_values.append(statistics.mean(values(r)[index] for r in final))
            differences = [b-a for a,b in zip(bc_values,selected_values)]
            half = 4.302652729911275*statistics.stdev(differences)/math.sqrt(3)
            primary_gain.append({'controller':family, 'metric':metric, 'bc_mean':statistics.mean(bc_values),
                'selected_mean':statistics.mean(selected_values), 'paired_change_mean':statistics.mean(differences),
                'paired_change_ci95_half_width':half, 'paired_change_ci95_low':statistics.mean(differences)-half,
                'paired_change_ci95_high':statistics.mean(differences)+half})
    write_csv(root/'bc-vs-selected-primary.csv',primary_gain)
    feasible = sum(r['max_crossing_p95_violation'] <= 0 for r in selected)
    lines = ['# V5 BC warm-started MAPPO', '',
        'Six accepted MAPPO runs: vehicle/combined x training seeds 100/200/300. Each run uses 20 actuated-expert episodes, 15 masked-cross-entropy epochs with Adam 1e-3, five critic-only rollout episodes, then 250 PPO episodes with learning rate 1e-4. All episodes last 3600s under fixed primary demand.', '',
        'Three consecutive 25-feature frames give 75 actor inputs and 225 centralized critic inputs. Reset repeats the initial frame three times; temporal order is oldest to newest. Switch cost 3, vehicle/pedestrian scales 20.8/20, combined urgency 0.5*sum((current max wait/60)^2), two actions and minimum/maximum green 10/45.', '',
        'The first uniform-minibatch BC attempt collapsed to hold because switches were rare. It is retained only under diagnostics/unbalanced-bc and excluded from all candidates and accepted results. Corrected BC balances legal action classes in each minibatch while retaining the requested loss, learning rate and 15 epochs. The critic initializes its value bias to empirical mean returns and fits discounted returns with MSE. Actor weights remain unchanged during critic fitting.', '',
        'The expert preserves the recorded actuated request rule, including holding when both phases have requests, at legal five-second MAPPO decisions. The recorded actuated baseline checks every second. Matching the baseline is measured, not guaranteed. Class-specific imitation statistics appear below because high hold accuracy can conceal poor switch recall.', '',
        f'Selections froze using validation seeds 21/22/23 and candidates BC episode 0 plus RL episodes 50/100/150/200/250. {feasible}/6 satisfy every crossing limit; infeasible runs use the declared normalized-positive-violation fallback. {sum(r["episode"] == 0 for r in selected)}/6 selected the BC checkpoint. Feasible candidates minimize mean total vehicle time loss; the fallback minimizes normalized violation, then total vehicle time loss.', '',
        '72 new selected-policy evaluations plus 24 exactly reused V3/V4 baseline measurements produce 96 accepted rows, 24 per scenario. A separate matched primary BC reference has 18 rows, reusing selected BC measurements where applicable and evaluating only the remaining BC policies after selection froze. Test seeds 41/42/43 were reused from prior experiments for a matched exploratory comparison, not untouched confirmatory testing.', '',
        'Mean +/- 95% Student-t CI, n=3 and df=2. Learned-policy test seeds are averaged within each training seed first; baselines use traffic seeds directly. Zero learned intervals do not mean zero traffic variability. Comparisons are descriptive and unadjusted. Multiple settings differ between versions, so no individual causal effect is isolated.', '',
        'Vehicle loss divides total observed time loss by departed vehicles. Throughput counts completions. Unfinished vehicles exclude demand waiting for insertion. Pedestrian waits include active trips and are right-censored at 3600s; crossing percentiles retain the prior route-assigned whole-trip definition. Worst-crossing p95 takes the maximum of 12 crossing percentiles within each evaluation, then averages. The per-crossing table averages each crossing separately.', '']
    comparison_table = []
    for scenario in SCENARIOS:
        lines += [f'## {scenario}', '', '| Controller | Loss s/veh | Throughput veh/h | Active unfinished vehicles | Pedestrian wait s | Worst crossing p95 s |', '|---|---:|---:|---:|---:|---:|']
        versions = [('Baseline', 'fixed_time'), ('Baseline', 'pedestrian_actuated')]+[(version, family) for version in ['V3','V4','V5'] for family in FAMILIES[2:]]
        for version, family in versions:
            source = old[version] if version in old else summary
            values = []
            for metric in METRICS:
                r = next(r for r in source if (r['scenario'],r['controller'],r['metric']) == (scenario,family,metric))
                mean, half = float(r['mean']), float(r['ci95_half_width'])
                values.append(f'{mean:.2f} +/- {half:.2f}')
                comparison_table.append({'scenario': scenario, 'version': version, 'controller': family,
                                         'metric': metric, 'mean': mean, 'ci95_half_width': half})
            lines.append('| '+' | '.join([version+' '+family]+values)+' |')
        lines.append('')
    write_csv(root/'comparison-summary.csv', comparison_table)
    lines += ['## Frozen selections', '', '| Controller | Episode | Largest validation crossing excess s | Normalized violation |', '|---|---:|---:|---:|']
    for r in selected:
        lines.append(f"| {r['controller']} | {r['episode']} | {r['max_crossing_p95_violation']:.3f} | {r['normalized_violation']:.6f} |")
    lines += ['', '## BC versus selected validation', '', '| Controller | Selected episode | BC loss s/veh | Selected loss s/veh | Change % | BC ped wait s | Selected ped wait s |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in gains:
        lines.append(f"| {r['controller']} | {r['selected_episode']} | {r['bc_vehicle_loss_s_per_vehicle']:.2f} | {r['selected_vehicle_loss_s_per_vehicle']:.2f} | {r['vehicle_loss_change_percent']:.2f} | {r['bc_pedestrian_mean_wait_s']:.2f} | {r['selected_pedestrian_mean_wait_s']:.2f} |")
    lines += ['', 'Validation gains above use the same data used for selection. The following primary comparison uses matched test traffic after frozen selection. The paired CI averages traffic seeds within each training seed, then uses the three training-seed differences.', '',
        '## BC versus selected primary tests', '', '| Controller | Metric | BC mean | Selected mean | Selected minus BC, 95% CI |', '|---|---|---:|---:|---|']
    for r in primary_gain:
        if r['metric'] in [METRICS[0],METRICS[3],METRICS[4]]:
            lines.append(f"| {r['controller']} | {r['metric']} | {r['bc_mean']:.2f} | {r['selected_mean']:.2f} | {r['paired_change_mean']:.2f} +/- {r['paired_change_ci95_half_width']:.2f} |")
    lines += ['', '## Imitation quality', '', 'Training recall is measured on demonstrations. Rollout recall and precision use the five different warm-up simulation seeds and the expert labels on the BC policy states. Few positive labels make these estimates unstable. High overall agreement alone is insufficient.', '', '| Controller | Demo switch labels | Training switch recall | Training false switches | Rollout expert switches | Rollout true switches | Rollout false switches |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in quality:
        lines.append(f"| {r['controller']} | {r['demonstration_switch_labels']} | {r['training_switch_recall']:.3f} | {r['training_false_switches']} | {r['rollout_expert_switches']} | {r['rollout_true_switches']} | {r['rollout_false_switches']} |")
    (root/'FINAL_RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'summary_rows':len(summary), 'crossing_rows':len(crossing), 'comparison_rows':len(comparison_table),
        'bc_checkpoints_selected':sum(r['episode']==0 for r in selected), 'validation_feasible_runs':feasible, 'accepted_rows':len(raw)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('scenario/generated/final-v5-results'))
    p.add_argument('--v3-root', type=Path, default=Path('scenario/generated/final-v3-results'))
    p.add_argument('--v4-root', type=Path, default=Path('scenario/generated/final-v4-results'))
    args = p.parse_args()
    main(args.root, args.v3_root, args.v4_root)
