"""Create V4 tables, crossing tails and matched V3 point-estimate comparisons."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re
import statistics

if __package__:
    from .summarize_proposal_pilot import summarize
    from .report_v3_results import read_csv,write_csv
else:
    from summarize_proposal_pilot import summarize
    from report_v3_results import read_csv,write_csv

FAMILIES=['fixed_time','pedestrian_actuated','mappo-vehicle','mappo-vehicle_pedestrian']
SCENARIOS=['primary','low','high','slow']


def main(root,v3):
    paths=[root/n for n in ['final-primary-results.csv','final-demand-sweep.csv','final-slow-pedestrian-results.csv']]
    summary=summarize(paths);write_csv(root/'final-summary.csv',summary)
    raw=[r for p in paths for r in read_csv(p)];assert len(raw)==96
    old=read_csv(v3/'final-summary.csv');old_lookup={(r['scenario'],r['controller'],r['metric']):r for r in old}
    actuated={(r['scenario'],r['metric']):r['mean'] for r in summary if r['controller']=='pedestrian_actuated'}
    comparison=[];baseline_comparison=[]
    for r in summary:
        key=(r['scenario'],r['controller'],r['metric']);b=float(old_lookup[key]['mean']);a=actuated[(key[0],key[2])]
        comparison.append({'scenario':key[0],'controller':key[1],'metric':key[2],'v3_mean':b,'v4_mean':r['mean'],
            'absolute_change':r['mean']-b,'percent_change':100*(r['mean']/b-1) if b else '',
            'interpretation':'matched traffic seeds; point estimate; multiple simultaneous training/design changes'})
        baseline_comparison.append({'scenario':key[0],'controller':key[1],'metric':key[2],'actuated_mean':a,
            'controller_mean':r['mean'],'percent_change':100*(r['mean']/a-1) if a else ''})
    write_csv(root/'v3-comparison.csv',comparison);write_csv(root/'actuated-comparison.csv',baseline_comparison)
    groups=defaultdict(lambda:defaultdict(list))
    for r in raw:
        match=re.fullmatch(r'(mappo-(?:vehicle|vehicle_pedestrian))-actions2-seed(\d+)',r['scenario_controller'])
        family,unit=(match[1],int(match[2])) if match else (r['scenario_controller'],int(r['seed']))
        groups[(r['scenario'],family)][unit].append(r)
    crossing=[]
    for (scenario,family),units in sorted(groups.items()):
        for j in range(3):
            for c,approach in enumerate(['North','East','South','West']):
                name=f':J{j}_c{c}'
                samples=[statistics.mean(json.loads(r['pedestrian_p95_by_crossing'])[name] for r in rows) for rows in units.values()]
                crossing.append({'scenario':scenario,'controller':family,'crossing':name,'approach':approach,
                    'mean_p95_seconds':statistics.mean(samples),'ci95_half_width':4.302652729911275*statistics.stdev(samples)/math.sqrt(3),
                    'maximum_observed_p95_seconds':max(json.loads(r['pedestrian_p95_by_crossing'])[name] for rows in units.values() for r in rows)})
    reference={(r['scenario'],r['crossing']):r['mean_p95_seconds'] for r in crossing if r['controller']=='pedestrian_actuated'}
    for r in crossing:r['excess_over_actuated_seconds']=r['mean_p95_seconds']-reference[(r['scenario'],r['crossing'])]
    write_csv(root/'per-crossing-p95.csv',crossing)
    old_cross={(r['scenario'],r['controller'],r['crossing']):r
               for r in read_csv(v3/'per-crossing-p95.csv')}
    crossing_comparison=[]
    for r in crossing:
        before=float(old_cross[(r['scenario'],r['controller'],r['crossing'])]['mean_p95_seconds'])
        crossing_comparison.append({'scenario':r['scenario'],'controller':r['controller'],
            'crossing':r['crossing'],'approach':r['approach'],'v3_mean_p95_seconds':before,
            'v4_mean_p95_seconds':r['mean_p95_seconds'],
            'absolute_change_seconds':r['mean_p95_seconds']-before,
            'percent_change':100*(r['mean_p95_seconds']/before-1) if before else ''})
    write_csv(root/'per-crossing-v3-comparison.csv',crossing_comparison)
    selected=json.loads((root/'selected-checkpoints.json').read_text());feasible=sum(r['max_crossing_p95_violation']<=0 for r in selected)
    lines=['# V4 Upstream-Lookahead & Calibrated-Switching MAPPO','',
        'Six MAPPO runs: vehicle/combined rewards x training seeds 100/200/300, 400 episodes x3600 seconds from scratch.',
        'Training uses fixed primary demand 1.4/6/20, not demand randomization. Observations have 25 features, including four total approach vehicle counts /25 in addition to four stopped queues. Central critic input is 75.',
        'Switch cost 3; vehicle/pedestrian scales 20.8/20; combined urgency 0.5*sum((current crossing max wait/60)^2), once per five-second step. Two actions, minimum/maximum green 10/45; network geometry and signal programs unchanged.',
        'Eight snapshots per run (50..400) were selected only on validation seeds 21/22/23. Test seeds 41/42/43 are reused from V3 for a matched exploratory comparison, not untouched confirmatory testing.',
        f'{feasible}/6 selections meet every validation crossing limit; others use the predeclared normalized-positive-violation fallback.',
        '72 new learned-policy evaluations plus 24 exactly reused V3 baseline rows, 24 per scenario. Baseline code/network hashes and every reused record are verified.',
        'Mean +/- 95% Student-t CI (n=3, df=2). Learned policies average tests within training seed, then form the interval across training seeds. Baseline intervals use traffic test seeds. A zero learned interval does not imply no traffic-seed variability.',
        'Vehicle loss is total observed loss / departed vehicles; throughput counts completions. Unfinished vehicles exclude demand awaiting insertion. Pedestrian waits include active trips and are right-censored at3600s. Crossing p95 retains the prior route-assigned whole-trip waiting definition.',
        'V4 versus V3 changes observation inputs, switch cost, training demand regime and training length together; it cannot isolate one causal contribution. Percent changes are descriptive, not significance tests.','']
    metrics=['vehicle_mean_time_loss_seconds','vehicle_throughput_per_hour','vehicle_unfinished','pedestrian_average_waiting_seconds','worst_crossing_p95_waiting_seconds']
    table=[]
    for scenario in SCENARIOS:
        lines += [f'## {scenario}','','| Controller | Loss (s/veh) | Throughput (veh/h) | Active unfinished vehicles | Pedestrian wait (s) | Worst crossing p95 (s) |','|---|---:|---:|---:|---:|---:|']
        versions=[('Baseline','fixed_time'),('Baseline','pedestrian_actuated'),('V3','mappo-vehicle'),('V3','mappo-vehicle_pedestrian'),('V4','mappo-vehicle'),('V4','mappo-vehicle_pedestrian')]
        for version,family in versions:
            values=[]
            for metric in metrics:
                r=next(x for x in (old if version=='V3' else summary) if (x['scenario'],x['controller'],x['metric'])==(scenario,family,metric))
                values.append(f"{float(r['mean']):.2f} +/- {float(r['ci95_half_width']):.2f}")
                table.append({'scenario':scenario,'version':version,'controller':family,'metric':metric,'mean':float(r['mean']),'ci95_half_width':float(r['ci95_half_width'])})
            lines.append('| '+' | '.join([version+' '+family]+values)+' |')
        lines.append('')
    write_csv(root/'comparison-summary.csv',table)
    lines += ['## Frozen selections','','| Controller | Episode | Validation feasible | Largest crossing excess (s) | Normalized positive violation |','|---|---:|---|---:|---:|']
    for r in selected:lines.append(f"| {r['controller']} | {r['episode']} | {r['max_crossing_p95_violation']<=0} | {r['max_crossing_p95_violation']:.3f} | {r['normalized_violation']:.6f} |")
    (root/'FINAL_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    result={'summary_rows':len(summary),'crossing_rows':len(crossing),'comparison_rows':len(table),'accepted_rows':len(raw),'validation_feasible_runs':feasible}
    print(json.dumps(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--v3-root',type=Path,required=True);a=p.parse_args();main(a.root,a.v3_root)
