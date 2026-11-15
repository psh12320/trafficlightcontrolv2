"""Build the complete V2 tables and a point-estimate comparison with the pilot."""
import argparse
import csv
import json
from pathlib import Path

if __package__:
    from .summarize_proposal_pilot import summarize
else:
    from summarize_proposal_pilot import summarize


def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(root:Path,pilot_root:Path):
    rows=summarize([root/name for name in ('final-primary-results.csv','final-demand-sweep.csv','final-slow-pedestrian-results.csv')])
    write_csv(root/'final-summary.csv',rows)
    with (pilot_root/'final-summary.csv').open(newline='',encoding='utf-8') as file:
        pilot={(row['scenario'],row['controller'],row['metric']):row for row in csv.DictReader(file)}
    comparison=[]
    for row in rows:
        old=pilot[(row['scenario'],row['controller'],row['metric'])]
        old_mean=float(old['mean'])
        comparison.append({'scenario':row['scenario'],'controller':row['controller'],'metric':row['metric'],
            'pilot_mean':old_mean,'v2_mean':row['mean'],'absolute_change':row['mean']-old_mean,
            'percent_change':100*(row['mean']/old_mean-1) if old_mean else '',
            'interpretation':'point estimate only; no confidence interval for this difference'})
    write_csv(root/'pilot-comparison.csv',comparison)
    actuated={(row['scenario'],row['metric']):row['mean'] for row in rows if row['controller']=='pedestrian_actuated'}
    baseline_comparison=[]
    for row in rows:
        baseline_mean=actuated[(row['scenario'],row['metric'])]
        baseline_comparison.append({'scenario':row['scenario'],'controller':row['controller'],'metric':row['metric'],
            'actuated_mean':baseline_mean,'controller_mean':row['mean'],
            'absolute_change':row['mean']-baseline_mean,
            'percent_change':100*(row['mean']/baseline_mean-1) if baseline_mean else '',
            'interpretation':'point estimate only; no confidence interval for this difference'})
    write_csv(root/'actuated-comparison.csv',baseline_comparison)
    selections=json.loads((root/'selected-checkpoints.json').read_text(encoding='utf-8'))
    feasible=sum(row['max_crossing_p95_violation']<=0 for row in selections)
    families=['fixed_time','pedestrian_actuated','ippo-vehicle','ippo-vehicle_pedestrian','mappo-vehicle','mappo-vehicle_pedestrian']
    metrics=['vehicle_mean_time_loss_seconds','vehicle_throughput_per_hour','vehicle_unfinished',
             'pedestrian_average_waiting_seconds','worst_crossing_p95_waiting_seconds']
    lines=['# V2 300-episode results','',
        'Twelve runs: IPPO/MAPPO × vehicle/vehicle-pedestrian rewards × training seeds 100/200/300. Each run trained from scratch for 300 episodes of 3,600 seconds.',
        'Vehicle waiting is divided by 20.8 and pedestrian waiting by 81.35 in the proposal objectives. The switch penalty is 1.5. Training demand samples vehicle periods 4/6/8 and pedestrian intervals 12/20/30 with a seeded private RNG and isolated network files.',
        'Snapshots at episodes 50/100/150/200/250/300 were evaluated on fixed validation seeds 21/22/23, primary demand, duration 3,600 seconds. Selections were frozen before V2 testing.',
        f'{feasible}/12 selected runs met every crossing\'s mean validation p95 limit. The other {12-feasible} used the predeclared lowest normalized positive violation fallback.',
        'Mean ± 95% Student-t CI (n=3, df=2). MARL: average three test seeds within each training seed, then form the CI across training seeds. Baseline: CI across the three test seeds. Intervals are descriptive and unadjusted for multiple comparisons.',
        'Test seeds 11/12/13 were excluded from V2 training and checkpoint selection, but reused from the pilot that informed the V2 changes. They are not a fresh independent confirmatory test set.',
        'Vehicle time loss is measured total loss divided by departed vehicles. Throughput is completed vehicles per simulated hour. Active unfinished counts exclude demand still waiting for insertion. Pedestrian means and percentiles include active unfinished pedestrians.',
        'Waiting metrics are observed to the simulation cutoff; waits for unfinished trips are right-censored.',
        'Worst crossing p95 is the maximum of the 12 per-crossing percentiles per evaluation, averaged across evaluations and training seeds. The summary CSV additionally retains total time loss, pedestrian unfinished counts, and maximum observed values.',
        'Primary: speed 1.4, vehicle period 6, pedestrian interval 20. Low: 1.4/10/30. High: 1.4/4/12. Slow: 0.9/6/20 with eight-second clearance. All tests last 3,600 seconds.',
        'The pilot comparison changes reward scaling, switching penalty, training duration, and demand randomization together. It compares configurations and cannot isolate the causal contribution of each change.','']
    for scenario in ('primary','low','high','slow'):
        lines += [f'## {scenario}','',
            '| Controller | Time loss (s/vehicle) | Throughput (veh/h) | Active unfinished | Mean pedestrian wait (s) | Worst crossing p95 (s) |',
            '|---|---:|---:|---:|---:|---:|']
        for family in families:
            values=[]
            for metric in metrics:
                value=next(row for row in rows if (row['scenario'],row['controller'],row['metric'])==(scenario,family,metric))
                values.append(f"{value['mean']:.2f} ± {value['ci95_half_width']:.2f}")
            lines.append('| '+' | '.join([family]+values)+' |')
        lines.append('')
    lines += ['## V2 versus archived pilot (point-estimate changes)','',
        'Negative changes favor V2 for time loss and waiting; positive changes favor V2 for throughput. These percentage changes are not significance tests.','',
        '| Scenario | Controller | Vehicle loss change | Throughput change | Pedestrian wait change | Worst crossing p95 change |',
        '|---|---|---:|---:|---:|---:|']
    for scenario in ('primary','low','high','slow'):
        for family in families:
            values=[]
            for metric in (metrics[0],metrics[1],metrics[3],metrics[4]):
                value=next(row for row in comparison if (row['scenario'],row['controller'],row['metric'])==(scenario,family,metric))
                values.append(f"{value['percent_change']:+.2f}%")
            lines.append('| '+' | '.join([scenario,family]+values)+' |')
    (root/'FINAL_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {'summary_rows':len(rows),'pilot_comparison_rows':len(comparison),
            'actuated_comparison_rows':len(baseline_comparison),'validation_feasible_runs':feasible}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--pilot-root',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(main(args.root,args.pilot_root)))
