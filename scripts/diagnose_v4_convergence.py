"""Read frozen V4 artifacts and report convergence and uncertainty by seed."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import zipfile


T95_DF2 = 4.302652729911275
SEEDS = [100, 200, 300]
SNAPSHOTS = list(range(50, 401, 50))
FAMILIES = ['mappo-vehicle', 'mappo-vehicle_pedestrian']
VALIDATION_KEYS = ['vehicle_time_loss_s_per_vehicle', 'pedestrian_mean_wait_s',
                   'worst_crossing_p95_s', 'normalized_positive_crossing_violation',
                   'largest_crossing_excess_s']


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ci95(values):
    assert len(values) == 3
    mean = statistics.mean(values)
    half = T95_DF2 * statistics.stdev(values) / math.sqrt(3)
    return {'mean': mean, 'ci95_half_width': half, 'ci95_low': mean-half,
            'ci95_high': mean+half}


def vehicle_loss(row):
    # Baseline rows use vehicle_arrivals for the departed-vehicle count.
    count = float(row.get('vehicle_departed') or row['vehicle_arrivals'])
    assert count > 0
    return float(row['vehicle_time_loss_seconds']) / count


def slope(pairs):
    result = statistics.linear_regression([e for e, _ in pairs], [r for _, r in pairs]).slope
    xm = statistics.mean(e for e, _ in pairs)
    ym = statistics.mean(r for _, r in pairs)
    independent = sum((e-xm)*(r-ym) for e, r in pairs) / sum((e-xm)**2 for e, _ in pairs)
    assert math.isclose(result, independent, rel_tol=1e-12, abs_tol=1e-12)
    return result


def main(root, output):
    manifest = json.loads((root/'LOCAL_ARTIFACT_MANIFEST.json').read_text())
    for relative, digest in manifest.items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest() == digest, relative
    inputs = {}

    def source(path):
        inputs[str(path.relative_to(root)).replace('\\', '/')] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    with zipfile.ZipFile(source(root/'v4-source-upload.zip')) as archive:
        train_source = archive.read('scripts/train_marl.py')
        evaluation_source = archive.read('scripts/evaluate_marl.py')
        assert b'seed + episode' in train_source and b'distribution.sample()' in train_source
        assert b'.probs.argmax()' in evaluation_source
        for name, payload in [('scripts/train_marl.py', train_source),
                              ('scripts/evaluate_marl.py', evaluation_source)]:
            inputs['v4-source-upload.zip::'+name] = hashlib.sha256(payload).hexdigest()

    baseline = read_csv(source(root/'validation-baseline/validation-baseline.csv'))
    assert sorted(int(r['seed']) for r in baseline) == [21, 22, 23]
    crossing_names = list(json.loads(baseline[0]['pedestrian_p95_by_crossing']))
    assert len(crossing_names) == 12
    thresholds = {c: statistics.mean(json.loads(r['pedestrian_p95_by_crossing'])[c] for r in baseline)
                  for c in crossing_names}
    validation = []
    rewards = []
    slopes = []
    paths = sorted(root.glob('training/v4-400ep-seed*/*/checkpoint-metrics.csv'))
    assert len(paths) == 6
    configurations = set()
    for path in paths:
        match = re.fullmatch(r'(mappo-(?:vehicle|vehicle_pedestrian))-actions2-seed(\d+)', path.parent.name)
        assert match is not None
        family, seed_text = match.groups()
        seed = int(seed_text)
        configurations.add((family, seed))
        rows = read_csv(source(path))
        assert len(rows) == 24
        for episode in SNAPSHOTS:
            subset = [r for r in rows if int(r['training_episode']) == episode]
            assert sorted(int(r['seed']) for r in subset) == [21, 22, 23]
            assert all(int(r['duration_seconds']) == 3600 and int(r['training_seed']) == seed for r in subset)
            crossing_means = {c: statistics.mean(json.loads(r['pedestrian_p95_by_crossing'])[c] for r in subset)
                              for c in crossing_names}
            excess = {c: crossing_means[c]-thresholds[c] for c in crossing_names}
            worst = max(excess, key=excess.get)
            validation.append({'controller': family, 'training_seed': seed, 'episode': episode,
                'vehicle_time_loss_s_per_vehicle': statistics.mean(vehicle_loss(r) for r in subset),
                'pedestrian_mean_wait_s': statistics.mean(float(r['pedestrian_average_waiting_seconds']) for r in subset),
                'worst_crossing_p95_s': statistics.mean(max(json.loads(r['pedestrian_p95_by_crossing']).values()) for r in subset),
                'normalized_positive_crossing_violation': sum(max(excess[c], 0.0)/max(thresholds[c], 1.0) for c in crossing_names),
                'largest_crossing_excess_s': excess[worst], 'largest_excess_crossing': worst,
                'crossings_within_threshold': sum(v <= 0 for v in excess.values())})
        log = source(path.parent/'training.log')
        pairs = [(int(e), float(r)) for e, r in re.findall(r'episode=(\d+)/400 reward=([^\s]+)', log.read_text())]
        assert [e for e, _ in pairs] == list(range(1, 401))
        assert all(math.isfinite(r) for _, r in pairs)
        for episode in SNAPSHOTS:
            window = [r for e, r in pairs if episode-25 < e <= episode]
            assert len(window) == 25
            rewards.append({'controller': family, 'training_seed': seed, 'episode': episode,
                'trailing25_mean_reward': statistics.mean(window),
                'trailing25_sample_sd_reward': statistics.stdev(window)})
        for first in [250, 300, 350]:
            window = [(e, r) for e, r in pairs if first <= e <= 400]
            assert len(window) == 401-first
            slopes.append({'controller': family, 'training_seed': seed, 'first_episode': first,
                'last_episode': 400, 'episodes_in_fit': len(window), 'reward_slope_per_episode': slope(window)})
    assert configurations == {(f, s) for f in FAMILIES for s in SEEDS}

    selection = json.loads(source(root/'selected-checkpoints.json').read_text())
    for selected in selection:
        match = re.fullmatch(r'(mappo-(?:vehicle|vehicle_pedestrian))-actions2-seed(\d+)', selected['controller'])
        row = next(r for r in validation if r['controller'] == match[1] and
                   r['training_seed'] == int(match[2]) and r['episode'] == selected['episode'])
        assert math.isclose(row['normalized_positive_crossing_violation'], selected['normalized_violation'], abs_tol=1e-12)
        assert math.isclose(row['largest_crossing_excess_s'], selected['max_crossing_p95_violation'], abs_tol=1e-12)
    trajectory = []
    for family in FAMILIES:
        for episode in SNAPSHOTS:
            subset = [r for r in validation if r['controller'] == family and r['episode'] == episode]
            assert len(subset) == 3
            trajectory.append({'controller': family, 'episode': episode,
                **{key: statistics.mean(r[key] for r in subset) for key in VALIDATION_KEYS},
                'largest_excess_any_training_seed_s': max(r['largest_crossing_excess_s'] for r in subset)})
    old_trajectory = read_csv(source(root/'validation-trajectory.csv'))
    for row in trajectory:
        old = next(r for r in old_trajectory if r['controller'] == row['controller'] and int(r['episode']) == row['episode'])
        for new_key, old_key in [('vehicle_time_loss_s_per_vehicle', 'vehicle_mean_time_loss_seconds'),
                                 ('pedestrian_mean_wait_s', 'pedestrian_mean_wait_seconds'),
                                 ('worst_crossing_p95_s', 'worst_crossing_p95_seconds')]:
            assert math.isclose(row[new_key], float(old[old_key]), rel_tol=1e-12)
    slope_summary = []
    for family in FAMILIES:
        for first in [250, 300, 350]:
            values = [r['reward_slope_per_episode'] for r in slopes if r['controller'] == family and r['first_episode'] == first]
            slope_summary.append({'controller': family, 'first_episode': first, 'last_episode': 400,
                'independent_training_seeds': 3, **ci95(values)})

    primary = read_csv(source(root/'final-primary-results.csv'))
    traffic = []
    training_means = defaultdict(list)
    for seed in [41, 42, 43]:
        row = {'traffic_seed': seed}
        values = []
        for training_seed in SEEDS:
            name = f'mappo-vehicle-actions2-seed{training_seed}'
            matching = [r for r in primary if int(r['seed']) == seed and r['scenario_controller'] == name]
            assert len(matching) == 1
            value = vehicle_loss(matching[0])
            row[f'v4_training_seed_{training_seed}_loss_s_per_vehicle'] = value
            values.append(value)
            training_means[training_seed].append(value)
        row['v4_mean_across_training_seeds_s_per_vehicle'] = statistics.mean(values)
        baseline_rows = [r for r in primary if int(r['seed']) == seed and r['scenario_controller'] == 'pedestrian_actuated']
        assert len(baseline_rows) == 1
        row['pedestrian_actuated_loss_s_per_vehicle'] = vehicle_loss(baseline_rows[0])
        row['v4_minus_actuated_s_per_vehicle'] = row['v4_mean_across_training_seeds_s_per_vehicle']-row['pedestrian_actuated_loss_s_per_vehicle']
        traffic.append(row)
    uncertainty = [
        {'estimate': 'V4 vehicle', 'uncertainty_unit': 'training_seed_average_of_traffic_seeds',
         **ci95([statistics.mean(training_means[s]) for s in SEEDS])},
        {'estimate': 'V4 vehicle', 'uncertainty_unit': 'traffic_seed_average_of_training_seeds',
         **ci95([r['v4_mean_across_training_seeds_s_per_vehicle'] for r in traffic])},
        {'estimate': 'pedestrian_actuated', 'uncertainty_unit': 'traffic_seed',
         **ci95([r['pedestrian_actuated_loss_s_per_vehicle'] for r in traffic])},
        {'estimate': 'V4 minus actuated', 'uncertainty_unit': 'paired_traffic_seed_average_of_training_seeds',
         **ci95([r['v4_minus_actuated_s_per_vehicle'] for r in traffic])},
    ]
    assert round(uncertainty[0]['ci95_half_width'], 2) == 10.94
    assert round(uncertainty[2]['ci95_half_width'], 2) == 63.63
    late = [r for r in trajectory if r['controller'] == 'mappo-vehicle_pedestrian' and r['episode'] >= 300]
    monotonic = {key: all(b[key] <= a[key] for a, b in zip(late, late[1:])) for key in VALIDATION_KEYS}

    # New diagnostics are isolated; frozen inputs and original reports stay unchanged.
    assert output.resolve() != root.resolve()
    assert not output.resolve().is_relative_to((root/'training').resolve())
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in [('validation-by-run.csv', validation), ('validation-trajectory.csv', trajectory),
                       ('reward-trailing25-by-run.csv', rewards), ('reward-slopes-by-run.csv', slopes),
                       ('reward-slope-summary.csv', slope_summary), ('primary-by-traffic-seed.csv', traffic),
                       ('primary-uncertainty-by-unit.csv', uncertainty)]:
        write_csv(output/name, rows)
    result = {'source_sha256': inputs, 'original_manifest_files_unchanged': len(manifest),
        'training_runs': 6, 'reward_log_episodes': 2400, 'validation_rows': 144,
        'combined_late_validation_monotonic_decreasing': monotonic,
        'validation_trajectory': trajectory, 'reward_trailing25': rewards,
        'reward_slopes': slopes, 'reward_slope_summary': slope_summary,
        'primary_by_traffic_seed': traffic, 'primary_uncertainty': uncertainty,
        'training_seed_primary_means': {s: statistics.mean(training_means[s]) for s in SEEDS}}
    (output/'diagnostics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    report(result, output)
    for relative, digest in manifest.items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest() == digest, relative
    print(json.dumps({'original_artifact_files_unchanged': len(manifest),
        'combined_late_validation_monotonic_decreasing': monotonic,
        'reward_slopes_250_400': [r for r in slope_summary if r['first_episode'] == 250],
        'primary_uncertainty': uncertainty, 'report': str(output/'CONVERGENCE_REPORT.md')}))


def report(data, output):
    lines = ['# V4 convergence diagnostics', '',
        'Local saved-data analysis only. No cloud operations or simulations were performed. All six logs contain consecutive episodes 1..400. All six checkpoint-metrics files contain eight snapshots x three validation seeds = 144 rows.', '',
        '**Conclusion:** convergence by episode 400 is not established. Combined pedestrian validation metrics and crossing fairness improved across 300/350/400, but vehicle loss was not monotonic and the late training reward slope is not consistently positive across seeds. Further improvement after 400 is unobserved.', '',
        '## Validation trajectory', '',
        'For each snapshot, average validation seeds 21/22/23 within each training seed, then average training seeds 100/200/300. Vehicle loss divides total loss by departed vehicles separately in each evaluation. Worst crossing p95 is the maximum of 12 crossing p95s within each evaluation, then averaged.', '',
        'For each training seed, crossing p95s are first averaged over validation seeds. Relative to the corresponding actuated crossing means b_c, normalized violation = sum_c max(p95_c-b_c,0)/max(b_c,1s). Largest excess = max_c(p95_c-b_c). The table averages these two quantities across training seeds, preserving the original selector order of operations. They are not calculated after pooling policies. Lower is better for every column.', '']
    for family in FAMILIES:
        lines += [f'### {family}', '', '| Episode | Loss s/veh | Mean ped wait s | Worst crossing p95 s | Normalized violation | Mean largest crossing excess s |', '|---:|---:|---:|---:|---:|---:|']
        for row in data['validation_trajectory']:
            if row['controller'] == family:
                lines.append('| '+str(row['episode'])+' | '+' | '.join(f'{row[k]:.4f}' for k in VALIDATION_KEYS)+' |')
        lines.append('')
    lines += ['## Trailing 25-episode rewards', '', 'Each cell is mean +/- sample standard deviation of the last 25 raw episode rewards, within one training seed. These are SDs, not confidence intervals. Higher (less negative) reward is better. Reward magnitudes differ between objectives and should not be compared across them.', '']
    for family in FAMILIES:
        lines += [f'### {family}', '', '| Episode | Training seed 100 | Training seed 200 | Training seed 300 |', '|---:|---:|---:|---:|']
        for episode in SNAPSHOTS:
            cells = []
            for seed in SEEDS:
                row = next(r for r in data['reward_trailing25'] if (r['controller'], r['episode'], r['training_seed']) == (family, episode, seed))
                cells.append(f"{row['trailing25_mean_reward']:.2f} +/- {row['trailing25_sample_sd_reward']:.2f}")
            lines.append('| '+str(episode)+' | '+' | '.join(cells)+' |')
        lines.append('')
    lines += ['## Reward slopes', '', 'Fit raw reward against episode with an intercept independently for each training seed. The primary window 250..400 includes 151 observations per fit. Form the 95% Student-t interval from the three seed slopes: mean +/- 4.3026527299*sample_SD/sqrt(3), df=2. Episodes are not treated as 151 independent training replicates. The CI describes variation in fitted slopes across training seeds. Serial dependence within a run and nonlinear changes remain possible.', '',
        '| Controller | Window | Seed 100 | Seed 200 | Seed 300 | Mean slope | 95% CI |', '|---|---|---:|---:|---:|---:|---|']
    for summary in data['reward_slope_summary']:
        values = [next(r['reward_slope_per_episode'] for r in data['reward_slopes'] if r['controller'] == summary['controller'] and r['first_episode'] == summary['first_episode'] and r['training_seed'] == seed) for seed in SEEDS]
        lines.append('| '+summary['controller']+' | '+str(summary['first_episode'])+'..400 | '+' | '.join(f'{v:.4f}' for v in values)+f" | {summary['mean']:.4f} | [{summary['ci95_low']:.4f}, {summary['ci95_high']:.4f}] |")
    lines += ['', 'Slope units are reward units per episode. Positive means improvement. Both primary intervals span zero, so neither a persistent upward reward trend nor equivalence to a flat plateau is established with three seeds. The 300..400 and 350..400 fits are descriptive window sensitivity checks, not additional independent tests.', '',
        '## Primary vehicle loss by traffic seed', '', '| Traffic seed | V4 train 100 | V4 train 200 | V4 train 300 | V4 policy average | Actuated | Paired V4-actuated |', '|---:|---:|---:|---:|---:|---:|---:|']
    for row in data['primary_by_traffic_seed']:
        lines.append('| '+str(row['traffic_seed'])+' | '+' | '.join(f'{v:.4f}' for k, v in row.items() if k != 'traffic_seed')+' |')
    lines += ['', 'All cells are seconds per departed vehicle. V4 policy average averages the three learned training seeds for that same traffic seed. Test seeds 41/42/43 were reused from V3 and held out of V4 training and selection.', '',
        '| Estimate | Uncertainty unit | Mean | 95% CI half-width |', '|---|---|---:|---:|']
    for row in data['primary_uncertainty']:
        lines.append(f"| {row['estimate']} | {row['uncertainty_unit']} | {row['mean']:.4f} | {row['ci95_half_width']:.4f} |")
    lines += ['', 'The original +/- 10.94 summarizes three training-seed means, each averaging three traffic seeds. The actuated +/- 63.63 summarizes three traffic seeds directly. Their widths therefore measure different sources of variability. For a shared traffic-seed unit, V4 is 178.14 +/- 21.75 versus actuated 178.03 +/- 63.63. Averaging three V4 policies can smooth policy-specific traffic responses. This does not prove a population variance advantage. The paired mean difference is 0.11 +/- 61.18 seconds per vehicle.', '',
        '## Interpretation', '',
        'Combined 300/350/400 mean pedestrian wait, worst-crossing p95, normalized crossing violation and mean largest crossing excess all decrease. Vehicle loss first rises at 350 before falling at 400. Normalized violation decreases for training seeds 100/200 and is flat then improves for 300. Largest crossing excess rises from 350 to 400 for seed 100 (67.83 to 89.97s), despite improvement in its summed normalized violation. Thus aggregate progress does not mean every crossing or every metric improved in every run.', '',
        'Training samples stochastic actions and changes simulation seed each episode (training_seed + zero_based_episode). Validation uses deterministic argmax actions on fixed seeds 21/22/23. Training reward also measures waiting costs, urgency and switching, while validation separately reports vehicle time loss and pedestrian tails. These differences can produce divergent reward and validation trends.', '',
        'The six frozen choices remain vehicle 200/50/100 and combined 150/400/400 for training seeds 100/200/300. Selection uses a fairness constraint/fallback, so a late selection is not by itself proof of an unconverged reward objective. Vehicle validation is oscillatory. Combined shows substantial late pedestrian/fairness validation improvement, but training reward is noisy and has no consistently positive late slope. The saved data support possible remaining headroom, not a claim that longer training will reliably help or that the objective has converged.', '',
        '## Sources and verification', '',
        f"All {data['original_manifest_files_unchanged']} files in the frozen V4 local artifact manifest were hash-verified before and after this analysis. Original logs, metrics, checkpoints, frozen selections, reports and result bundles were not overwritten. Input SHA256s and unrounded calculations are in diagnostics.json.", '']
    lines += ['- '+path for path in data['source_sha256']]
    (output/'CONVERGENCE_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('scenario/generated/final-v4-results'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    main(args.root, args.output or args.root/'convergence-diagnostics')
