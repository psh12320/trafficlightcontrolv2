"""Summarize saved V4 trajectories without changing checkpoint selection."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import statistics


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(root):
    baseline = read_csv(root/'validation-baseline/validation-baseline.csv')
    thresholds = {crossing: statistics.mean(json.loads(r['pedestrian_p95_by_crossing'])[crossing]
                  for r in baseline) for crossing in json.loads(baseline[0]['pedestrian_p95_by_crossing'])}
    validation = []
    rewards = []
    for path in sorted((root/'training').glob('v4-400ep-seed*/*/checkpoint-metrics.csv')):
        controller = path.parent.name
        family, seed = re.fullmatch(r'(mappo-(?:vehicle|vehicle_pedestrian))-actions2-seed(\d+)', controller).groups()
        rows = read_csv(path)
        for episode in range(50, 401, 50):
            selected = [r for r in rows if int(r['training_episode']) == episode]
            assert len(selected) == 3
            p95 = {c: statistics.mean(json.loads(r['pedestrian_p95_by_crossing'])[c] for r in selected)
                   for c in thresholds}
            validation.append({'controller': family, 'training_seed': int(seed), 'episode': episode,
                'vehicle_mean_time_loss_seconds': statistics.mean(float(r['vehicle_time_loss_seconds']) /
                    float(r.get('vehicle_departed') or r['vehicle_arrivals']) for r in selected),
                'pedestrian_mean_wait_seconds': statistics.mean(float(r['pedestrian_average_waiting_seconds']) for r in selected),
                'worst_crossing_p95_seconds': statistics.mean(max(json.loads(r['pedestrian_p95_by_crossing']).values()) for r in selected),
                'crossings_meeting_validation_threshold': sum(p95[c] <= thresholds[c] for c in thresholds)})
        pairs = [(int(e), float(r)) for e, r in re.findall(r'episode=(\d+)/400 reward=([^\s]+)', (path.parent/'training.log').read_text())]
        assert [e for e, _ in pairs] == list(range(1, 401))
        for episode in [25, 50, 100, 150, 200, 250, 300, 350, 400]:
            values = [r for e, r in pairs if episode-25 < e <= episode]
            rewards.append({'controller': family, 'training_seed': int(seed), 'episode': episode,
                'rolling25_mean': statistics.mean(values), 'rolling25_stddev': statistics.stdev(values)})
    assert len(validation) == 48 and len(rewards) == 54
    write_csv(root/'validation-trajectory-by-run.csv', validation)
    write_csv(root/'reward-rolling25-by-run.csv', rewards)
    groups = defaultdict(list)
    for r in validation:
        groups[(r['controller'], r['episode'])].append(r)
    means = [{'controller': family, 'episode': episode, **{key: statistics.mean(r[key] for r in rows)
              for key in rows[0] if key not in ['controller', 'training_seed', 'episode']}}
             for (family, episode), rows in sorted(groups.items())]
    write_csv(root/'validation-trajectory.csv', means)
    print(json.dumps({'validation_runs': len(validation), 'validation_means': len(means), 'reward_windows': len(rewards)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
