"""Freeze V5 BC-or-PPO selections using validation seeds only."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import zipfile

if __package__:
    from .select_checkpoint import _mean_crossing_p95
else:
    from select_checkpoint import _mean_crossing_p95


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(root, remote_root):
    baseline = read_csv(root/'validation-baseline/validation-baseline.csv')
    assert sorted(int(r['seed']) for r in baseline) == [21, 22, 23]
    thresholds = _mean_crossing_p95(baseline)
    paths = sorted(root.glob('training/v5-250ep-seed*/*/checkpoint-metrics.csv'))
    assert len(paths) == 6
    records, local, all_candidates, gains = [], [], [], []
    for path in paths:
        rows = read_csv(path)
        assert len(rows) == 18
        candidates = []
        for episode in [0, 50, 100, 150, 200, 250]:
            subset = [r for r in rows if int(r['training_episode']) == episode]
            assert sorted(int(r['seed']) for r in subset) == [21, 22, 23]
            assert all(int(r['duration_seconds']) == 3600 and int(r['vehicle_period']) == 6 and int(r['pedestrian_interval']) == 20 for r in subset)
            crossing = _mean_crossing_p95(subset)
            excess = {c: crossing[c]-thresholds[c] for c in thresholds}
            name = 'episode-0-bc.pt' if episode == 0 else f'episode-{episode}.pt'
            checkpoint = path.parent/'snapshots'/name
            assert checkpoint.is_file()
            candidates.append({'controller': path.parent.name, 'episode': episode,
                'checkpoint': str(checkpoint), 'vehicle_time_loss': statistics.mean(float(r['vehicle_time_loss_seconds']) for r in subset),
                'max_crossing_p95_violation': max(excess.values()),
                'normalized_violation': sum(max(excess[c], 0)/max(thresholds[c], 1) for c in thresholds)})
        feasible = [r for r in candidates if r['max_crossing_p95_violation'] <= 0]
        selected = dict(min(feasible or candidates, key=lambda r: (r['vehicle_time_loss'] if feasible else r['normalized_violation'], r['vehicle_time_loss'])))
        original = Path(selected['checkpoint'])
        destination = root/'checkpoints'/path.parent.name/original.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        selected.update(sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
            training_checkpoint_relative=str(original.relative_to(root)).replace('\\', '/'),
            frozen_utc=datetime.now(timezone.utc).isoformat(),
            checkpoint=remote_root.rstrip('/')+'/'+str(destination.relative_to(root)).replace('\\', '/'))
        records.append(selected)
        local.append({**selected, 'remote_checkpoint': selected['checkpoint'], 'checkpoint': str(destination.resolve())})
        all_candidates.extend(candidates)
        bc = candidates[0]
        for candidate in candidates[1:]:
            gains.append({'controller': path.parent.name, 'rl_episode': candidate['episode'],
                'bc_vehicle_time_loss': bc['vehicle_time_loss'], 'rl_vehicle_time_loss': candidate['vehicle_time_loss'],
                'vehicle_time_loss_change_percent': 100*(candidate['vehicle_time_loss']/bc['vehicle_time_loss']-1),
                'bc_normalized_violation': bc['normalized_violation'], 'rl_normalized_violation': candidate['normalized_violation'],
                'normalized_violation_change': candidate['normalized_violation']-bc['normalized_violation']})
    (root/'selected-checkpoints.json').write_text(json.dumps(records, indent=2))
    (root/'selected-checkpoints-local.json').write_text(json.dumps(local, indent=2))
    write_csv(root/'validation-candidates.csv', all_candidates)
    write_csv(root/'bc-vs-rl-validation.csv', gains)
    with zipfile.ZipFile(root/'selected-upload.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(root/'selected-checkpoints.json', 'selected-checkpoints.json')
        for path in (root/'checkpoints').rglob('*.pt'):
            archive.write(path, str(path.relative_to(root)))
    print(json.dumps({'selected': len(records), 'selected_bc': sum(r['episode'] == 0 for r in records),
        'validation_feasible': sum(r['max_crossing_p95_violation'] <= 0 for r in records)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--remote-root', required=True)
    args = parser.parse_args()
    main(args.root, args.remote_root)
