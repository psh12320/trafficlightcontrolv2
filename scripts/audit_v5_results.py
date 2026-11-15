"""Audit the accepted balanced-BC V5 protocol and frozen prior artifacts."""
import argparse
import base64
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import torch

ROOT = Path(__file__).resolve().parents[1]
EPISODES = [0, 50, 100, 150, 200, 250]


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def main(root):
    prior = json.loads((root/'PRIOR_ARTIFACT_MANIFEST.json').read_text())
    for rel, digest in prior.items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest, rel
    markers = list(root.glob('training/v5-250ep-seed*/*/training-complete.json'))
    assert len(markers) == 6
    validation_rows = 0
    bc_quality = []
    for path in markers:
        d = json.loads(path.read_text())
        assert d['episodes'] == 250 and d['horizon'] == 3600
        assert d['algorithm'] == 'mappo' and d['training_seed'] in [100, 200, 300]
        assert d['randomize_demand'] is False and d['switch_penalty'] == 3
        assert d['frame_stack'] == 3 and d['bc_episodes'] == 20 and d['learning_rate'] == 1e-4
        assert not (path.parent/'demand-schedule.jsonl').exists()
        text = (path.parent/'training.log').read_text()
        assert [int(x) for x in re.findall(r'episode=(\d+)/250 reward=', text)] == list(range(1, 251))
        assert [int(x) for x in re.findall(r'bc_episode=(\d+)/20', text)] == list(range(1, 21))
        assert [int(x) for x in re.findall(r'critic_warmup_episode=(\d+)/5', text)] == list(range(1, 6))
        assert {p.name for p in (path.parent/'snapshots').glob('*.pt')} == {'episode-0-bc.pt'} | {f'episode-{e}.pt' for e in EPISODES[1:]}
        rows = read_csv(path.parent/'checkpoint-metrics.csv')
        assert len(rows) == 18
        for episode in EPISODES:
            subset = [r for r in rows if int(r['training_episode']) == episode]
            assert sorted(int(r['seed']) for r in subset) == [21, 22, 23]
            assert all(int(r['duration_seconds']) == 3600 and int(r['vehicle_period']) == 6
                       and int(r['pedestrian_interval']) == 20 and int(r['action_size']) == 2 for r in subset)
        validation_rows += len(rows)
        provenance = json.loads((path.parent.parent/'provenance.json').read_text())
        command = provenance['command']
        for flag, value in [('--episodes', '250'), ('--bc-episodes', '20'), ('--frame-stack', '3'), ('--learning-rate', '1e-4')]:
            assert command[command.index(flag)+1] == value
        assert '--randomize-demand' not in command
        for name in ['train_marl.py', 'marl_bc.py', 'evaluate_marl.py', 'run_marl_sweep.py', 'run_baseline.py', 'build_network.py']:
            assert provenance['source_sha256'][name] == hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()
        report = json.loads((path.parent/'bc-metrics.json').read_text())
        assert report['bc_episodes'] == 20 and report['bc_transitions_per_agent'] == 14400
        assert report['bc_epochs'] == 15 and report['bc_learning_rate'] == 1e-3
        assert report['bc_sampling'] == 'balanced-legal-action-minibatches'
        assert report['critic_warmup_episodes'] == 5 and report['critic_warmup_epochs'] == 15
        assert report['critic_warmup_actor_parameters_unchanged'] is True
        assert len(report['bc_simulation_seeds']) == 20 and len(report['critic_warmup_simulation_seeds']) == 5
        assert not set(report['bc_simulation_seeds']+report['critic_warmup_simulation_seeds']) & {21, 22, 23, 41, 42, 43}
        for actor in report['actor_fits']:
            assert len(actor['cross_entropy_by_epoch']) == 15
            assert sum(sum(r) for r in actor['confusion_true_pred']) == 14400
            assert actor['expert_switches'] > 0 and actor['confusion_true_pred'][1][1] > 0
            assert 0 <= actor['switch_recall'] <= 1
        assert all(sum(sum(row) for row in matrix) == 3600 for matrix in report['bc_policy_rollout_confusion_by_agent'])
        for fit in report['critic_fits']:
            assert math.isfinite(fit['mse_before_warmup']) and math.isfinite(fit['mse_after_warmup'])
            assert fit['mse_after_warmup'] < fit['mse_before_warmup']
        bc_quality.append({'controller': path.parent.name, 'training_switch_recall': [a['switch_recall'] for a in report['actor_fits']]})
    selected = json.loads((root/'selected-checkpoints-local.json').read_text())
    assert len(selected) == 6
    for row in selected:
        path = Path(row['checkpoint'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        saved = torch.load(path, map_location='cpu', weights_only=True)
        m = saved['metadata']
        assert saved['algorithm'] == 'mappo' and m['action_size'] == 2
        assert m['frame_stack'] == 3 and m['base_observation_size'] == 25 and m['observation_size'] == 75
        assert m['centralized_critic_input_size'] == 225 and m['bc_episodes'] == 20
        assert m['bc_sampling'] == 'balanced-legal-action-minibatches' and m['critic_warmup_episodes'] == 5
        assert m['learning_rate'] == 1e-4 and m['switch_penalty'] == 3 and m['randomize_demand'] is False
        assert m['vehicle_scale'] == 20.8 and m['pedestrian_scale'] == 20
        assert m['urgency_coefficient'] == 0.5 and m['urgency_wait_scale'] == 60
        assert m['training_episode'] == row['episode'] and row['episode'] in EPISODES
        assert all(a['network.0.weight'].shape[1] == 75 for a in saved['actors'])
        assert saved['critics'][0]['network.0.weight'].shape[1] == 225
    raw = [r for n in ['final-primary-results.csv', 'final-demand-sweep.csv', 'final-slow-pedestrian-results.csv'] for r in read_csv(root/n)]
    assert len(raw) == 96
    for scenario, people, speed, period, interval in [('primary',2148,1.4,6,20), ('low',1428,1.4,10,30), ('high',3588,1.4,4,12), ('slow',2148,0.9,6,20)]:
        rows = [r for r in raw if r['scenario'] == scenario]
        assert len(rows) == 24 and len({r['scenario_controller'] for r in rows}) == 8
        for row in rows:
            assert int(row['duration_seconds']) == 3600
            assert int(row.get('pedestrian_departed') or row.get('pedestrian_arrivals')) == people
            assert float(row['pedestrian_speed_meters_per_second']) == speed
            assert int(row['vehicle_period']) == period and int(row['pedestrian_interval']) == interval
            crossing = json.loads(row['pedestrian_p95_by_crossing'])
            assert len(crossing) == 12 and all(math.isfinite(v) and v >= 0 for v in crossing.values())
        for controller in {r['scenario_controller'] for r in rows}:
            assert sorted(int(r['seed']) for r in rows if r['scenario_controller'] == controller) == [41, 42, 43]
    for old in read_csv(root/'baseline-reuse.csv'):
        new = next(r for r in raw if all(r[k] == old[k] for k in ['scenario', 'scenario_controller', 'seed']))
        assert all(new.get(k, '') == value for k, value in old.items())
    baseline_provenance = json.loads((root/'BASELINE_REUSE_PROVENANCE.json').read_text())
    for rel, digest in baseline_provenance['source_csv_sha256'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest
    for rel, digest in baseline_provenance['copied_baseline_json_sha256'].items():
        assert hashlib.sha256((root/rel).read_bytes()).hexdigest() == digest
    manifests = list(root.glob('training/*/training-seed*-manifest.json'))+[root/'evaluation-manifest.json']
    assert len(manifests) == 4
    for tag in ['seed100','seed200','seed300','evaluation']:
        receipt = json.loads((root/f'RECEIPT-{tag}.json').read_text())
        payload = base64.b64decode((root/f'download-{tag}.b64').read_text())
        assert len(payload) == receipt['pack']['archive_bytes']
        assert hashlib.sha256(payload).hexdigest() == receipt['pack']['archive_sha256']
    verified = 0
    for path in manifests:
        for rel, digest in json.loads(path.read_text()).items():
            assert hashlib.sha256((root/rel).read_bytes()).hexdigest() == digest, rel
            verified += 1
    status = json.loads((root/'v5-status.json').read_text())
    assert status['stage'] == 'complete'
    assert status['selection_sha256'] == hashlib.sha256((root/'selected-checkpoints.json').read_bytes()).hexdigest()
    bc_reference = read_csv(root/'bc-primary-results.csv')
    assert len(bc_reference) == 18 and status['bc_reference_rows'] == 18
    assert status['bc_reference_new_evaluations'] == 3*sum(r['episode'] != 0 for r in selected)
    assert status['bc_reference_reused_selected_rows'] == 3*sum(r['episode'] == 0 for r in selected)
    assert len(list((root/'bc-primary').glob('*.json'))) == 18
    for controller in {r['scenario_controller'] for r in bc_reference}:
        assert sorted(int(r['seed']) for r in bc_reference if r['scenario_controller'] == controller) == [41,42,43]
    for r in bc_reference:
        assert int(r['duration_seconds']) == 3600 and r['checkpoint'].endswith('episode-0-bc.pt')
        assert int(r['vehicle_period']) == 6 and int(r['pedestrian_interval']) == 20
        assert int(r.get('pedestrian_departed') or r.get('pedestrian_arrivals')) == 2148
    patch = json.loads((root/'EVALUATION_SOURCE_PATCH.json').read_text())
    assert patch['v5_finalize_sha256'] == hashlib.sha256((ROOT/'scripts/v5_finalize.py').read_bytes()).hexdigest()
    report = {'training_runs':6, 'accepted_rl_episodes':1500, 'accepted_bc_episodes':120,
        'accepted_critic_warmup_episodes':30, 'snapshots_including_bc':36,
        'validation_evaluations':validation_rows, 'selected_checkpoints':6,
        'bc_checkpoints_selected':sum(r['episode'] == 0 for r in selected),
        'validation_feasible_runs':sum(r['max_crossing_p95_violation'] <= 0 for r in selected),
        'new_policy_test_evaluations':72, 'reused_baseline_rows':24, 'accepted_test_rows':96,
        'primary_bc_reference_rows':18, 'additional_bc_reference_evaluations':status['bc_reference_new_evaluations'],
        'reused_selected_bc_reference_rows':status['bc_reference_reused_selected_rows'],
        'verified_manifest_entries':verified, 'prior_artifact_files_unchanged':len(prior),
        'verified_transfer_archive_sha256_count':4,
        'baseline_reuse_exact':True, 'selection_unchanged_during_test':True,
        'discarded_uniform_bc_attempt_excluded':True, 'bc_quality':bc_quality}
    (root/'AUDIT_RESULTS.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('scenario/generated/final-v5-results'))
    main(p.parse_args().root)
