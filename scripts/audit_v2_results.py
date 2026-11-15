"""Independently audit V2 demand schedules, checkpoints, manifests, and test rows."""
import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import torch


def main(root:Path,pilot:Path,version:str='v2'):
    v3 = version == 'v3'
    switch_penalty = 8.0 if v3 else 1.5
    plan = json.loads((root/'RUN_PLAN.json').read_text()) if v3 else {}
    test_seeds = plan['test_seeds'] if v3 else [11,12,13]
    assert len(test_seeds)==3 and len(set(test_seeds))==3
    assert not plan.get('test_seed_confirmation_pending',False)
    archive=json.loads((pilot/'ARCHIVE_MANIFEST.json').read_text())
    for relative,digest in archive.items():
        assert hashlib.sha256((pilot/relative).read_bytes()).hexdigest()==digest,relative
    training_rows=0
    configurations=sorted((root/'training').glob(version+'-300ep-seed*/*/training-complete.json'))
    assert len(configurations)==12
    for marker in configurations:
        data=json.loads(marker.read_text())
        seed=int(data['training_seed'])
        assert seed in (100,200,300)
        assert data['episodes']==300 and data['horizon']==3600
        assert data['randomize_demand'] is True and data['switch_penalty']==switch_penalty
        directory=marker.parent
        schedule=[json.loads(line) for line in (directory/'demand-schedule.jsonl').read_text().splitlines()]
        assert len(schedule)==300
        rng=random.Random(seed)
        for episode,row in enumerate(schedule):
            assert row=={'episode':episode+1,'simulation_seed':seed+episode,
                'vehicle_period':rng.choice((4,6,8)),'pedestrian_interval':rng.choice((12,20,30))}
            if v3:
                assert row['simulation_seed'] not in test_seeds
        assert {row['vehicle_period'] for row in schedule}=={4,6,8}
        assert {row['pedestrian_interval'] for row in schedule}=={12,20,30}
        assert sorted(int(path.stem.split('-')[1]) for path in (directory/'snapshots').glob('episode-*.pt'))==[50,100,150,200,250,300]
        with (directory/'checkpoint-metrics.csv').open(newline='',encoding='utf-8') as file:
            rows=list(csv.DictReader(file))
        assert len(rows)==18
        for episode in range(50,301,50):
            subset=[row for row in rows if int(row['training_episode'])==episode]
            assert sorted(int(row['seed']) for row in subset)==[21,22,23]
            for row in subset:
                assert int(row['duration_seconds'])==3600
                assert int(row['vehicle_period'])==6 and int(row['pedestrian_interval'])==20
                assert int(row['pedestrian_departed'])==2148
        training_rows+=len(rows)
    selected=json.loads((root/'selected-checkpoints-local.json').read_text())
    assert len(selected)==12
    for row in selected:
        checkpoint=Path(row['checkpoint'])
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==row['sha256']
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        metadata=saved['metadata']
        assert metadata['action_size']==(2 if v3 else 4) and metadata['randomize_demand'] is True
        assert metadata['switch_penalty']==switch_penalty
        assert metadata['vehicle_scale']==20.8 and metadata['pedestrian_scale']==(20 if v3 else 81.35)
        if v3:
            assert metadata['observation_size']==21
            assert metadata['urgency_coefficient']==0.5 and metadata['urgency_wait_scale']==60
            assert metadata['reward_objective_version']=='v3-fairness-calibrated'
            assert all(actor['network.0.weight'].shape[1]==21 for actor in saved['actors'])
        assert metadata['reward_mode'] in ('vehicle','vehicle_pedestrian')
        assert metadata['vehicle_period']==6 and metadata['pedestrian_interval']==20
        assert row['episode'] in (50,100,150,200,250,300)
    scenario_rows=[]
    for name in ('final-primary-results.csv','final-demand-sweep.csv','final-slow-pedestrian-results.csv'):
        with (root/name).open(newline='',encoding='utf-8') as file:
            scenario_rows.extend(csv.DictReader(file))
    for scenario,people,speed,period,interval in (('primary',2148,1.4,6,20),('low',1428,1.4,10,30),
                                                ('high',3588,1.4,4,12),('slow',2148,0.9,6,20)):
        rows=[row for row in scenario_rows if row['scenario']==scenario]
        assert len(rows)==42 and len({row['scenario_controller'] for row in rows})==14
        for row in rows:
            assert int(row['duration_seconds'])==3600
            assert int(row.get('pedestrian_departed') or row.get('pedestrian_arrivals'))==people
            assert float(row['pedestrian_speed_meters_per_second'])==speed
            assert int(row['vehicle_period'])==period and int(row['pedestrian_interval'])==interval
            crossing=json.loads(row['pedestrian_p95_by_crossing'])
            assert len(crossing)==12
            assert all(math.isfinite(float(value)) and float(value)>=0 for value in crossing.values())
        for controller in {row['scenario_controller'] for row in rows}:
            assert sorted(int(row['seed']) for row in rows if row['scenario_controller']==controller)==test_seeds
    manifests=list((root/'training').glob('*/training-seed*-manifest.json'))+[root/'evaluation-manifest.json']
    assert len(manifests)==4
    verified=0
    for manifest in manifests:
        for relative,digest in json.loads(manifest.read_text()).items():
            assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest,relative
            verified+=1
    frozen=json.loads((root/(version+'-status.json')).read_text())
    assert frozen['stage']=='complete'
    assert frozen['selection_sha256']==hashlib.sha256((root/'selected-checkpoints.json').read_bytes()).hexdigest()
    report={'training_runs':12,'training_episodes':3600,'validation_checkpoint_evaluations':training_rows,
        'selected_checkpoints':12,'accepted_test_evaluations':len(scenario_rows),
        'validation_feasible_runs':sum(row['max_crossing_p95_violation']<=0 for row in selected),
        'verified_manifest_entries':verified,'archived_prior_experiment_files_unchanged':len(archive),
        'all_demand_schedules_reproducible':True,'all_test_demands_matched':True,
        'selection_unchanged_during_test':True}
    (root/'AUDIT_RESULTS.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--pilot-root',type=Path,required=True)
    parser.add_argument('--version',choices=('v2','v3'),default='v2')
    args=parser.parse_args()
    main(args.root,args.pilot_root,args.version)
