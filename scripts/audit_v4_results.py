"""Verify V4 protocol, baseline reuse, frozen selections and prior preservation."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import torch

ROOT=Path(__file__).resolve().parents[1]


def read_csv(p):
    with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def main(root):
    prior=json.loads((root/'PRIOR_ARTIFACT_MANIFEST.json').read_text())
    for rel,digest in prior.items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
    markers=list((root/'training').glob('v4-400ep-seed*/*/training-complete.json'))
    assert len(markers)==6
    validation_rows=0
    for p in markers:
        d=json.loads(p.read_text());seed=d['training_seed']
        assert seed in [100,200,300] and d['algorithm']=='mappo'
        assert d['episodes']==400 and d['horizon']==3600
        assert d['randomize_demand'] is False and d['switch_penalty']==3
        assert not (p.parent/'demand-schedule.jsonl').exists()
        log=(p.parent/'training.log').read_text()
        assert [int(x) for x in re.findall(r'episode=(\d+)/400 reward=',log)]==list(range(1,401))
        assert sorted(int(x.stem.split('-')[1]) for x in (p.parent/'snapshots').glob('episode-*.pt'))==list(range(50,401,50))
        rows=read_csv(p.parent/'checkpoint-metrics.csv');assert len(rows)==24
        for episode in range(50,401,50):
            subset=[x for x in rows if int(x['training_episode'])==episode]
            assert sorted(int(x['seed']) for x in subset)==[21,22,23]
            assert all(int(x['duration_seconds'])==3600 and int(x['vehicle_period'])==6
                       and int(x['pedestrian_interval'])==20 and int(x['action_size'])==2 for x in subset)
        provenance=json.loads((p.parent.parent/'provenance.json').read_text())
        assert '--randomize-demand' not in provenance['command']
        assert provenance['command'][provenance['command'].index('--episodes')+1]=='400'
        validation_rows+=len(rows)
    selected=json.loads((root/'selected-checkpoints-local.json').read_text());assert len(selected)==6
    for row in selected:
        p=Path(row['checkpoint']);assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
        saved=torch.load(p,map_location='cpu',weights_only=True);m=saved['metadata']
        assert saved['algorithm']=='mappo' and m['action_size']==2 and m['observation_size']==25
        assert m['randomize_demand'] is False and m['switch_penalty']==3
        assert m['vehicle_scale']==20.8 and m['pedestrian_scale']==20
        assert m['urgency_coefficient']==0.5 and m['urgency_wait_scale']==60
        assert m['vehicle_period_choices']==[6] and m['pedestrian_interval_choices']==[20]
        assert m['approach_vehicle_density_scale']==25
        assert m['reward_objective_version']=='v4-upstream-calibrated-switching'
        assert all(a['network.0.weight'].shape[1]==25 for a in saved['actors'])
        assert saved['critics'][0]['network.0.weight'].shape[1]==75
    raw=[]
    for name in ['final-primary-results.csv','final-demand-sweep.csv','final-slow-pedestrian-results.csv']:
        raw.extend(read_csv(root/name))
    assert len(raw)==96
    for scenario,people,speed,period,interval in [('primary',2148,1.4,6,20),('low',1428,1.4,10,30),('high',3588,1.4,4,12),('slow',2148,0.9,6,20)]:
        rows=[r for r in raw if r['scenario']==scenario]
        assert len(rows)==24 and len({r['scenario_controller'] for r in rows})==8
        for r in rows:
            assert int(r['duration_seconds'])==3600
            assert int(r.get('pedestrian_departed') or r.get('pedestrian_arrivals'))==people
            assert float(r['pedestrian_speed_meters_per_second'])==speed
            assert int(r['vehicle_period'])==period and int(r['pedestrian_interval'])==interval
            c=json.loads(r['pedestrian_p95_by_crossing']);assert len(c)==12
            assert all(math.isfinite(v) and v>=0 for v in c.values())
            assert float(r['pedestrian_max_waiting_seconds'])>=max(c.values())-1e-9
        for name in {r['scenario_controller'] for r in rows}:
            assert sorted(int(r['seed']) for r in rows if r['scenario_controller']==name)==[41,42,43]
    reuse=read_csv(root/'baseline-reuse.csv');assert len(reuse)==24
    for old in reuse:
        new=next(r for r in raw if all(r[k]==old[k] for k in ['scenario','scenario_controller','seed']))
        assert all(new.get(k,'')==v for k,v in old.items())
    provenance=json.loads((root/'BASELINE_REUSE_PROVENANCE.json').read_text())
    for rel,digest in provenance['source_csv_sha256'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest
    for rel,digest in provenance['copied_baseline_json_sha256'].items():
        assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest
    verified=0
    manifests=list((root/'training').glob('*/training-seed*-manifest.json'))+[root/'evaluation-manifest.json']
    assert len(manifests)==4
    for p in manifests:
        for rel,digest in json.loads(p.read_text()).items():
            assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest,rel
            verified+=1
    frozen=json.loads((root/'v4-status.json').read_text());assert frozen['stage']=='complete'
    assert frozen['selection_sha256']==hashlib.sha256((root/'selected-checkpoints.json').read_bytes()).hexdigest()
    report={'training_runs':6,'training_episodes':2400,'snapshots':48,'validation_checkpoint_evaluations':validation_rows,
        'selected_checkpoints':6,'new_policy_evaluations':72,'reused_baseline_rows':24,'accepted_rows':96,
        'validation_feasible_runs':sum(r['max_crossing_p95_violation']<=0 for r in selected),
        'verified_manifest_entries':verified,'prior_artifact_files_unchanged':len(prior),
        'fixed_primary_training_verified':True,'all_test_demands_matched':True,'baseline_reuse_exact':True,
        'selection_unchanged_during_test':True,'test_seed_status':'reused V3 seeds for matched exploratory comparison'}
    (root/'AUDIT_RESULTS.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
