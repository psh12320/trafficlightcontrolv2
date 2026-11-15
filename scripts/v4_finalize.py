"""Evaluate six frozen V4 MAPPO checkpoints with immutable reused V3 baselines."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from xml.etree import ElementTree

if __package__:
    from .build_network import GENERATED
    from .run_selected_evaluations import evaluate_scenario
    from .run_marl_sweep import _write_rows
else:
    from build_network import GENERATED
    from run_selected_evaluations import evaluate_scenario
    from run_marl_sweep import _write_rows


def status(root,stage,**extra):
    temporary=root/'v4-status.tmp'
    temporary.write_text(json.dumps({'stage':stage,'updated_unix':time.time(),**extra},indent=2))
    temporary.replace(root/'v4-status.json')
    print(stage,flush=True)


def main(args):
    selection=args.root/'selected-checkpoints.json'
    records=json.loads(selection.read_text());assert len(records)==6
    assert len({r['controller'] for r in records})==6
    selected=[]
    for r in records:
        p=Path(r['checkpoint']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
        selected.append((r['controller'],p))
    frozen=hashlib.sha256(selection.read_bytes()).hexdigest()
    with (args.root/'baseline-reuse.csv').open(newline='') as f:baselines=list(csv.DictReader(f))
    assert len(baselines)==24 and sorted(args.seeds)==[41,42,43]
    outputs={}
    for name,speed,period,interval in [('primary',1.4,6,20),('low',1.4,10,30),('high',1.4,4,12),('slow',0.9,6,20)]:
        reuse=[r for r in baselines if r['scenario']==name]
        assert len(reuse)==6
        status(args.root,'test_'+name,selection_sha256=frozen)
        output=evaluate_scenario(name,selected,args.seeds,3600,speed,period,interval,args.root,args.workers,baseline_rows=reuse)
        with output.open(newline='') as f:rows=list(csv.DictReader(f))
        assert len(rows)==24
        assert len(list((args.root/name).glob('*.json')))==24
        expected=12*sum(t%interval==0 for t in range(10,3600))
        for r in rows:
            assert int(r.get('pedestrian_departed') or r.get('pedestrian_arrivals'))==expected
            assert int(r['duration_seconds'])==3600
        for controller in {r['scenario_controller'] for r in rows}:
            assert sorted(int(r['seed']) for r in rows if r['scenario_controller']==controller)==[41,42,43]
        if name=='slow':
            assert {int(x.findall('phase')[i].get('duration')) for x in ElementTree.parse(GENERATED/'network.net.xml').findall('.//tlLogic') for i in (2,5)}=={8}
        outputs[name]=rows
    _write_rows(args.root/'final-primary-results.csv',outputs['primary'])
    _write_rows(args.root/'final-demand-sweep.csv',outputs['low']+outputs['high'])
    _write_rows(args.root/'final-slow-pedestrian-results.csv',outputs['slow'])
    assert hashlib.sha256(selection.read_bytes()).hexdigest()==frozen
    status(args.root,'complete',selection_sha256=frozen,accepted_rows=96,new_policy_evaluations=72,reused_baseline_rows=24)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16);p.add_argument('--seeds',nargs='+',type=int,default=[41,42,43]);args=p.parse_args()
    try:main(args)
    except BaseException as e:
        status(args.root,'failed',error=str(e));raise
