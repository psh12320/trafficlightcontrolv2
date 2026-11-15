"""Validation baseline and frozen-checkpoint evaluation for the V2 experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from xml.etree import ElementTree

if __package__:
    from .build_network import GENERATED, build
    from .run_baseline import run
    from .run_marl_sweep import _write_rows
    from .run_selected_evaluations import evaluate_scenario
else:
    from build_network import GENERATED, build
    from run_baseline import run
    from run_marl_sweep import _write_rows
    from run_selected_evaluations import evaluate_scenario


def status(root: Path, stage: str, **extra):
    root.mkdir(parents=True, exist_ok=True)
    version = "v3" if "v3" in root.name else "v2"
    temp = root/(version+"-status.tmp")
    temp.write_text(json.dumps({"stage":stage,"updated_unix":time.time(),**extra},indent=2),encoding="utf-8")
    temp.replace(root/(version+"-status.json"))
    print(stage,flush=True)


def _baseline(task):
    seed, duration, root = task
    return run(seed,duration,root/f"pedestrian-actuated-{seed}.json",False,
               "pedestrian_actuated",37,1.4,6,20)


def baseline(args):
    status(args.root,"validation_baseline")
    build(37,1.4,6)
    directory = args.root/"validation-baseline"
    directory.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(_baseline,[(seed,args.duration,directory) for seed in (21,22,23)]))
    _write_rows(directory/"validation-baseline.csv",rows)
    status(args.root,"validation_baseline_complete")


def _combine(paths, output):
    rows=[]
    for path in paths:
        with path.open(newline="",encoding="utf-8") as file:
            rows.extend(csv.DictReader(file))
    _write_rows(output,rows)


def evaluate(args):
    records=json.loads((args.root/"selected-checkpoints.json").read_text(encoding="utf-8"))
    if len(records)!=12 or len({row['controller'] for row in records})!=12:
        raise ValueError("expected 12 distinct frozen checkpoints")
    selected=[]
    for row in records:
        checkpoint=Path(row['checkpoint'])
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError(f"selected checkpoint changed: {checkpoint}")
        selected.append((row['controller'],checkpoint))
    selection_hash=hashlib.sha256((args.root/"selected-checkpoints.json").read_bytes()).hexdigest()
    status(args.root,"frozen_selection_verified",selection_sha256=selection_hash)
    outputs={}
    for name,speed,period,interval in (("primary",1.4,6,20),("low",1.4,10,30),
                                      ("high",1.4,4,12),("slow",0.9,6,20)):
        status(args.root,"test_"+name)
        output=evaluate_scenario(name,selected,args.seeds,args.duration,speed,period,interval,args.root,args.workers)
        with output.open(newline="",encoding="utf-8") as file:
            rows=list(csv.DictReader(file))
        expected=12*sum(now%interval==0 for now in range(10,args.duration))
        if len(rows)!=(2+len(selected))*len(args.seeds) or any(int(row.get('pedestrian_departed') or row.get('pedestrian_arrivals'))!=expected for row in rows):
            raise ValueError(f"incomplete or mismatched demand in {name}")
        for controller,_ in [('fixed_time',None),('pedestrian_actuated',None)]+selected:
            if sorted(int(row['seed']) for row in rows if row['scenario_controller']==controller)!=sorted(args.seeds):
                raise ValueError(f"missing/duplicate test seed for {controller}")
        if name=='slow':
            programs=ElementTree.parse(GENERATED/'network.net.xml').getroot().findall('.//tlLogic')
            if {int(program.findall('phase')[index].get('duration')) for program in programs for index in (2,5)}!={8}:
                raise ValueError('slow scenario needs eight-second clearance')
        outputs[name]=output
    _combine([outputs['primary']],args.root/'final-primary-results.csv')
    _combine([outputs['low'],outputs['high']],args.root/'final-demand-sweep.csv')
    _combine([outputs['slow']],args.root/'final-slow-pedestrian-results.csv')
    if hashlib.sha256((args.root/'selected-checkpoints.json').read_bytes()).hexdigest()!=selection_hash:
        raise ValueError('checkpoint selections changed during testing')
    status(args.root,'complete',accepted_test_evaluations=4*(2+len(selected))*len(args.seeds),selection_sha256=selection_hash)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('baseline','evaluate'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--duration',type=int,default=3600)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--seeds',nargs='+',type=int,default=[11,12,13])
    args=parser.parse_args()
    try:
        (baseline if args.command=='baseline' else evaluate)(args)
    except BaseException as error:
        status(args.root,'failed',error=str(error))
        raise
