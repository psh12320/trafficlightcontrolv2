"""Select V2 checkpoints locally, then create a portable bundle for the retained pod."""
import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from datetime import datetime,timezone
from pathlib import Path

if __package__:
    from .select_checkpoint import select_from_metrics
else:
    from select_checkpoint import select_from_metrics


def main(root:Path,remote_root:str,training_prefix:str='v2-300ep',expected_runs:int=12,episodes:int=300):
    baseline=root/'validation-baseline/validation-baseline.csv'
    with baseline.open(newline='',encoding='utf-8') as file:
        if sorted(int(row['seed']) for row in csv.DictReader(file))!=[21,22,23]:
            raise ValueError('expected all three validation baselines')
    metrics=sorted((root/'training').glob(training_prefix+'-seed*/*/checkpoint-metrics.csv'))
    if len(metrics)!=expected_runs:
        raise ValueError(f'expected {expected_runs} training configurations, got {len(metrics)}')
    records=[]
    local_records=[]
    for path in metrics:
        with path.open(newline='',encoding='utf-8') as file:
            rows=list(csv.DictReader(file))
        if len(rows)!=(episodes//50)*3:
            raise ValueError('unexpected snapshot validation row count')
        for episode in range(50,episodes+1,50):
            matching=[row for row in rows if int(row['training_episode'])==episode]
            if sorted(int(row['seed']) for row in matching)!=[21,22,23]:
                raise ValueError(f'validation mismatch {path} episode {episode}')
            if any(int(row['vehicle_period'])!=6 or int(row['pedestrian_interval'])!=20 for row in matching):
                raise ValueError('validation demand was randomized')
        selection_file=path.parent/'selection.json'
        original=select_from_metrics(path,baseline,path.parent/'snapshots',list(range(50,episodes+1,50)),selection_file)
        selected=json.loads(selection_file.read_text())
        destination=root/'checkpoints'/selected['controller']/original.name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(original,destination)
        selected['sha256']=hashlib.sha256(destination.read_bytes()).hexdigest()
        selected['training_checkpoint_relative']=str(original.relative_to(root)).replace('\\','/')
        selected['frozen_utc']=datetime.now(timezone.utc).isoformat()
        selected['checkpoint']=remote_root.rstrip('/')+'/'+str(destination.relative_to(root)).replace('\\','/')
        records.append(selected)
        local_records.append({**selected,'remote_checkpoint':selected['checkpoint'],'checkpoint':str(destination.resolve())})
    (root/'selected-checkpoints.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    (root/'selected-checkpoints-local.json').write_text(json.dumps(local_records,indent=2),encoding='utf-8')
    with zipfile.ZipFile(root/'selected-upload.zip','w',zipfile.ZIP_DEFLATED) as archive:
        archive.write(root/'selected-checkpoints.json','selected-checkpoints.json')
        for path in (root/'checkpoints').rglob('*.pt'):
            archive.write(path,str(path.relative_to(root)))
    print(json.dumps({'selected':len(records),'validation_feasible':sum(row['max_crossing_p95_violation']<=0 for row in records),
                      'upload_bytes':(root/'selected-upload.zip').stat().st_size}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--remote-root',required=True)
    parser.add_argument('--training-prefix',default='v2-300ep')
    parser.add_argument('--expected-runs',type=int,default=12)
    parser.add_argument('--episodes',type=int,default=300)
    args=parser.parse_args()
    main(args.root,args.remote_root,args.training_prefix,args.expected_runs,args.episodes)
