"""Capture authentic local SUMO GUI panels without changing saved experiments."""
import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from xml.etree import ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main(out):
    import sumo
    home = Path(sumo.__file__).resolve().parent
    os.environ['PATH'] = str(home/'bin')+os.pathsep+os.environ['PATH']
    sys.path.insert(0, str(home/'tools'))
    import traci
    import torch
    torch.set_num_threads(1)
    support = out/'_support'
    capture = support/'captures'
    capture.mkdir(parents=True, exist_ok=True)
    replay = support/'v2-replay'
    replay.mkdir(exist_ok=True)
    archive = ROOT/'scenario/generated/final-v2-results/reproducibility-source.zip'
    with zipfile.ZipFile(archive) as z:
        z.extractall(replay)
    sys.path.insert(0, str(replay/'scripts'))
    import train_marl as legacy
    assert legacy.OBSERVATION_SIZE == 13
    assert Path(legacy.__file__).resolve().is_relative_to(replay.resolve())
    network_dir = (support/'capture-network').resolve()
    assert network_dir.is_relative_to(out.resolve())
    legacy.build(37, 1.4, 6, output_dir=network_dir)
    settings = ET.parse(home/'tools/game/A10KW/osm.view.xml')
    scheme = settings.getroot().find('scheme')
    scheme.set('name', 'publication')
    scheme.find('background').set('backgroundColor', '255,255,255')
    scheme.find('vehicles').set('vehicleMode', '0')
    scheme.find('vehicles').set('vehicle_exaggeration', '1.0')
    scheme.find('persons').set('personMode', '0')
    scheme.find('persons').set('person_exaggeration', '1.6')
    scheme.find('junctions').set('drawCrossingsAndWalkingareas', '1')
    scheme.find('legend').set('showSizeLegend', '0')
    # The example scheme includes a rotated, zoomed viewport for another map.
    # Remove it; all captures use an explicit north-up world-coordinate view.
    for viewport in list(settings.getroot().findall('viewport')):
        settings.getroot().remove(viewport)
    view_file = support/'publication.view.xml'
    settings.write(view_file, encoding='utf-8', xml_declaration=True)
    record = next(r for r in json.loads((ROOT/'scenario/generated/final-v2-results/selected-checkpoints-local.json').read_text(encoding='utf-8'))
                  if r['controller'] == 'mappo-vehicle-actions4-seed100')
    checkpoint = Path(record['checkpoint'])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == record['sha256']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    actors = [legacy.Actor(4) for _ in legacy.AGENTS]
    for actor, state in zip(actors, saved['actors']):
        actor.load_state_dict(state)
        actor.eval()
    metadata = saved['metadata']
    environment = legacy.SignalEnvironment(41, 2005, metadata['reward_mode'],
        metadata.get('vehicle_weight',1), metadata.get('pedestrian_weight',1),
        metadata.get('switch_penalty',1.5), metadata.get('observation_mode','full'),4,37,1.4,20,
        configuration=network_dir/'simulation.sumocfg')
    environment.traci = traci
    legacy._sumo_binary = lambda name: str(home/'bin/sumo-gui.exe')
    start = traci.start
    def gui_start(command, *args, **kwargs):
        return start(command+['--gui-settings-file',str(view_file),'--start','--quit-on-end',
            '--window-size','1600,1000','--delay','0'], *args, **kwargs)
    traci.start = gui_start
    observation = environment.reset()
    def cleanup_gui():
        try:traci.close(False)
        except Exception:pass
    atexit.register(cleanup_gui)
    traci.vehicletype.setColor('car',(0,114,178,255))
    if 'DEFAULT_PEDTYPE' in traci.vehicletype.getIDList():
        traci.vehicletype.setColor('DEFAULT_PEDTYPE',(230,95,0,255))
    views = {'wide':('View #0',(-20,-20,620,220),(3200,1400)),
             'j1':('View #0',(250,50,350,150),(2000,2000)),
             'j0':('View #0',(45,45,155,155),(2000,2000)),
             'j2':('View #0',(285,-10,615,210),(2700,1800))}
    traci.gui.setSchema('View #0','publication')
    traci.gui.setAngle('View #0', 0)
    network = ET.parse(network_dir/'network.net.xml')
    geometry = {'junctions':{j.get('id'):[float(j.get('x')),float(j.get('y'))] for j in network.findall('junction') if j.get('id') in legacy.AGENTS},
                'lanes':{e.get('id'):{'function':e.get('function'), 'shape':e.find('lane').get('shape'),
                                     'outline':e.find('lane').get('outlineShape')} for e in network.findall('edge') if e.get('function') in ['crossing','walkingarea']}}
    lanes = {j:list(dict.fromkeys(l for l in traci.trafficlight.getControlledLanes(j) if not l.startswith(':'))) for j in legacy.AGENTS}
    snapshots, pending, trace = {}, [], []
    yellow_steps = dict.fromkeys(legacy.AGENTS,0)
    yellow_entries = dict.fromkeys(legacy.AGENTS,0)
    previous_yellow = dict.fromkeys(legacy.AGENTS,False)
    last_step = traci.simulationStep
    def state():
        people = list(traci.person.getIDList())
        waiting = {j:[] for j in legacy.AGENTS}
        for person in people:
            if traci.person.getSpeed(person)>=0.1:continue
            road = traci.person.getRoadID(person)
            next_edge = traci.person.getNextEdge(person)
            if next_edge.startswith(':J') and '_c' in next_edge:
                waiting[next_edge[1:3]].append(person)
            elif road.startswith(':J') and '_w' in road:
                waiting[road[1:3]].append(person)
        return {'time':int(traci.simulation.getTime()),'phase':{j:traci.trafficlight.getPhase(j) for j in legacy.AGENTS},
            'signal_state':{j:traci.trafficlight.getRedYellowGreenState(j) for j in legacy.AGENTS},
            'halting_by_lane':{j:{l:traci.lane.getLastStepHaltingNumber(l) for l in lanes[j]} for j in legacy.AGENTS},
            'curb_waiting_count':{j:len(waiting[j]) for j in legacy.AGENTS},
            'active_people':len(people),'yellow_entries':dict(yellow_entries),'yellow_seconds':dict(yellow_steps)}
    def schedule(key,view_name):
        view,boundary,size = views[view_name]
        traci.gui.setAngle(view, 0)
        traci.gui.setBoundary(view,*boundary)
        path = capture/(key+'.png')
        traci.gui.screenshot(view,str(path),width=size[0],height=size[1])
        pending.append((key,view_name,path))
    peak = -1
    def gui_step(*args,**kwargs):
        nonlocal peak
        last_step(*args,**kwargs)
        current = state()
        for j in legacy.AGENTS:
            yellow = 'y' in current['signal_state'][j].lower()
            yellow_steps[j] += int(yellow)
            yellow_entries[j] += int(yellow and not previous_yellow[j])
            previous_yellow[j] = yellow
        for key,view_name,path in pending:
            view,_,size = views[view_name]
            boundary = traci.gui.getBoundary(view)
            # Offscreen screenshots use the requested aspect ratio. The GUI
            # boundary is square before offscreen resize; preserve horizontal FOV.
            cx=(boundary[0][0]+boundary[1][0])/2;cy=(boundary[0][1]+boundary[1][1])/2
            width=boundary[1][0]-boundary[0][0];height=width*size[1]/size[0]
            snapshots[key] = {**current,'file':str(path.relative_to(out)), 'image_size':size,
                'world_boundary':[cx-width/2,cy-height/2,cx+width/2,cy+height/2],
                'gui_boundary':boundary,'view_angle_degrees':traci.gui.getAngle(view)}
        pending.clear()
        t=current['time']
        trace.append({'time':t,**{j+'_halts':sum(current['halting_by_lane'][j].values()) for j in legacy.AGENTS},
                      **{j+'_phase':current['phase'][j] for j in legacy.AGENTS}})
        if t==119:
            schedule('overview_t120','wide')
        if t==120:
            schedule('j1_t120','j1')
        if 'j0_clearance' not in snapshots and t>=130 and 'y' in current['signal_state']['J0'].lower() and traci.trafficlight.getNextSwitch('J0')-t>1 and sum(current['halting_by_lane']['J0'].values())>=4 and current['curb_waiting_count']['J0']>=4:
            schedule('j0_clearance','j0')
        queue=sum(current['halting_by_lane']['J2'].values())
        if 1650<=t<=1950 and queue>peak:
            peak=queue;schedule('j2_peak','j2')
        if t%300==0:print('capture_time='+str(t),flush=True)
    traci.simulationStep = gui_step
    try:
        done=False
        while not done:
            with torch.no_grad():
                masks=environment.action_masks()
                actions=torch.stack([a.distribution(observation[i],masks[i]).probs.argmax() for i,a in enumerate(actors)])
            observation,_,done=environment.step(actions)
    finally:
        traci.simulationStep=last_step
        environment.close()
        traci.start=start
    assert set(snapshots)=={'overview_t120','j1_t120','j0_clearance','j2_peak'},snapshots.keys()
    from PIL import Image
    for d in snapshots.values():
        path=out/d['file']
        assert path.is_file() and path.stat().st_size>1000
        assert Image.open(path).size==tuple(d['image_size'])
    import csv
    with (support/'capture_trace.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(trace[0]));writer.writeheader();writer.writerows(trace)
    result={'seed':41,'demand':{'vehicle_period':6,'pedestrian_interval':20,'pedestrian_speed':1.4},
        'controller':'V2 MAPPO-Veh, training seed100, selected episode50', 'checkpoint_sha256':record['sha256'],
        'sumo_version':'1.28.0','person_visual_exaggeration':1.6,'snapshots':snapshots,'geometry':geometry,
        'network_phase_states':{x.get('id'):[p.attrib for p in x.findall('phase')] for x in network.findall('tlLogic')}}
    (support/'capture_metadata.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:{'time':v['time'],'halts':{j:sum(v['halting_by_lane'][j].values()) for j in legacy.AGENTS},'curb_waiting':v['curb_waiting_count']} for k,v in snapshots.items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'report_figures');a=p.parse_args();main(a.output.resolve())
