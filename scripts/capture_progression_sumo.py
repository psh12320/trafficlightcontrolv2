"""Capture seed-41 V2--V5 GUI replays using each archived implementation."""
import argparse
import atexit
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
from xml.etree import ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / 'report_figures/_support/progression'
CONFIGS = {
    2: ('reproducibility-source.zip', 'mappo-vehicle_pedestrian-actions4-seed100', 240, 6, 20),
    3: ('v3-source-upload.zip', 'ippo-vehicle_pedestrian-actions2-seed100', 260, 6, 20),
    4: ('v4-source-upload.zip', 'mappo-vehicle-actions2-seed100', 300, 6, 20),
    5: ('v5-source-upload.zip', 'mappo-vehicle-actions2-seed100', 400, 10, 30),
}


def main(version):
    import sumo
    import torch
    from PIL import Image
    torch.set_num_threads(1)
    home = Path(sumo.__file__).resolve().parent
    os.environ['PATH'] = str(home/'bin') + os.pathsep + os.environ['PATH']
    sys.path.insert(0, str(home/'tools'))
    import traci

    name, controller, duration, period, interval = CONFIGS[version]
    folder = ROOT/f'scenario/generated/final-v{version}-results'
    work = (SUPPORT/f'v{version}').resolve()
    work.mkdir(parents=True, exist_ok=True)
    source = work/'source'
    with zipfile.ZipFile(folder/name) as archive:
        for member in archive.namelist():
            assert (source/member).resolve().is_relative_to(source.resolve())
        archive.extractall(source)
    sys.path.insert(0, str(source/'scripts'))
    import train_marl as marl
    assert Path(marl.__file__).resolve().is_relative_to(source)
    selected = next(r for r in json.loads((folder/'selected-checkpoints-local.json').read_text(encoding='utf-8'))
                    if r['controller'] == controller)
    checkpoint = Path(selected['checkpoint'])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == selected['sha256']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    metadata = saved['metadata']
    action_size = metadata['action_size']
    obs_size = saved['actors'][0]['network.0.weight'].shape[1]
    actors = []
    for state in saved['actors']:
        actor = marl.Actor(action_size, obs_size) if 'observation_size' in inspect.signature(marl.Actor).parameters else marl.Actor(action_size)
        actor.load_state_dict(state); actor.eval(); actors.append(actor)
    network = work/'network'
    assert network.resolve().is_relative_to(SUPPORT.resolve())
    marl.build(37, 1.4, period, output_dir=network)
    kwargs = {'configuration': network/'simulation.sumocfg'}
    if 'frame_stack' in inspect.signature(marl.SignalEnvironment).parameters:
        kwargs['frame_stack'] = metadata.get('frame_stack', 1)
    environment = marl.SignalEnvironment(41, duration, metadata['reward_mode'],
        metadata.get('vehicle_weight', 1), metadata.get('pedestrian_weight', 1),
        metadata['switch_penalty'], metadata.get('observation_mode', 'full'),
        action_size, 37, 1.4, interval, **kwargs)
    environment.traci = traci
    marl._sumo_binary = lambda _: str(home/'bin/sumo-gui.exe')
    settings = ET.parse(ROOT/'report_figures/_support/publication.view.xml')
    settings.getroot().find('scheme').find('persons').set('person_exaggeration', '2.0')
    for viewport in list(settings.getroot().findall('viewport')):
        settings.getroot().remove(viewport)
    viewfile = work/'publication.view.xml'
    settings.write(viewfile, encoding='utf-8', xml_declaration=True)
    original_start = traci.start
    def gui_start(command, *args, **kwargs):
        return original_start(command + ['--gui-settings-file', str(viewfile), '--start',
            '--quit-on-end', '--window-size', '1400,900', '--delay', '0'], *args, **kwargs)
    traci.start = gui_start
    observation = environment.reset()
    version_info = traci.getVersion()
    def close():
        try: traci.close(False)
        except Exception: pass
    atexit.register(close)
    traci.gui.setSchema('View #0', 'publication'); traci.gui.setAngle('View #0', 0)
    traci.vehicletype.setColor('car', (0,114,178,255))
    traci.vehicletype.setColor('DEFAULT_PEDTYPE', (230,95,0,255))
    incoming = {j: sorted(set(l for l in traci.trafficlight.getControlledLanes(j) if not l.startswith(':')))
                for j in marl.AGENTS}
    snapshots, pending, trace, marked = {}, [], [], []
    pair_start, expert_queue = None, None
    previous_phase = {j: traci.trafficlight.getPhase(j) for j in marl.AGENTS}
    original_step = traci.simulationStep

    def state():
        people = [{'id': p, 'road': traci.person.getRoadID(p), 'next': traci.person.getNextEdge(p),
                   'speed': traci.person.getSpeed(p), 'xy': traci.person.getPosition(p)}
                  for p in traci.person.getIDList()]
        return {'time': int(traci.simulation.getTime()),
            'phase': {j: traci.trafficlight.getPhase(j) for j in marl.AGENTS},
            'signal_state': {j: traci.trafficlight.getRedYellowGreenState(j) for j in marl.AGENTS},
            'halts': {j: {l: traci.lane.getLastStepHaltingNumber(l) for l in incoming[j]} for j in marl.AGENTS},
            'approaching_counts': {j: {l: traci.lane.getLastStepVehicleNumber(l) for l in incoming[j]} for j in marl.AGENTS},
            'people': people,
            'highlighted_vehicles': {v: {'xy': traci.vehicle.getPosition(v), 'speed': traci.vehicle.getSpeed(v)}
                                     for v in marked if v in traci.vehicle.getIDList()}}

    def schedule(key, boundary, size):
        traci.gui.setAngle('View #0', 0); traci.gui.setBoundary('View #0', *boundary)
        path = work/(key+'.png')
        traci.gui.screenshot('View #0', str(path), width=size[0], height=size[1])
        pending.append((key, path, size))

    def gui_step(*args, **kwargs):
        nonlocal pair_start, expert_queue
        original_step(*args, **kwargs)
        current = state(); t = current['time']
        for key, path, size in pending:
            b = traci.gui.getBoundary('View #0'); width = b[1][0]-b[0][0]
            cx = (b[0][0]+b[1][0])/2; cy = (b[0][1]+b[1][1])/2
            height = width*size[1]/size[0]
            snapshots[key] = {**current, 'file': path.relative_to(ROOT/'report_figures').as_posix(),
                'pixels': size, 'world_boundary': [cx-width/2, cy-height/2, cx+width/2, cy+height/2],
                'angle_degrees': traci.gui.getAngle('View #0')}
            if version == 5 and expert_queue:
                lane = expert_queue['lane']
                snapshots[key]['queue_clearance'] = {**expert_queue, 'end_time': t,
                    'end_halts': current['halts']['J1'][lane], 'elapsed_seconds': t-expert_queue['start_time']}
        pending.clear()
        trace.append({'time': t, 'phase': current['phase'], 'halts': current['halts']})
        if version == 2 and not snapshots and not pending and t >= 100:
            curb = [p for p in current['people'] if p['road'].startswith(':J0_w') and p['speed'] < .1]
            if 'y' in current['signal_state']['J0'] and len(curb) >= 3 and traci.trafficlight.getNextSwitch('J0')-t > 1:
                schedule('v2_j0_clearance', (70,82,130,118), (2100,1260))
        elif version == 3 and 120 <= t <= 240:
            crossing = [p for p in current['people'] if p['road'].startswith(':J2_c') and p['speed'] > .1]
            best = len(snapshots.get('v3_j2_service', {}).get('crossing_people', []))
            if len(crossing) >= max(2, best+1) and current['phase']['J2'] in (0,3):
                schedule('v3_j2_service', (470,82,530,118), (2100,1260))
        elif version == 4:
            if pair_start is None and t >= 80:
                moving = [v for v in traci.vehicle.getIDList() if traci.vehicle.getLaneID(v) in
                          ('J0_J1_1','J1_J2_1','J1_J0_1','J2_J1_1') and traci.vehicle.getSpeed(v)>2]
                if len(moving) >= 3:
                    marked.extend(moving[:6]); pair_start = t+1
                    for v in marked: traci.vehicle.setColor(v, (204,51,153,255))
                    schedule('v4_corridor_before', (20,50,580,150), (2800,500))
            elif pair_start is not None and t == pair_start+9:
                schedule('v4_corridor_after', (20,50,580,150), (2800,500))
        elif version == 5 and t >= 80 and not snapshots and not pending:
            phase = current['phase']['J1']
            if phase in (0,3) and previous_phase['J1'] != phase:
                lanes = traci.trafficlight.getControlledLanes('J1'); signal = current['signal_state']['J1']
                legal = set(l for i,l in enumerate(lanes) if signal[i] in 'gG' and not l.startswith(':'))
                candidates = [(current['halts']['J1'].get(l, 0), l) for l in legal]
                if candidates and max(candidates)[0] >= 3:
                    count, lane = max(candidates)
                    expert_queue = {'lane': lane, 'start_time': t, 'start_halts': count, 'phase': phase}
            if expert_queue:
                if phase != expert_queue['phase'] or t-expert_queue['start_time'] > 30:
                    expert_queue = None
                elif t-expert_queue['start_time'] >= 4 and current['halts']['J1'][expert_queue['lane']] <= 1:
                    schedule('v5_low_queue_clearance', (270,82,330,118), (2100,1260))
        if 'v3_j2_service' in snapshots:
            snapshots['v3_j2_service']['crossing_people'] = [p for p in snapshots['v3_j2_service']['people'] if p['road'].startswith(':J2_c')]
        previous_phase.update(current['phase'])
        if t % 100 == 0: print(f'V{version}: t={t}', flush=True)

    traci.simulationStep = gui_step
    try:
        done = False
        while not done:
            with torch.no_grad():
                masks = environment.action_masks()
                actions = torch.stack([a.distribution(observation[i], masks[i]).probs.argmax() for i,a in enumerate(actors)])
            observation, _, done = environment.step(actions)
            if (version in (2,5) and snapshots) or (version == 4 and len(snapshots) == 2): break
    finally:
        traci.simulationStep = original_step
        environment.close(); traci.start = original_start
    expected = 2 if version == 4 else 1
    assert len(snapshots) == expected, (version, list(snapshots))
    for item in snapshots.values():
        with Image.open(ROOT/'report_figures'/item['file']) as image:
            assert image.size == tuple(item['pixels'])
        assert item['angle_degrees'] == 0
    net = ET.parse(network/'network.net.xml')
    geometry = {e.get('id'): {'shape': e.find('lane').get('shape'), 'function': e.get('function')}
                for e in net.findall('edge') if e.get('function') in ('crossing','walkingarea')}
    result = {'version': version, 'traffic_seed': 41, 'training_seed': 100,
        'controller': controller, 'selected_episode': selected['episode'],
        'checkpoint_sha256': selected['sha256'], 'source_zip_sha256': hashlib.sha256((folder/name).read_bytes()).hexdigest(),
        'observation_inputs': obs_size, 'checkpoint_metadata': metadata,
        'demand': {'pedestrian_speed': 1.4, 'vehicle_period': period, 'pedestrian_interval': interval},
        'person_display_exaggeration': 2, 'sumo_version': version_info[1],
        'snapshots': snapshots, 'geometry': geometry, 'trace': trace}
    (work/'capture_metadata.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'version': version, 'captures': {k:v['time'] for k,v in snapshots.items()}}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', type=int, choices=[2,3,4,5], required=True)
    main(parser.parse_args().version)
