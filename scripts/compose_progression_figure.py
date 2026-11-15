"""Compose the compact, provenance-labelled V1--V5 progression figure."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'report_figures'
SUPPORT = OUT/'_support/progression'
NEW_NAME = 'fig2_v1_to_v5_progression.png'


def read(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def shape_points(shape):
    return np.array([[float(x) for x in point.split(',')] for point in shape.split()])


def source_claims():
    """Compute every numeric annotation from the saved records, without writes."""
    diagnostics = json.loads((ROOT/'scenario/generated/v2-local-diagnostics/diagnostics.json').read_text(encoding='utf-8'))
    vehicle = next(r for r in diagnostics['rollouts'] if r['controller'] == 'mappo-vehicle')
    combined = next(r for r in diagnostics['rollouts'] if r['controller'] == 'mappo-vehicle_pedestrian')
    slots = vehicle['observation_slots']
    blind = 100*slots['outside_waiting_with_zero_observation']/slots['outside_waiting']
    j0 = next(j for j in combined['junctions'] if j.get('junction', j.get('agent')) == 'J0')
    phase = j0['phase_seconds']
    clearance = 100*sum(phase[str(i)] for i in [1,2,4,5])/combined['duration']
    yellow = 100*sum(phase[str(i)] for i in [2,5])/combined['duration']
    crossing = read(ROOT/'scenario/generated/final-v3-results/per-crossing-p95.csv')
    actuated = np.mean([float(r['mean_p95_seconds']) for r in crossing if r['scenario']=='primary' and r['controller']=='pedestrian_actuated' and r['crossing'].startswith(':J2_')])
    ippo = np.mean([float(r['mean_p95_seconds']) for r in crossing if r['scenario']=='primary' and r['controller']=='ippo-vehicle_pedestrian' and r['crossing'].startswith(':J2_')])
    ci = read(ROOT/'scenario/generated/final-v4-results/convergence-diagnostics/primary-uncertainty-by-unit.csv')
    traffic = next(r for r in ci if r['estimate']=='V4 vehicle' and r['uncertainty_unit']=='traffic_seed_average_of_training_seeds')
    baseline = next(r for r in ci if r['estimate']=='pedestrian_actuated')
    low = read(ROOT/'scenario/generated/final-v5-results/final-summary.csv')
    lookup = {r['metric']: float(r['mean']) for r in low if r['scenario']=='low' and r['controller']=='mappo-vehicle'}
    return {'v2_blind_curb_slots_percent': blind, 'v2_prior_diagnostic_traffic_seed': 11,
        'v2_j0_phase_transitions': j0['all_phase_transitions'], 'v2_j0_green_terminations': j0['green_terminations'],
        'v2_j0_clearance_phase_percent': clearance, 'v2_j0_yellow_percent': yellow,
        'v3_j2_mean_crossing_p95_reduction_percent': 100*(1-ippo/actuated),
        'v4_primary_loss_seconds': float(traffic['mean']),
        'v4_policy_averaged_traffic_ci_half_width': float(traffic['ci95_half_width']),
        'actuated_traffic_ci_half_width': float(baseline['ci95_half_width']),
        'v4_ci_ratio': float(baseline['ci95_half_width'])/float(traffic['ci95_half_width']),
        'v5_low_vehicle_loss_seconds': lookup['vehicle_mean_time_loss_seconds'],
        'v5_low_worst_crossing_p95_seconds': lookup['worst_crossing_p95_waiting_seconds']}


def screenshot(ax, metadata, key, out):
    item = metadata['snapshots'][key]
    x0,y0,x1,y1 = item['world_boundary']
    ax.imshow(Image.open(out/item['file']), extent=(x0,x1,y0,y1), origin='upper')
    ax.set_xlim(x0,x1); ax.set_ylim(y0,y1); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color('#D6D6D6'); spine.set_linewidth(.7)
    return item


def note(ax, text, target, position, color, fontsize=18):
    return ax.annotate(text, xy=target, xytext=position, ha='center', va='center',
        color=color, fontsize=fontsize,
        bbox={'boxstyle':'round,pad=.18','facecolor':'white','edgecolor':color,'lw':.6,'alpha':.95},
        arrowprops={'arrowstyle':'-','color':color,'lw':.9,'shrinkA':3,'shrinkB':2})


def render_progression(out=OUT):
    support = out/'_support/progression'
    plt.rcParams.update({'font.family':'DejaVu Sans', 'figure.facecolor':'white', 'savefig.facecolor':'white'})
    data = {v: json.loads((support/f'v{v}/capture_metadata.json').read_text(encoding='utf-8')) for v in [2,3,4,5]}
    assert all(d['traffic_seed']==41 and d['training_seed']==100 for d in data.values())
    claims = source_claims()
    fig = plt.figure(figsize=(16,4.5), dpi=300)
    xs = [.017,.264,.511,.758]; width = .225
    colors = ['#9A6700','#009E73','#0072B2','#882255']
    titles = ['(a) V1 / V2\nBlind curbs; chatter',
              '(b) V3\nWalking-area fix',
              '(c) V4\nUpstream lookahead',
              '(d) V5\nBC warm-start: Low']
    headers=[]
    for x,title,color in zip(xs,titles,colors):
        headers.append(fig.text(x,.967,title,fontsize=22,weight='bold',va='top',linespacing=1.13,color=color))
    for x in [.253,.500,.747]:
        fig.add_artist(plt.Line2D([x,x],[.105,.965],transform=fig.transFigure,color='#DDDDDD',lw=.7))

    a = fig.add_axes([xs[0],.375,width,.425])
    item = screenshot(a,data[2],'v2_j0_clearance',out)
    for edge,g in data[2]['geometry'].items():
        if edge.startswith(':J0_w'):
            a.add_patch(Polygon(shape_points(g['shape']),fill=False,ec='#C65D00',lw=1.25))
    note(a,'Yellow',(100,107.4),(82,113),'#9A6700')
    note(a,':J0_w*',(105.5,105.3),(117,87),'#C65D00')
    fig.text(xs[0]+width/2,.335,f'V2 MAPPO-VehPed · t = {item["time"]} s',ha='center',fontsize=15.5,color='#444444')

    b = fig.add_axes([xs[1],.375,width,.425])
    item = screenshot(b,data[3],'v3_j2_service',out)
    assert item['crossing_people'], 'V3 frame must show actual crossing pedestrians'
    for c,offset in enumerate([(0,12),(13,0),(0,-12),(-13,0)]):
        target = shape_points(data[3]['geometry'][f':J2_c{c}']['shape']).mean(axis=0)
        note(b,f'c{c}',target,(500+offset[0],100+offset[1]),'#007456',fontsize=17.5)
    fig.text(xs[1]+width/2,.335,f'V3 IPPO-VehPed · t = {item["time"]} s',ha='center',fontsize=15.5,color='#444444')

    first = data[4]['snapshots']['v4_corridor_before']['highlighted_vehicles']
    platoon = [v for v,s in first.items() if s['xy'][0]>300]
    for index,key in enumerate(['v4_corridor_before','v4_corridor_after']):
        ax = fig.add_axes([xs[2],.622-index*.193,width,.16])
        item = screenshot(ax,data[4],key,out)
        locations = [item['highlighted_vehicles'][v]['xy'][0] for v in platoon if v in item['highlighted_vehicles']]
        ax.add_patch(Rectangle((min(locations)-7,92),max(locations)-min(locations)+12,13,
            ec='#CC3399',fc=(.8,.2,.6,.10),lw=1.4))
        for j,x in enumerate([100,300,500]):
            ax.text(x,133,f'J{j}',ha='center',fontsize=18,weight='bold',color='#333333')
        ax.text(30,60,f't = {item["time"]} s',ha='left',fontsize=16,color='#444444',bbox={'fc':'white','ec':'none','pad':1})
    fig.text(xs[2]+width/2,.335,'Same magenta platoon · 10 s apart',ha='center',fontsize=14.5,color='#882255')

    d = fig.add_axes([xs[3],.375,width,.425])
    item = screenshot(d,data[5],'v5_low_queue_clearance',out)
    q = item['queue_clearance']
    d.text(.5,.045,f'{q["start_halts"]} → {q["end_halts"]} queued · {q["elapsed_seconds"]} s',
        transform=d.transAxes,ha='center',fontsize=18,color='#882255',bbox={'fc':'white','ec':'#882255','lw':.6,'pad':2})
    fig.text(xs[3]+width/2,.335,f'MAPPO-Veh · BC+PPO · {item["time"]} s',ha='center',fontsize=15.5,color='#444444')

    text = [f'Curb-blind slots: {claims["v2_blind_curb_slots_percent"]:.1f}%*\n{claims["v2_j0_phase_transitions"]} J0 phase changes*\n{claims["v2_j0_clearance_phase_percent"]:.1f}% clearance time*',
        f'J2 p95: −{claims["v3_j2_mean_crossing_p95_reduction_percent"]:.1f}%†\nMAPPO-Veh = fixed-time\nswitch_penalty = 8.0',
        f'25 features; cost 3.0\n{claims["v4_primary_loss_seconds"]:.1f} s ≈ actuated loss\n95% CI: {claims["v4_ci_ratio"]:.1f}× tighter‡',
        f'BC-initialized MAPPO\nLow loss: {claims["v5_low_vehicle_loss_seconds"]:.1f} s†\nWorst p95: {claims["v5_low_worst_crossing_p95_seconds"]:.1f} s†']
    boxes=[]
    for x,body in zip(xs,text):
        boxes.append(fig.text(x,.300,body,fontsize=20.5,va='top',linespacing=1.16,color='#222222'))
    fig.text(.017,.059,'Seed-41 replays (train 100): Primary in (a–c), Low in (d). Blue vehicles; orange pedestrians (×2 display size). Metrics describe saved aggregates.',fontsize=15,color='#555555')
    fig.text(.017,.012,'*Prior seed-11 diagnostics; clearance includes vehicle green. †J2 mean / test means. ‡Traffic-seed CI after policy averaging; descriptive comparison.',fontsize=15,color='#555555')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for box in headers+boxes:
        assert box.get_window_extent(renderer).width < width*fig.bbox.width+15, box.get_text()
    target = out/NEW_NAME
    fig.savefig(target, dpi=300, facecolor='white')
    plt.close(fig)
    preview = Image.open(target).convert('RGB')
    # 6.5-inch-wide proof at 150 pixels/inch: typical report text area.
    preview.resize((975,274),Image.Resampling.LANCZOS).save(support/'report_width_proof.png',dpi=(150,150))
    preview.resize((1920,540),Image.Resampling.LANCZOS).save(support/'preview.png')
    result = {'file': NEW_NAME, 'pixels': Image.open(target).size,
        'dpi': Image.open(target).info['dpi'], 'bytes': target.stat().st_size,
        'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'claims': claims,
        'report_width_inches':6.5, 'main_annotation_points_at_report_width':20.5*6.5/16,
        'capture_metadata': [str((support/f'v{v}/capture_metadata.json').relative_to(out)) for v in [2,3,4,5]]}
    with Image.open(target) as image:
        assert image.size == (4800,1350)
        assert all(abs(v-300)<.1 for v in image.info['dpi'])
        assert np.asarray(image.convert('RGB')).std()>10
    with Image.open(target) as image:
        image.verify()
    (support/'figure2_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    render_progression()
