"""Generate the five 300-DPI publication figures from frozen local artifacts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'scenario/generated'
FAMILIES=['fixed_time','pedestrian_actuated','ippo-vehicle','ippo-vehicle_pedestrian','mappo-vehicle','mappo-vehicle_pedestrian']
LABELS=['Fixed-Time','Pedestrian-Actuated','IPPO-Veh','IPPO-VehPed','MAPPO-Veh','MAPPO-VehPed']
COLORS=['#777777','#009E73','#56B4E9','#E69F00','#0072B2','#D55E00']
SCENARIOS=['primary','low','high','slow']
TITLES=['Primary','Low demand','High demand','Slow pedestrians (0.9 m/s)']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
    'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,
    'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white',
    'savefig.facecolor':'white','axes.linewidth':0.7,'pdf.fonttype':42})


def read(path):
    with path.open(newline='',encoding='utf-8') as stream:return list(csv.DictReader(stream))


def save(fig,out,name):
    fig.savefig(out/name,dpi=300,facecolor='white')
    plt.close(fig)


def statistics_figures(out):
    v3=read(DATA/'final-v3-results/final-summary.csv')
    lookup={(r['scenario'],r['controller'],r['metric']):r for r in v3}
    fig=plt.figure(figsize=(15,10.5))
    outer=fig.add_gridspec(2,2,left=.155,right=.98,bottom=.12,top=.91,wspace=.36,hspace=.47)
    for i,(scenario,title) in enumerate(zip(SCENARIOS,TITLES)):
        inner=outer[i//2,i%2].subgridspec(1,2,wspace=.23)
        for m,(metric,heading,unit) in enumerate([
            ('vehicle_mean_time_loss_seconds','Vehicle time loss','s/vehicle'),
            ('worst_crossing_p95_waiting_seconds','Worst-crossing p95 wait','s')]):
            ax=fig.add_subplot(inner[0,m])
            for j,(family,color) in enumerate(zip(FAMILIES,COLORS)):
                r=lookup[(scenario,family,metric)];mean=float(r['mean']);half=float(r['ci95_half_width'])
                ax.errorbar(mean,5-j,xerr=half,fmt='o' if j<2 else 's',color=color,
                    ecolor=color,markersize=5.5,capsize=3,elinewidth=1.4,capthick=1.1,zorder=3)
            ax.set_yticks(range(6),list(reversed(LABELS)) if m==0 else ['']*6)
            ax.set_ylim(-.6,5.6);ax.set_xlabel(unit);ax.set_title(heading,pad=11,fontsize=10.5)
            ax.axvline(0,color='#999999',lw=.8,ls=':');ax.grid(axis='x',color='#E6E6E6',lw=.6)
            ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        pos=outer[i//2,i%2].get_position(fig)
        fig.text((pos.x0+pos.x1)/2,pos.y1+.053,f'({"abcd"[i]}) {title}',ha='center',fontsize=13,weight='bold')
    fig.text(.5,.055,'V3 test seeds 41–43; 3,600 s per evaluation. Points: means; whiskers: 95% Student-t CI (n = 3).',ha='center',fontsize=10)
    fig.text(.5,.031,'Learned: CI across training-seed means after traffic averaging. Baselines: CI across traffic seeds. Panel scales differ; intervals are untruncated.',ha='center',fontsize=9)
    save(fig,out,'fig3_scenario_comparison.png')

    crossing=read(DATA/'final-v3-results/per-crossing-p95.csv')
    chosen=['pedestrian_actuated','ippo-vehicle_pedestrian','mappo-vehicle','mappo-vehicle_pedestrian']
    names=[f':J{j}_c{c}' for j in range(3) for c in range(4)]
    grid=np.array([[float(next(r['mean_p95_seconds'] for r in crossing if (r['scenario'],r['controller'],r['crossing'])==('primary',family,c))) for family in chosen] for c in names])
    delta=100*(grid[:,1]/grid[:,0]-1)
    reduction=100*(1-grid[8:,1].mean()/grid[8:,0].mean())
    fig=plt.figure(figsize=(11.8,8))
    gs=fig.add_gridspec(1,2,left=.17,right=.96,top=.86,bottom=.18,width_ratios=[3.7,1.7],wspace=.18)
    ax=fig.add_subplot(gs[0,0]);im=ax.imshow(grid,cmap='YlOrRd',vmin=0,vmax=1000,aspect='auto')
    ax.set_xticks(range(4),['Pedestrian-\nActuated','IPPO-\nVehPed','MAPPO-\nVeh','MAPPO-\nVehPed'])
    ax.xaxis.tick_top();ax.tick_params(axis='x',length=0,pad=10)
    ax.set_yticks(range(12),[n+'  '+['N','E','S','W'][i%4] for i,n in enumerate(names)])
    ax.tick_params(axis='y',length=0,pad=9)
    for i in range(12):
        for j in range(4):ax.text(j,i,f'{grid[i,j]:.0f}',ha='center',va='center',fontsize=10,color='white' if grid[i,j]>620 else '#222222')
    for y in [3.5,7.5]:ax.axhline(y,color='white',lw=5)
    ax.add_patch(Rectangle((-.5,7.5),4,4,fill=False,edgecolor='#0072B2',lw=2.3,clip_on=False))
    for label in ax.get_yticklabels()[8:]:label.set_weight('bold');label.set_color('#005B8E')
    right=fig.add_subplot(gs[0,1],sharey=ax)
    right.barh(range(12),delta,color=['#0072B2' if i>=8 else '#B7B7B7' for i in range(12)],height=.66)
    right.axvline(0,color='#555555',lw=.8);right.tick_params(axis='y',left=False,labelleft=False)
    right.set_title('IPPO-VehPed change\nrelative to actuated',fontsize=10.5,pad=18)
    right.set_xlabel('Change (%)');right.grid(axis='x',color='#E6E6E6',lw=.6);right.set_axisbelow(True)
    lo=min(-40,float(delta.min())-12);hi=max(30,float(delta.max())+30);right.set_xlim(lo,hi)
    for i,d in enumerate(delta):right.text(d+(3 if d>=0 else -3),i,f'{d:+.1f}%',ha='left' if d<0 else 'left',va='center',fontsize=8.5,
        color='#005B8E' if i>=8 else '#555555',bbox={'facecolor':'white','alpha':.8,'edgecolor':'none','pad':.4})
    cax=fig.add_axes([.17,.116,.49,.022]);cbar=fig.colorbar(im,cax=cax,orientation='horizontal');cbar.set_label('Mean per-crossing p95 pedestrian wait (s)',fontsize=9,labelpad=5)
    fig.text(.5,.95,'Primary scenario: spatial concentration of pedestrian waiting',ha='center',fontsize=14,weight='bold')
    fig.text(.17,.035,f'J2 mean across four crossings: {grid[8:,0].mean():.1f} s → {grid[8:,1].mean():.1f} s  ({reduction:.1f}% lower with IPPO-VehPed)',fontsize=9.5,color='#005B8E',weight='bold')
    fig.text(.17,.014,'V3 means over training seeds 100/200/300 and test seeds 41–43. The J2 reduction does not imply a lower corridor-wide worst-crossing metric.',fontsize=8.2)
    save(fig,out,'fig4_per_crossing_p95.png')

    trajectory=read(DATA/'final-v4-results/convergence-diagnostics/validation-trajectory.csv')
    fig=plt.figure(figsize=(14,6.5))
    outer=fig.add_gridspec(1,2,left=.075,right=.98,top=.86,bottom=.20,width_ratios=[1.05,1.15],wspace=.29)
    left=outer[0,0].subgridspec(2,1,hspace=.22)
    axes=[fig.add_subplot(left[i,0]) for i in range(2)]
    for family,color,label in [('mappo-vehicle',COLORS[4],'MAPPO-Veh'),('mappo-vehicle_pedestrian',COLORS[5],'MAPPO-VehPed')]:
        rows=[r for r in trajectory if r['controller']==family]
        for ax,key in zip(axes,['vehicle_time_loss_s_per_vehicle','normalized_positive_crossing_violation']):
            ax.plot([int(r['episode']) for r in rows],[float(r[key]) for r in rows],'-o',color=color,lw=1.7,ms=4,label=label)
    axes[0].set_ylabel('Vehicle time loss\n(s/vehicle)');axes[1].set_ylabel('Normalized positive\ncrossing violation');axes[1].set_xlabel('Training episode')
    axes[0].tick_params(labelbottom=False);axes[0].legend(loc='upper right',frameon=False,ncol=2,fontsize=9)
    for ax in axes:ax.grid(color='#E4E4E4',lw=.6);ax.set_xticks(range(50,401,50))
    ax=fig.add_subplot(outer[0,1]);versions=['V2','V3','V4','V5'];x=np.arange(4)
    for offset,family,color,label,marker in [(-.21,'fixed_time',COLORS[0],'Fixed-Time','o'),(-.07,'pedestrian_actuated',COLORS[1],'Pedestrian-Actuated','o'),(.07,'mappo-vehicle',COLORS[4],'MAPPO-Veh','s'),(.21,'mappo-vehicle_pedestrian',COLORS[5],'MAPPO-VehPed','s')]:
        rows=[next(r for r in read(DATA/f'final-{v.lower()}-results/final-summary.csv') if (r['scenario'],r['controller'],r['metric'])==('primary',family,'vehicle_mean_time_loss_seconds')) for v in versions]
        ax.errorbar(x+offset,[float(r['mean']) for r in rows],yerr=[float(r['ci95_half_width']) for r in rows],fmt=marker+'-',color=color,lw=1,ms=5,capsize=3,label=label)
    ax.set_xticks(x,versions);ax.set_ylabel('Primary vehicle time loss (s/vehicle)');ax.grid(axis='y',color='#E4E4E4',lw=.6)
    ax.axvline(.5,color='#999999',lw=1,ls=':');ax.set_xlim(-.5,3.5);ax.legend(frameon=False,loc='upper right',fontsize=8.5)
    ax.set_ylim(bottom=0)
    fig.text(.29,.94,'(a) V4 validation trajectory',ha='center',fontsize=13,weight='bold')
    fig.text(.75,.94,'(b) Primary configuration comparison',ha='center',fontsize=13,weight='bold')
    fig.text(.075,.11,'Left: means over training seeds 100/200/300 and validation seeds 21–23; violations computed per training run before averaging.',fontsize=9)
    fig.text(.075,.075,'Right: mean ± 95% Student-t CI. V2 uses test seeds 11–13; V3–V5 use 41–43. Version-specific baseline records are shown.',fontsize=9)
    fig.text(.075,.04,'Versions change multiple design/training settings together. These are descriptive configuration comparisons, not isolated causal effects.',fontsize=9)
    save(fig,out,'fig5_ablation_progression.png')
    return {'j2_mean_reduction_percent':float(reduction)}


def points(shape):return np.array([[float(v) for v in p.split(',')] for p in shape.split()])


def screenshot_axis(ax,out,metadata,key):
    item=metadata['snapshots'][key];image=Image.open(out/item['file']);x0,y0,x1,y1=item['world_boundary']
    ax.imshow(image,extent=(x0,x1,y0,y1),origin='upper');ax.set_xlim(x0,x1);ax.set_ylim(y0,y1);ax.set_aspect('equal');ax.axis('off')
    return item


def label(ax,text,xy,xytext,color='#222222',size=10):
    ax.annotate(text,xy=xy,xytext=xytext,ha='center',va='center',fontsize=size,color=color,
        bbox=dict(boxstyle='round,pad=.25',facecolor='white',edgecolor=color,lw=.6,alpha=.95),
        arrowprops=dict(arrowstyle='-',color=color,lw=.85,shrinkA=3,shrinkB=3))


def crossing_labels(ax,metadata,junction,scale=1):
    center=metadata['geometry']['junctions'][junction]
    for c,(dx,dy) in enumerate([(0,18),(20,0),(0,-18),(-20,0)]):
        shape=points(metadata['geometry']['lanes'][f':{junction}_c{c}']['shape']);target=shape.mean(axis=0)
        label(ax,f'c{c}',target,(center[0]+dx*scale,center[1]+dy*scale),'#0072B2',8.5)


def screenshot_figures(out):
    metadata=json.loads((out/'_support/capture_metadata.json').read_text(encoding='utf-8'))
    fig=plt.figure(figsize=(14.5,5.6));gs=fig.add_gridspec(1,2,left=.025,right=.98,bottom=.19,top=.86,width_ratios=[2.35,1],wspace=.06)
    wide=fig.add_subplot(gs[0,0]);detail=fig.add_subplot(gs[0,1])
    a=screenshot_axis(wide,out,metadata,'overview_t120');b=screenshot_axis(detail,out,metadata,'j1_t120')
    for junction,xy in metadata['geometry']['junctions'].items():
        wide.text(xy[0]-28,xy[1]+32,junction,ha='center',fontsize=12,weight='bold',bbox=dict(fc='white',ec='none',alpha=.9,pad=2))
    wide.annotate('',xy=(300,20),xytext=(100,20),arrowprops=dict(arrowstyle='|-|',lw=1.1,color='#222222'))
    wide.text(200,4,'200 m',ha='center',fontsize=11,bbox=dict(fc='white',ec='none',pad=1))
    wide.add_patch(Rectangle((250,50),100,100,fill=False,ec='#0072B2',lw=1.3,ls='--'))
    crossing_labels(detail,metadata,'J1',.78)
    for edge,g in metadata['geometry']['lanes'].items():
        if edge.startswith(':J1_w'):
            shape=points(g['shape']);detail.add_patch(Polygon(shape,closed=True,fill=False,ec='#882255',lw=.9))
    label(detail,'Crosswalks', (300,105.2),(270,132),'#0072B2',9.5)
    walk=points(next(g['shape'] for e,g in metadata['geometry']['lanes'].items() if e.startswith(':J1_w'))).mean(axis=0)
    label(detail,'Walking Areas',walk,(328,68),'#882255',9.5)
    wide.set_title(f'(a) Three-junction corridor · t = {a["time"]} s',loc='left',fontsize=12,weight='bold',pad=14)
    detail.set_title(f'(b) J1 detail · t = {b["time"]} s',loc='left',fontsize=12,weight='bold',pad=14)
    fig.text(.035,.105,'Primary demand · seed 41 · V2 MAPPO-Veh replay (training seed 100). Blue vehicles; orange pedestrians.',fontsize=10)
    fig.text(.035,.06,'c0 / c1 / c2 / c3 = north / east / south / west crosswalks. Walking-area outlines follow SUMO geometry; pedestrians enlarged ×1.6 for visibility.',fontsize=9)
    save(fig,out,'fig1_sumo_corridor_overview.png')

    if __package__:
        from .compose_progression_figure import render_progression
    else:
        from compose_progression_figure import render_progression
    render_progression(out)


def verify(out):
    names=['fig1_sumo_corridor_overview.png','fig2_v1_to_v5_progression.png','fig3_scenario_comparison.png','fig4_per_crossing_p95.png','fig5_ablation_progression.png']
    records=[]
    for name in names:
        p=out/name
        if not p.exists():continue
        with Image.open(p) as im:
            assert im.width>=3000 and im.height>=1350
            assert all(abs(d-300)<.1 for d in im.info.get('dpi',(0,0)))
            assert np.asarray(im.convert('RGB')).std()>10
            records.append({'file':name,'pixels':im.size,'dpi':im.info['dpi'],'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    prior=json.loads((out/'_support/prior_artifact_manifest.json').read_text(encoding='utf-8'))
    for rel,digest in prior.items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
    (out/'figure_verification.json').write_text(json.dumps({'figures':records,'prior_files_unchanged':len(prior)},indent=2),encoding='utf-8')
    print(json.dumps(records))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'report_figures');p.add_argument('--statistics-only',action='store_true');p.add_argument('--screenshots-only',action='store_true');args=p.parse_args();out=args.output.resolve();out.mkdir(exist_ok=True)
    if not args.screenshots_only:statistics_figures(out)
    if not args.statistics_only:screenshot_figures(out)
    verify(out)
